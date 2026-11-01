// Package auditscan assembles trusted scanner journals into canonical Audit
// results. It never invokes scanners or accepts Worker-authored item identities.
package auditscan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
)

const (
	// The scan planner's Artifact adapter accepts at most 4 MiB per artifact.
	MaxResultBytes = planner.MaxScanArtifactBytes
	// Tool Worker scan reports have the same 1 MiB bound at observation time.
	MaxScannerReportBytes = planner.MaxScanReportBytes
	journalSchema         = "contractor.audit.scan-journal.v1"
	resultSummary         = "Scanner execution and observations are retained; security conclusions remain unverified."
)

// ErrInvalid classifies a deterministic failure: the trusted scan assignment
// differs from its execution manifest, the exact retained scan content is
// invalid or over its bound, or the assembled result breaks its own contract.
// The same input can never build a result, so a caller must not retry it.
// Artifact read and storage failures and cancellation are never classified so.
var ErrInvalid = errors.New("Audit scan result input is invalid")

// ArtifactReader reads the payload of one exact Run artifact. limit is the
// artifact's bound: a reader may refuse a larger payload before reading it,
// and BuildResult classifies a larger returned payload as ErrInvalid.
type ArtifactReader func(context.Context, contracts.ArtifactRef, int) (artifacts.Payload, error)

type ResultInput struct {
	RunID        string
	CurrentStage string
	Task         auditdomain.ItemTask
	Manifest     auditdomain.ExecutionManifest
	Attempts     []planner.ScanAttempt
}

type Result struct {
	Package   []byte
	Retryable bool
	Summary   string
}

type resultBuilder struct {
	input         ResultInput
	read          ArtifactReader
	members       []auditdomain.PackageInput
	evidence      []auditdomain.Evidence
	evidenceBytes int
	gaps          map[string]bool
	completed     bool
	observed      bool
	needsRecovery bool
	retryable     bool
}

// BuildResult retains plans and raw reports inside the ZIP. Collection and
// deletion of the source Run therefore cannot sever its evidence. Assessments
// distinguish execution coverage from a verified security conclusion.
//
// Only the artifact reads perform I/O, and their errors, cancellation
// included, return unchanged; a missing plan or report is a coverage gap
// instead. An invalid assignment, retained artifact or assembled result is
// ErrInvalid.
func BuildResult(ctx context.Context, input ResultInput, read ArtifactReader) (Result, error) {
	if err := validateAssignment(input); err != nil {
		return Result{}, invalidInput(err)
	}
	if read == nil {
		return Result{}, fmt.Errorf("scan evidence reader is required")
	}
	builder := resultBuilder{input: input, read: read, gaps: make(map[string]bool)}
	for _, gap := range input.Task.Scan.Gaps {
		builder.gaps[gap] = true
	}
	if err := builder.addJournal(); err != nil {
		return Result{}, invalidInput(err)
	}
	for _, attempt := range input.Attempts {
		if err := builder.addAttempt(ctx, attempt); err != nil {
			return Result{}, err
		}
	}
	result, err := builder.packageResult()
	if err != nil {
		return Result{}, invalidInput(err)
	}
	return result, nil
}

func invalidInput(err error) error {
	return fmt.Errorf("%w: %w", ErrInvalid, err)
}

func validateAssignment(input ResultInput) error {
	if _, err := auditdomain.EncodeItemTask(input.Task); err != nil {
		return err
	}
	if input.Task.Scan == nil || len(input.Manifest.Items) != 1 {
		return fmt.Errorf("scan result requires one assigned scan item")
	}
	if err := auditdomain.ValidateDispatchExecutionManifest(input.Manifest); err != nil {
		return err
	}
	item := input.Manifest.Items[0]
	if item.ItemKey != input.Task.ItemKey || item.SubjectKey != input.Task.SubjectKey {
		return fmt.Errorf("scan result assignment differs from execution manifest")
	}
	return nil
}

func (b *resultBuilder) addJournal() error {
	journal := struct {
		Schema   string                `json:"schema"`
		RunID    string                `json:"runId"`
		Task     auditdomain.ItemTask  `json:"task"`
		Attempts []planner.ScanAttempt `json:"attempts"`
	}{journalSchema, b.input.RunID, b.input.Task, b.input.Attempts}
	data, err := strictjson.Canonical(journal)
	if err != nil {
		return err
	}
	if !b.addEvidence("scan-journal", "scan-journal", data) {
		return fmt.Errorf("scan journal exceeds result bound")
	}
	return nil
}

func (b *resultBuilder) packageResult() (Result, error) {
	resultSet, err := b.resultSet()
	if err != nil {
		return Result{}, err
	}
	resultBytes, err := auditdomain.EncodeCheckResultSet(resultSet)
	if err != nil {
		return Result{}, err
	}
	evidenceBytes, err := auditdomain.EncodeEvidence(auditdomain.EvidenceEnvelope{
		Schema: auditdomain.EvidenceSchema, Evidence: b.evidence,
	})
	if err != nil {
		return Result{}, err
	}
	b.members = append(b.members,
		auditdomain.PackageInput{ID: auditdomain.CheckResultsMemberID, Path: "check-results.json", MediaType: auditdomain.JSONMediaType, Data: resultBytes},
		auditdomain.PackageInput{ID: auditdomain.EvidenceMemberID, Path: "evidence.json", MediaType: auditdomain.JSONMediaType, Data: evidenceBytes},
	)
	basis, err := strictjson.Canonical(b.members)
	if err != nil {
		return Result{}, err
	}
	payload, _, err := auditdomain.BuildPackage("scan-result-"+hash(basis), auditdomain.PackageKindCheckResults, "", b.members)
	if err != nil {
		return Result{}, err
	}
	if len(payload) > MaxResultBytes {
		return Result{}, fmt.Errorf("Audit scan result exceeds bound")
	}
	if _, err := auditdomain.DecodeCheckResultPackage(payload); err != nil {
		return Result{}, err
	}
	return Result{Package: payload, Retryable: b.retryable && !b.needsRecovery, Summary: resultSummary}, nil
}

func (b *resultBuilder) resultSet() (auditdomain.CheckResultSet, error) {
	manifestDigest, err := auditdomain.DigestExecutionManifest(b.input.Manifest)
	if err != nil {
		return auditdomain.CheckResultSet{}, err
	}
	task := b.input.Task
	coverage := auditdomain.ResultCoverage{
		Requested: []string{task.Scan.CoverageRequirement()}, Completed: []string{}, Gaps: []string{},
	}
	if b.completed && task.Scan.Runnable {
		coverage.Completed = append(coverage.Completed, coverage.Requested...)
	}
	if !b.observed && task.Scan.Runnable && len(b.gaps) == 0 {
		b.gaps["scan_not_invoked"] = true
	}
	for gap := range b.gaps {
		coverage.Gaps = append(coverage.Gaps, gap)
	}
	sort.Strings(coverage.Gaps)
	assessment := "not-tested"
	if b.observed {
		assessment = "inconclusive"
	}
	evidenceIDs := make([]string, len(b.evidence))
	for index, evidence := range b.evidence {
		evidenceIDs[index] = evidence.ID
	}
	result := auditdomain.CheckResult{
		ItemKey: task.ItemKey, SubjectKey: task.SubjectKey, Assessment: assessment,
		Summary: resultSummary, EvidenceIDs: evidenceIDs, Coverage: coverage, Proposals: []auditdomain.ProposalSelection{},
	}
	set := auditdomain.CheckResultSet{Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: manifestDigest, Results: []auditdomain.CheckResult{result}}
	return set, auditdomain.ValidateResultSet(set, b.input.Manifest)
}

func hash(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
