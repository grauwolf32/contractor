// Package auditscan retains scanner journals and normalized reports as ordinary
// JSON. Provider parsing remains in Runtime adapters; this package never infers
// a security verdict or accepts Worker-authored Audit identities.
package auditscan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/planner"
)

const (
	MaxResultBytes        = planner.MaxScanArtifactBytes
	MaxScannerReportBytes = planner.MaxScanReportBytes
	resultSummary         = "Scanner execution and observations are retained; security conclusions remain unverified."
)

var ErrInvalid = errors.New("scan report input is invalid")

type ArtifactReader func(context.Context, contracts.ArtifactRef, int) (artifacts.Payload, error)

type ResultInput struct {
	RunID             string
	CurrentStage      string
	Scanner           string
	PreparationDigest string
	Runnable          bool
	Gaps              []string
	Attempts          []planner.ScanAttempt
}

type Result struct {
	Data               []byte
	Retryable          bool
	ExecutionCompleted bool
	Summary            string
}

type retainedArtifact struct {
	ID        string                `json:"id"`
	Kind      string                `json:"kind"`
	Ref       contracts.ArtifactRef `json:"ref"`
	Digest    string                `json:"digest"`
	SizeBytes int                   `json:"size_bytes"`
}

type operationReport struct {
	Schema             string                `json:"schema"`
	RunID              string                `json:"run_id"`
	Scanner            string                `json:"scanner"`
	PreparationDigest  string                `json:"preparation_digest,omitempty"`
	Runnable           bool                  `json:"runnable"`
	ExecutionCompleted bool                  `json:"execution_completed"`
	Limitations        []string              `json:"limitations"`
	Attempts           []planner.ScanAttempt `json:"attempts"`
	Artifacts          []retainedArtifact    `json:"artifacts"`
}

type resultBuilder struct {
	input         ResultInput
	read          ArtifactReader
	artifacts     []retainedArtifact
	gaps          map[string]bool
	completed     bool
	observed      bool
	needsRecovery bool
	retryable     bool
}

// BuildResult retains bounded exact plans and normalized reports inline, so
// deleting a source Run cannot sever the report's evidence. References remain
// provenance. No recursive Artifact traversal or provider-specific parsing occurs.
func BuildResult(ctx context.Context, input ResultInput, read ArtifactReader) (Result, error) {
	if !contracts.ValidResourceID(input.RunID) || (input.Scanner != "nuclei" && input.Scanner != "sqlmap") || read == nil {
		return Result{}, fmt.Errorf("%w: invalid scan report assignment", ErrInvalid)
	}
	builder := resultBuilder{input: input, read: read, gaps: map[string]bool{}, artifacts: []retainedArtifact{}}
	for _, gap := range input.Gaps {
		builder.gaps[gap] = true
	}
	for _, attempt := range input.Attempts {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		if err := builder.addAttempt(ctx, attempt); err != nil {
			return Result{}, err
		}
	}
	if !builder.observed && input.Runnable && len(builder.gaps) == 0 {
		builder.gaps["scan_not_invoked"] = true
	}
	gaps := make([]string, 0, len(builder.gaps))
	for gap := range builder.gaps {
		gaps = append(gaps, gap)
	}
	sort.Strings(gaps)
	attempts := input.Attempts
	if attempts == nil {
		attempts = []planner.ScanAttempt{}
	}
	report := operationReport{Schema: "contractor.scan.operation-report.v1", RunID: input.RunID, Scanner: input.Scanner, PreparationDigest: input.PreparationDigest, Runnable: input.Runnable, ExecutionCompleted: builder.completed && input.Runnable, Limitations: gaps, Attempts: attempts, Artifacts: builder.artifacts}
	data, err := strictjson.Canonical(report)
	if err != nil || len(data) > MaxResultBytes {
		return Result{}, fmt.Errorf("%w: scan report exceeds JSON bound", ErrInvalid)
	}
	return Result{Data: data, Retryable: builder.retryable && !builder.needsRecovery, ExecutionCompleted: report.ExecutionCompleted, Summary: resultSummary}, nil
}

func hash(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
