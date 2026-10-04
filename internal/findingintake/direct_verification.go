package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

const directVerificationContractSchema = "contractor.audit.direct-verification-contract.v1"

type directVerificationInput struct {
	AuditID               string
	ProjectID             string
	ReceiptID             string
	RunID                 string
	InvocationID          string
	ClientKey             string
	WorkflowClosureDigest string
	Proposal              ExactArtifact
}

type directVerificationContract struct {
	Schema          string                            `json:"schema"`
	ResultSchema    string                            `json:"resultSchema"`
	ResultMediaType string                            `json:"resultMediaType"`
	Workflow        json.RawMessage                   `json:"workflow"`
	ClosureDigest   string                            `json:"workflowClosureDigest"`
	OutputName      string                            `json:"outputName"`
	Output          workflowconfig.ArtifactSlot       `json:"output"`
	Binding         directVerificationContractBinding `json:"binding"`
}

type directVerificationContractBinding struct {
	InvocationField string `json:"invocationField"`
	ClientKeyField  string `json:"clientKeyField"`
	EvidenceRule    string `json:"evidenceRule"`
}

type directVerificationCacheKey struct{}

type directVerificationCache struct {
	ready     bool
	runID     string
	projectID string
	closure   string
	prepared  *preparedDirectVerification
	err       error
}

type preparedDirectVerification struct {
	descriptor    artifacts.Metadata
	contractBytes []byte
	results       map[string]auditdomain.DirectVerificationResult
	// budgetExhaustedIn is the transaction that found the Audit evidence
	// budget exhausted. Another transaction checks again: a rolled-back one
	// returns the bytes it charged, and collection may continue after it.
	budgetExhaustedIn pgx.Tx
}

// WithCollectionDirectVerificationCache shares one immutable terminal Run
// output across the receipts of a single Audit collection attempt. Ordinary
// owner imports still resolve each transaction independently.
func WithCollectionDirectVerificationCache(ctx context.Context) context.Context {
	return context.WithValue(ctx, directVerificationCacheKey{}, &directVerificationCache{})
}

func resolveDirectVerification(
	ctx context.Context, tx pgx.Tx, service *artifacts.Service, input directVerificationInput,
) (*preparedDirectVerification, error) {
	cache, _ := ctx.Value(directVerificationCacheKey{}).(*directVerificationCache)
	if cache == nil {
		return prepareDirectVerification(ctx, tx, service, input)
	}
	if cache.ready {
		if cache.runID != input.RunID || cache.projectID != input.ProjectID || cache.closure != input.WorkflowClosureDigest {
			return nil, artifacts.ErrArtifactIntegrity
		}
		return cache.prepared, cache.err
	}
	cache.runID, cache.projectID, cache.closure = input.RunID, input.ProjectID, input.WorkflowClosureDigest
	cache.prepared, cache.err = prepareDirectVerification(ctx, tx, service, input)
	cache.ready = true
	return cache.prepared, cache.err
}

// tryAcceptDirectVerification is intentionally best-effort only for absent or
// invalid opt-in output. Proposal intake remains valid and leaves a candidate
// unassessed in those cases. Durable or integrity failures still abort the
// transaction so a replay can retry safely.
func tryAcceptDirectVerification(
	ctx context.Context,
	tx pgx.Tx,
	artifactService *artifacts.Service,
	input directVerificationInput,
) error {
	prepared, err := resolveDirectVerification(ctx, tx, artifactService, input)
	if err != nil || prepared == nil {
		return err
	}
	verification, exists := prepared.results[input.InvocationID+"\x00"+input.ClientKey]
	if !exists {
		return nil
	}
	if prepared.budgetExhaustedIn == tx {
		return nil
	}
	runArtifacts, err := artifactService.Run(input.RunID)
	if err != nil {
		return err
	}
	proposal, err := readExactFindingProposal(ctx, runArtifacts, input.Proposal)
	if err != nil {
		return err
	}
	if proposal.ClientKey != input.ClientKey ||
		!isEvidenceSubset(verification.EvidenceIDs, proposal.EvidenceIDs) {
		return nil
	}
	contractBytes, descriptor := prepared.contractBytes, prepared.descriptor
	assessmentID := deterministicID("direct-assessment", input.AuditID, input.ReceiptID)
	if replayed, err := directAssessmentReplay(
		ctx, tx, assessmentID, input, verification.Assessment,
		descriptor.Digest, auditdomain.DigestBytes(contractBytes),
	); err != nil || replayed {
		return err
	}
	retainedBytes := descriptor.Size + int64(len(contractBytes))
	var withinBudget bool
	if err := tx.QueryRow(ctx, `
SELECT retained_evidence_bytes + $2 <= max_evidence_bytes
  FROM audits
 WHERE audit_id = $1`, input.AuditID, retainedBytes).Scan(&withinBudget); err != nil {
		return err
	}
	if !withinBudget {
		prepared.budgetExhaustedIn = tx
		return nil
	}

	namespace := auditdomain.ArtifactNamespace(input.AuditID)
	resultArtifact, err := retainFindingArtifact(
		ctx, artifactService, input.RunID, input.ProjectID, namespace,
		deterministicID("direct-result", input.ReceiptID),
		ExactArtifact{
			Ref: descriptor.Ref, Digest: descriptor.Digest,
			MediaType: descriptor.MediaType, SizeBytes: descriptor.Size,
		},
	)
	if err != nil {
		return err
	}
	contractWrite, err := artifactService.WriteAuditArtifact(
		ctx, input.ProjectID,
		contracts.ArtifactRef{
			Namespace: namespace,
			Name:      deterministicID("direct-contract", input.ReceiptID),
		},
		artifacts.Payload{
			MediaType: "application/vnd.contractor.audit.direct-verification-contract+json",
			Data:      contractBytes,
		},
	)
	if err != nil {
		return err
	}
	contractArtifact := ExactArtifact{
		Ref: contractWrite.Ref, Digest: auditdomain.DigestBytes(contractBytes),
		MediaType: contractWrite.MediaType, SizeBytes: contractWrite.Size,
	}
	return commitDirectVerification(
		ctx, tx, assessmentID, input, verification,
		resultArtifact, contractArtifact, retainedBytes,
	)
}

func prepareDirectVerification(
	ctx context.Context, tx pgx.Tx, artifactService *artifacts.Service, input directVerificationInput,
) (*preparedDirectVerification, error) {
	run, err := runstore.NewPostgresStore(tx).GetRun(ctx, input.RunID)
	if err != nil {
		return nil, err
	}
	if run.State != runstore.RunSucceeded {
		return nil, nil
	}
	if run.OwnerID == "" || run.ProjectID == nil || *run.ProjectID != input.ProjectID ||
		run.RunID != input.RunID || auditdomain.DigestBytes(run.WorkflowSnapshot) != input.WorkflowClosureDigest {
		return nil, artifacts.ErrArtifactIntegrity
	}
	workflow, err := workflowconfig.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil || workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion {
		return nil, artifacts.ErrArtifactIntegrity
	}
	outputName, outputContract, selected := selectDirectVerificationOutput(workflow)
	if !selected {
		return nil, nil
	}
	runArtifacts, err := artifactService.Run(run.RunID)
	if err != nil {
		return nil, err
	}
	descriptor, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{
		Namespace: "outputs", Name: outputName,
	})
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !descriptor.Frozen || descriptor.MediaType != auditdomain.DirectVerificationsMediaType ||
		descriptor.Ref.Revision == nil {
		return nil, nil
	}
	read, err := runArtifacts.Read(ctx, descriptor.Ref)
	if err != nil {
		return nil, err
	}
	if read.Payload.MediaType != descriptor.MediaType ||
		int64(len(read.Payload.Data)) != descriptor.Size ||
		auditdomain.DigestBytes(read.Payload.Data) != descriptor.Digest {
		return nil, artifacts.ErrArtifactIntegrity
	}
	document, err := auditdomain.DecodeDirectVerificationSet(read.Payload.Data)
	if err != nil {
		return nil, nil
	}
	contractBytes, err := json.Marshal(directVerificationContract{
		Schema:          directVerificationContractSchema,
		ResultSchema:    auditdomain.DirectVerificationsSchema,
		ResultMediaType: auditdomain.DirectVerificationsMediaType,
		Workflow:        append(json.RawMessage(nil), run.WorkflowSnapshot...),
		ClosureDigest:   input.WorkflowClosureDigest,
		OutputName:      outputName,
		Output:          outputContract,
		Binding: directVerificationContractBinding{
			InvocationField: "invocation_id",
			ClientKeyField:  "client_key",
			EvidenceRule:    "selected IDs must belong to the exact proposal receipt",
		},
	})
	if err != nil {
		return nil, err
	}
	results := make(map[string]auditdomain.DirectVerificationResult, len(document.Verifications))
	for _, verification := range document.Verifications {
		results[verification.InvocationID+"\x00"+verification.ClientKey] = verification
	}
	return &preparedDirectVerification{
		descriptor: descriptor, contractBytes: contractBytes, results: results,
	}, nil
}

func selectDirectVerificationOutput(
	workflow workflowconfig.ResolvedWorkflow,
) (string, workflowconfig.ArtifactSlot, bool) {
	names := make([]string, 0, len(workflow.Outputs))
	for name := range workflow.Outputs {
		names = append(names, name)
	}
	sort.Strings(names)
	var selectedName string
	var selected workflowconfig.ArtifactSlot
	for _, name := range names {
		slot := workflow.Outputs[name]
		if !slot.Required || !slot.Primary ||
			!slices.Contains(slot.MediaTypes, auditdomain.DirectVerificationsMediaType) {
			continue
		}
		if selectedName != "" {
			return "", workflowconfig.ArtifactSlot{}, false
		}
		selectedName, selected = name, slot
	}
	return selectedName, selected, selectedName != ""
}

func readExactFindingProposal(
	ctx context.Context,
	store artifacts.ScopedStore,
	expected ExactArtifact,
) (auditdomain.FindingProposal, error) {
	read, err := store.Read(ctx, expected.Ref)
	if err != nil {
		return auditdomain.FindingProposal{}, err
	}
	if read.Payload.MediaType != expected.MediaType ||
		int64(len(read.Payload.Data)) != expected.SizeBytes ||
		auditdomain.DigestBytes(read.Payload.Data) != expected.Digest {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	proposal, err := auditdomain.DecodeFindingProposal(read.Payload.Data)
	if err != nil {
		return auditdomain.FindingProposal{}, artifacts.ErrArtifactIntegrity
	}
	return proposal, nil
}

func isEvidenceSubset(selected, available []string) bool {
	known := make(map[string]struct{}, len(available))
	for _, value := range available {
		known[value] = struct{}{}
	}
	for _, value := range selected {
		if _, exists := known[value]; !exists {
			return false
		}
	}
	return true
}

func directAssessmentReplay(
	ctx context.Context,
	tx pgx.Tx,
	assessmentID string,
	input directVerificationInput,
	semantic, resultDigest, contractDigest string,
) (bool, error) {
	// Replay only the current identity scoped to the destination Audit.
	var auditID, receiptID, storedSemantic, storedResultDigest, storedContractDigest string
	var direct bool
	err := tx.QueryRow(ctx, `
SELECT audit_id, receipt_id, semantic_assessment, result_digest,
       direct_verification, contract_digest
  FROM audit_finding_assessments
 WHERE audit_id = $2 AND assessment_id = $1`,
		assessmentID, input.AuditID,
	).Scan(
		&auditID, &receiptID, &storedSemantic, &storedResultDigest, &direct, &storedContractDigest,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if auditID != input.AuditID || receiptID != input.ReceiptID || storedResultDigest != resultDigest ||
		!direct || storedContractDigest != contractDigest || semantic != storedSemantic {
		return true, fmt.Errorf("%w: direct verification replay differs", ErrConflict)
	}
	return true, nil
}

func commitDirectVerification(
	ctx context.Context,
	tx pgx.Tx,
	assessmentID string,
	input directVerificationInput,
	verification auditdomain.DirectVerificationResult,
	result, contract ExactArtifact,
	retainedBytes int64,
) error {
	resultRef, _ := json.Marshal(result.Ref)
	contractRef, _ := json.Marshal(contract.Ref)
	resultProvenance, _ := json.Marshal(map[string]any{
		"schema":                "contractor.audit.direct-verification-provenance.v1",
		"kind":                  "result",
		"receiptId":             input.ReceiptID,
		"runId":                 input.RunID,
		"invocationId":          input.InvocationID,
		"clientKey":             input.ClientKey,
		"workflowClosureDigest": input.WorkflowClosureDigest,
	})
	contractProvenance, _ := json.Marshal(map[string]any{
		"schema":                "contractor.audit.direct-verification-provenance.v1",
		"kind":                  "contract",
		"receiptId":             input.ReceiptID,
		"runId":                 input.RunID,
		"workflowClosureDigest": input.WorkflowClosureDigest,
	})
	logicalResult := "finding/" + input.ReceiptID + "/direct-result"
	logicalContract := "finding/" + input.ReceiptID + "/direct-contract"
	if _, err := tx.Exec(ctx, `
INSERT INTO audit_artifact_links (
    audit_id, logical_key, artifact_ref, artifact_digest,
    media_type, size_bytes, source_provenance, display_ref
) VALUES
    ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, 'direct verification result'),
    ($1, $8, $9::jsonb, $10, $11, $12, $13::jsonb, 'direct verification contract')`,
		input.AuditID,
		logicalResult, resultRef, result.Digest, result.MediaType, result.SizeBytes, resultProvenance,
		logicalContract, contractRef, contract.Digest, contract.MediaType, contract.SizeBytes, contractProvenance,
	); err != nil {
		return err
	}
	var findingID string
	err := tx.QueryRow(ctx, commitDirectVerificationSQL,
		assessmentID, verification.Assessment, resultRef, result.Digest,
		contractRef, contract.Digest, input.AuditID, input.ReceiptID,
	).Scan(&findingID)
	if err != nil {
		return err
	}
	var findingRevision int64
	if err := tx.QueryRow(ctx, `
UPDATE audit_findings
   SET current_assessment_id = $1, current_decision_id = NULL,
       state = 'proposed', rejection_reason = NULL, duplicate_target_id = NULL,
       revision = revision + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $2 AND finding_id = $3
RETURNING revision`, assessmentID, input.AuditID, findingID).Scan(&findingRevision); err != nil {
		return err
	}
	return auditstore.NewPostgresStore(tx).RecordDirectFindingAssessment(ctx,
		auditstore.DirectFindingAssessmentParams{
			AuditID: input.AuditID, FindingID: findingID,
			FindingRevision: findingRevision, AssessmentID: assessmentID,
			RetainedBytes: retainedBytes,
		})
}
