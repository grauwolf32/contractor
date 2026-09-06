package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
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
	run, err := runstore.NewPostgresStore(tx).GetRun(ctx, input.RunID)
	if err != nil {
		return err
	}
	if run.State != runstore.RunSucceeded {
		return nil
	}
	if run.OwnerID == "" || run.ProjectID == nil || *run.ProjectID != input.ProjectID ||
		run.RunID != input.RunID || digestBytes(run.WorkflowSnapshot) != input.WorkflowClosureDigest {
		return artifacts.ErrArtifactIntegrity
	}
	workflow, err := workflowconfig.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil || workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion {
		return artifacts.ErrArtifactIntegrity
	}
	outputName, outputContract, selected := selectDirectVerificationOutput(workflow)
	if !selected {
		return nil
	}
	runArtifacts, err := artifactService.Run(run.RunID)
	if err != nil {
		return err
	}
	descriptor, err := runArtifacts.Metadata(ctx, contracts.ArtifactRef{
		Namespace: "outputs", Name: outputName,
	})
	if errors.Is(err, artifacts.ErrArtifactNotFound) {
		return nil
	}
	if err != nil {
		return err
	}
	if !descriptor.Frozen || descriptor.MediaType != auditdomain.DirectVerificationsMediaType ||
		descriptor.Ref.Revision == nil {
		return nil
	}
	read, err := runArtifacts.Read(ctx, descriptor.Ref)
	if err != nil {
		return err
	}
	if read.Payload.MediaType != descriptor.MediaType ||
		int64(len(read.Payload.Data)) != descriptor.Size ||
		digestBytes(read.Payload.Data) != descriptor.Digest {
		return artifacts.ErrArtifactIntegrity
	}
	document, err := auditdomain.DecodeDirectVerificationSet(read.Payload.Data)
	if err != nil {
		return nil
	}
	verification, exists := findDirectVerification(
		document, input.InvocationID, input.ClientKey,
	)
	if !exists {
		return nil
	}
	proposal, err := readExactFindingProposal(ctx, runArtifacts, input.Proposal)
	if err != nil {
		return err
	}
	if proposal.ClientKey != input.ClientKey ||
		!isEvidenceSubset(verification.EvidenceIDs, proposal.EvidenceIDs) {
		return nil
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
		return err
	}
	assessmentID := deterministicID("direct-assessment", input.ReceiptID)
	if replayed, err := directAssessmentReplay(
		ctx, tx, assessmentID, input, verification.Assessment,
		descriptor.Digest, digestBytes(contractBytes),
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
		Ref: contractWrite.Ref, Digest: digestBytes(contractBytes),
		MediaType: contractWrite.MediaType, SizeBytes: contractWrite.Size,
	}
	return commitDirectVerification(
		ctx, tx, assessmentID, input, verification,
		resultArtifact, contractArtifact, retainedBytes,
	)
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
			!containsString(slot.MediaTypes, auditdomain.DirectVerificationsMediaType) {
			continue
		}
		if selectedName != "" {
			return "", workflowconfig.ArtifactSlot{}, false
		}
		selectedName, selected = name, slot
	}
	return selectedName, selected, selectedName != ""
}

func findDirectVerification(
	document auditdomain.DirectVerificationSet,
	invocationID, clientKey string,
) (auditdomain.DirectVerificationResult, bool) {
	for _, result := range document.Verifications {
		if result.InvocationID == invocationID && result.ClientKey == clientKey {
			return result, true
		}
	}
	return auditdomain.DirectVerificationResult{}, false
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
		digestBytes(read.Payload.Data) != expected.Digest {
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
	var auditID, receiptID, storedSemantic, storedResultDigest, storedContractDigest string
	var direct bool
	err := tx.QueryRow(ctx, `
SELECT audit_id, receipt_id, semantic_assessment, result_digest,
       direct_verification, contract_digest
  FROM audit_finding_assessments
 WHERE assessment_id = $1`, assessmentID).Scan(
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
	err := tx.QueryRow(ctx, `
INSERT INTO audit_finding_assessments (
    assessment_id, finding_id, audit_id, receipt_id,
    semantic_assessment, result_ref, result_digest,
    direct_verification, contract_ref, contract_digest
)
SELECT $1, contribution.finding_id, contribution.audit_id, contribution.receipt_id,
       $2, $3::jsonb, $4, true, $5::jsonb, $6
  FROM audit_finding_contributions AS contribution
 WHERE contribution.audit_id = $7 AND contribution.receipt_id = $8
RETURNING finding_id`,
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
	var sequence int64
	if err := tx.QueryRow(ctx, `
UPDATE audits
   SET retained_evidence_bytes = retained_evidence_bytes + $2,
       revision = revision + 1,
       next_event_sequence = next_event_sequence + 1,
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE audit_id = $1
   AND retained_evidence_bytes + $2 <= max_evidence_bytes
RETURNING next_event_sequence - 1`, input.AuditID, retainedBytes).Scan(&sequence); err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `
INSERT INTO audit_events (
    audit_id, sequence_number, kind, entity_id, entity_revision, summary
) VALUES (
    $1, $2, 'finding.assessed', $3, $4,
    jsonb_build_object('assessmentId', $5::text, 'directVerification', true)
)`, input.AuditID, sequence, findingID, findingRevision, assessmentID)
	return err
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
