package public

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func seedPublicPaginationFinding(t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	ownerID, projectID, auditID, suffix string, fingerprints ...auditdomain.FindingFingerprint,
) string {
	t.Helper()
	document := auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + suffix,
		Title: "Candidate " + suffix, Description: "Retained pagination test candidate.",
		Subject:       &auditdomain.FindingSubject{Kind: "component", Key: suffix},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{"evidence-1"}, ProposedChecks: []auditdomain.ProposedCheck{},
		SeveritySuggestion: "medium", Limitations: []string{},
	}

	evidenceBytes := []byte("Exact source evidence " + suffix)
	storedEvidence, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: "evidence-" + suffix}, artifacts.Payload{MediaType: "text/plain", Data: evidenceBytes})
	if err != nil {
		t.Fatal(err)
	}
	evidenceJSON, _ := json.Marshal([]findingintake.ExactArtifact{{Ref: storedEvidence.Ref, Digest: auditdomain.DigestBytes(evidenceBytes), MediaType: storedEvidence.MediaType, SizeBytes: storedEvidence.Size}})
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(document))
	if err != nil {
		t.Fatal(err)
	}
	written, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifacts.NewPostgresRepository(pool)), projectID,
		contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: document.ClientKey},
		artifacts.Payload{MediaType: "application/json", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{Ref: written.Ref, Digest: auditHandlerDigest(string(payload)),
		MediaType: written.MediaType, SizeBytes: written.Size}
	proposalJSON, _ := json.Marshal(proposal)
	refJSON, _ := json.Marshal(proposal.Ref)
	fingerprint := auditdomain.UnavailableFindingFingerprint("not_recorded")
	if len(fingerprints) == 1 {
		fingerprint = fingerprints[0]
	}
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	receiptID := "receipt-" + suffix
	_, err = pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id, client_key,
    request_digest, run_id, owner_id, project_id, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence,
    report_projection, fingerprint_projection, project_proposal,project_evidence,publication_requested,retention_state,source_run_deleted_at
, checks_projection) VALUES ($1, $2, $3, 'runtime-pages', 'instance-pages', 'stage-pages', 'worker', $4, $5, $6,
    $7, $8, $9, $10, 'finding-source', '1', 'contractor/v1alpha1',
    '{"name":"finding-source","version":"1"}'::jsonb, $11, $12::jsonb, $13, 'application/json', $14, $17::jsonb, $15,$16,$18::jsonb,$17::jsonb,true,'project-held',clock_timestamp(), json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($13)::text,'size_bytes',($14)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix, "invocation-"+suffix, "submission-"+suffix,
		document.ClientKey, auditHandlerDigest("request-"+suffix), "deleted-run-"+suffix,
		ownerID, projectID, auditHandlerDigest("workflow"), refJSON, proposal.Digest, proposal.SizeBytes,
		string(projection), string(fingerprintJSON), evidenceJSON, proposalJSON)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds (receipt_id, audit_id, project_id, proposal_ref, evidence)
VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)`, receiptID, auditID, projectID, proposalJSON, evidenceJSON); err != nil {
		t.Fatal(err)
	}

	var findingID string
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		hooks := hookstore.NewFindingEventsTx(tx)
		var err error
		findingID, err = hooks.AdmitProjectFinding(ctx, projectID, receiptID, "finding-"+suffix)
		if err != nil {
			return err
		}
		return hooks.RecordFindingSourceAdded(ctx, auditID, receiptID, "deleted-run-"+suffix)
	})
	if err != nil {
		t.Fatal(err)
	}
	return findingID
}
