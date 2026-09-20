package postgres

import (
	"context"
	"testing"
	"time"
)

func TestPostgresAuditFindingIdentityUpgradePreservesHistory(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	installStorageReviewPrefix(t, ctx, pool, 61)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id,owner_id,kind,name,request_idempotency_key,request_digest)
VALUES ('upgrade-project','owner','project','Upgrade','upgrade','sha256:'||repeat('a',64));
INSERT INTO audits (
    audit_id,owner_id,project_id,profile_name,profile_version,profile_digest,
    profile_snapshot,input_selection,max_rounds,batch_size,max_items_per_round,
    max_items_total,max_submitted_runs,max_item_run_attempts,max_evidence_bytes
)
SELECT id,'owner','upgrade-project','profile','1','sha256:'||repeat('a',64),
       '{}','{}',1,1,10,10,10,1,1024
FROM unnest(ARRAY['legacy-audit','new-audit-b','new-audit-c']) AS ids(id);
INSERT INTO finding_proposal_receipts (
    receipt_id,proposal_id,allocation_id,runtime_agent_id,runtime_instance_id,
    stage_execution_id,logical_agent_name,invocation_id,submission_id,client_key,
    request_digest,run_id,owner_id,project_id,workflow_name,workflow_version,
    workflow_schema_version,workflow_configuration_ref,workflow_closure_digest,
    proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes,evidence
) VALUES (
    'shared-receipt','proposal','allocation','runtime','instance','stage','worker',
    'invocation','submission','candidate','sha256:'||repeat('a',64),'historical-run',
    'owner','upgrade-project','workflow','1','contractor/v1alpha1','{}',
    'sha256:'||repeat('a',64),'{"namespace":"findings","name":"proposal","revision":"r1"}',
    'sha256:'||repeat('a',64),'application/json',128,'[]'
);
INSERT INTO finding_proposal_retention(receipt_id) VALUES ('shared-receipt');
INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
VALUES ('shared-receipt','legacy-audit','upgrade-project','{}','[]');
INSERT INTO audit_finding_assessments (
    assessment_id,finding_id,audit_id,receipt_id,semantic_assessment,result_ref,
    result_digest,direct_verification,contract_ref,contract_digest
) VALUES (
    'legacy-assessment','finding-shared-receipt','legacy-audit','shared-receipt',
    'supported','{}','sha256:'||repeat('b',64),true,'{}','sha256:'||repeat('c',64)
);
INSERT INTO audit_review_requests (
    request_id,audit_id,finding_id,kind,subject_revision,subject_digest,
    requested_actions,state,idempotency_key,request_digest,subject_kind,subject_id
) VALUES (
    'legacy-review','legacy-audit','finding-shared-receipt','finding-triage',1,
    'sha256:'||repeat('d',64),'["true_positive"]','decided','legacy-review',
    'sha256:'||repeat('d',64),'finding','finding-shared-receipt'
);
INSERT INTO audit_review_decisions (
    decision_id,request_id,audit_id,finding_id,actor_id,verdict,severity,rationale,
    subject_revision,subject_digest,idempotency_key,request_digest,action
) VALUES (
    'legacy-decision','legacy-review','legacy-audit','finding-shared-receipt',
    'owner','true_positive','medium','Historical owner decision',1,
    'sha256:'||repeat('d',64),'legacy-decision','sha256:'||repeat('d',64),'true_positive'
);
UPDATE audit_findings SET state='confirmed', current_assessment_id='legacy-assessment',
    current_decision_id='legacy-decision',revision=2,updated_at=clock_timestamp()
WHERE audit_id='legacy-audit';
`); err != nil {
		t.Fatalf("create historical finding and history on schema 61: %v", err)
	}
	const admit = `INSERT INTO finding_proposal_audit_holds
    (receipt_id,audit_id,project_id,proposal_ref,evidence)
VALUES ('shared-receipt',$1,'upgrade-project','{}','[]')
ON CONFLICT (receipt_id,audit_id) DO NOTHING`
	if _, err := pool.Exec(ctx, admit, "new-audit-b"); SQLState(err) != "23505" {
		t.Fatalf("schema 61 must reproduce cross-Audit finding collision: %v", err)
	}
	// Compare all legacy columns, exact refs, timestamps and history, not only
	// the old primary key. New admission must not rewrite an accepted decision.
	const legacyHistory = `SELECT jsonb_build_object(
    'audit',(SELECT to_jsonb(a) FROM audits a WHERE audit_id='legacy-audit'),
    'findings',(SELECT jsonb_agg(to_jsonb(f) ORDER BY finding_id) FROM audit_findings f WHERE audit_id='legacy-audit'),
    'contributions',(SELECT jsonb_agg(to_jsonb(c) ORDER BY receipt_id) FROM audit_finding_contributions c WHERE audit_id='legacy-audit'),
    'assessments',(SELECT jsonb_agg(to_jsonb(a) ORDER BY assessment_id) FROM audit_finding_assessments a WHERE audit_id='legacy-audit'),
    'requests',(SELECT jsonb_agg(to_jsonb(r) ORDER BY request_id) FROM audit_review_requests r WHERE audit_id='legacy-audit'),
    'decisions',(SELECT jsonb_agg(to_jsonb(d) ORDER BY decision_id) FROM audit_review_decisions d WHERE audit_id='legacy-audit'),
    'events',(SELECT jsonb_agg(to_jsonb(e) ORDER BY sequence_number) FROM audit_events e WHERE audit_id='legacy-audit')
)::text`
	var before, after string
	if err := pool.QueryRow(ctx, legacyHistory).Scan(&before); err != nil {
		t.Fatal(err)
	}
	result, err := ApplyMigrations(ctx, pool)
	if err != nil || len(result.AppliedVersions) == 0 || result.AppliedVersions[0] != 62 {
		t.Fatalf("forward identity migration: %+v, %v", result, err)
	}
	if err := pool.QueryRow(ctx, legacyHistory).Scan(&after); err != nil || after != before {
		t.Fatalf("migration rewrote legacy finding history: %v", err)
	}
	for _, audit := range []string{"legacy-audit", "new-audit-b", "new-audit-c"} {
		for attempt := 0; attempt < 2; attempt++ {
			if _, err := pool.Exec(ctx, admit, audit); err != nil {
				t.Fatalf("admit/replay same receipt in %s: %v", audit, err)
			}
		}
	}
	var findings, distinctIDs, holds, contributions, events int
	if err := pool.QueryRow(ctx, `SELECT
    (SELECT count(*) FROM audit_findings),
    (SELECT count(DISTINCT finding_id) FROM audit_findings),
    (SELECT count(*) FROM finding_proposal_audit_holds),
    (SELECT count(*) FROM audit_finding_contributions),
    (SELECT count(*) FROM audit_events WHERE kind='finding.proposed')`).Scan(&findings, &distinctIDs, &holds, &contributions, &events); err != nil {
		t.Fatal(err)
	}
	if findings != 3 || distinctIDs != 3 || holds != 3 || contributions != 3 || events != 3 {
		t.Fatalf("independent admission/replay counts: findings=%d identities=%d holds=%d contributions=%d events=%d", findings, distinctIDs, holds, contributions, events)
	}
	if err := pool.QueryRow(ctx, legacyHistory).Scan(&after); err != nil || after != before {
		t.Fatalf("new Audit admission changed legacy finding history: %v", err)
	}
	again, err := ApplyMigrations(ctx, pool)
	if err != nil || len(again.AppliedVersions) != 0 {
		t.Fatalf("replayed forward migration: %+v, %v", again, err)
	}
}
