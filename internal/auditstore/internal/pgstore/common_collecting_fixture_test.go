package pgstore

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type collectingAuditFixture struct {
	ctx       context.Context
	pool      *pgxpool.Pool
	store     *testStore
	project   projectstore.Project
	audit     auditstore.Audit
	claim     auditstore.ControllerClaim
	execution auditstore.Execution
	memberID  string
	itemID    string
}

type collectingAuditFixtureOptions struct {
	extraPending    bool
	nonterminal     bool
	baselineExtras  map[string]any
	terminalOutcome string
	// ownDatabase builds the fixture in a database of its own
	// (budgetAuditPool), for a test that budgets its reads.
	ownDatabase bool
}

func newCollectingAuditFixture(t *testing.T, name string, maximum int64, options ...collectingAuditFixtureOptions) collectingAuditFixture {
	t.Helper()
	var option collectingAuditFixtureOptions
	if len(options) > 0 {
		option = options[0]
	}
	if option.terminalOutcome == "" {
		option.terminalOutcome = "succeeded"
	}
	ctx := t.Context()
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	var pool *pgxpool.Pool
	if option.ownDatabase {
		pool = budgetAuditPool(t, ctx, url)
	} else {
		pool = isolatedAuditPool(t, ctx, url)
	}
	common := setupCommonTaskFixture(t, pool, name, false, func(l *auditstore.Limits) { l.MaxEvidenceBytes = maximum })
	var baseline map[string]any
	if err := json.Unmarshal(commonTaskBaseline(common.source, common.material.Digest, common.material.MediaType, common.material.SizeBytes), &baseline); err != nil {
		t.Fatal(err)
	}
	for k, v := range option.baselineExtras {
		baseline[k] = v
	}
	encoded, _ := json.Marshal(baseline)
	var err error
	common.audit, _, err = common.owners.Activate(ctx, auditstore.ActivateParams{OwnerID: common.audit.OwnerID, AuditID: common.audit.AuditID, ExpectedRevision: common.audit.Revision, BaselineSnapshot: encoded, DeadlineAt: time.Now().Add(time.Hour), IdempotencyKey: "start", RequestDigest: testDigest("4")})
	if err != nil {
		t.Fatal(err)
	}
	count := 1
	if option.extraPending {
		count = 2
	}
	accepted, _, err := common.owners.AcceptTasks(ctx, common.acceptance(count, "accept"))
	if err != nil {
		t.Fatal(err)
	}
	store := newTestStore(pool)
	claimed, err := store.Claim(ctx, auditstore.ClaimParams{HolderID: name + "-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claimed) != 1 {
		t.Fatalf("claim: %v %v", claimed, err)
	}
	claim := claimed[0]
	execution, _, err := store.SelectTasks(ctx, auditstore.SelectTasksParams{Claim: claim, TaskIDs: accepted.TaskIDs[:1]})
	if err != nil {
		t.Fatal(err)
	}
	execution, err = insertAndBindTestRun(t, ctx, pool, claim, execution, "run-"+name, common.audit.OwnerID, common.audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	if !option.nonterminal {
		generation, sequence := terminateTestRun(t, ctx, pool, *execution.RunID, option.terminalOutcome)
		execution, err = store.ObserveTerminal(ctx, auditstore.ObserveTerminalParams{Claim: claim, ExecutionID: execution.ExecutionID, RunID: *execution.RunID, Generation: generation, Sequence: sequence})
		if err != nil {
			t.Fatal(err)
		}
	}
	audit, err := store.Get(ctx, common.audit.OwnerID, common.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	project, err := projectstore.NewPostgresStore(pool).Get(ctx, common.audit.OwnerID, common.audit.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	return collectingAuditFixture{ctx: ctx, pool: pool, store: store, project: project, audit: audit, claim: claim, execution: execution, itemID: accepted.TaskIDs[0], memberID: auditdomain.DeterministicID("attempt", execution.ExecutionID, accepted.TaskIDs[0])}
}

func insertAuditChildFinding(
	t *testing.T, f collectingAuditFixture, receiptID string, proposal auditstore.ExactArtifact,
) string {
	t.Helper()
	proposalJSON, err := json.Marshal(proposal)
	if err != nil {
		t.Fatal(err)
	}
	proposalRefJSON, err := json.Marshal(proposal.Ref)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id, audit_execution_id, audit_id, audit_task_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, publication_requested, project_proposal, project_evidence, retention_state, report_projection
, checks_projection) VALUES (
    $1, $1 || '-proposal', $1 || '-allocation', 'runtime-agent', 'runtime-instance',
    'stage-execution', 'worker', 'invocation', $1 || '-submission', 'candidate', $2,
    $3, $4, $5, $6, $7, $13,
    'audit-check', '1', 'contractor/v1alpha1',
    '{"name":"audit-check","version":"1"}'::jsonb, $8,
    $9::jsonb, $10, 'application/json', $11, '[]'::jsonb, true, $12::jsonb, jsonb_build_array($12::jsonb), 'project-held',
    '{"description":"Test finding.","limitations":[],"preconditions":[],"severity_suggestion":"","standard_refs":[],"subject":null,"title":"Test finding."}'
, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($10)::text,'size_bytes',($11)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, testDigest("d"), *f.execution.RunID, f.audit.OwnerID, f.project.ProjectID,
		f.execution.ExecutionID, f.audit.AuditID, testDigest("e"),
		proposalRefJSON, proposal.Digest, proposal.SizeBytes, proposalJSON, f.itemID,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_audit_holds (
    receipt_id, audit_id, project_id, proposal_ref, evidence
) VALUES ($1, $2, $3, $4::jsonb, '[]'::jsonb)`,
		receiptID, f.audit.AuditID, f.project.ProjectID, proposalJSON); err != nil {
		t.Fatal(err)
	}
	var findingID string
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		var err error
		findingID, err = NewFindingEventsTx(tx, artifactpgtest.NewStorage()).AdmitProjectFinding(f.ctx, f.project.ProjectID, receiptID, "finding-"+receiptID)
		return err
	}); err != nil {
		t.Fatal(err)
	}

	return findingID
}
