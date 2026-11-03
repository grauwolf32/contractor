package auditservice

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	managedcredentials "github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
	"github.com/grauwolf32/contractor/internal/persistence/postgres"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/persistence/postgrestest"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestCommonAuditDraftStartAndOriginalReplay(t *testing.T) {
	ctx := t.Context()
	pool := isolatedAuditServicePool(t, ctx, os.Getenv("CONTRACTOR_TEST_DATABASE_URL"))
	service := newStartStagingService(t, pool, &countingCredentialGuard{})
	project, draft := createStartStagingDraft(t, ctx, pool, service, "common-start", 1)
	params := StartParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision, IdempotencyKey: "start", RequestDigest: serviceTestDigest("start")}
	started, err := service.Start(ctx, params)
	if err != nil || started.Audit.State != auditstore.AuditActive || started.Audit.CurrentRoundID != nil || len(started.Items) != 0 {
		t.Fatalf("ordinary start=%+v error=%v", started, err)
	}
	pin, err := auditprofile.DecodePinnedProgram(started.Audit.ProfileSnapshot)
	if err != nil || len(pin.Program.InitialWork) != 1 || len(pin.Program.Hypotheses) != 0 {
		t.Fatal("invalid exploratory pin", err)
	}
	replay, err := service.Start(ctx, params)
	if err != nil || !replay.Replayed || replay.Audit.Revision != started.Audit.Revision {
		t.Fatalf("original replay=%+v error=%v", replay, err)
	}
	var runs, tasks int
	if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM workflow_runs),(SELECT count(*) FROM audit_tasks)`).Scan(&runs, &tasks); err != nil || runs != 0 || tasks != 0 {
		t.Fatalf("Start dispatched outside the controller: Runs=%d Tasks=%d error=%v", runs, tasks, err)
	}
}

func TestAuditStartUsesOwningTransactionWithSaturatedPool(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	development, err := managedcredentials.NewStaticProvider([]managedcredentials.StaticEntry{{
		Metadata: resolved.CredentialMetadata{
			Ref:        llmgateway.LLMCredentialRef{CredentialID: "development-worker"},
			LLMGateway: gateway.Ref, Unrestricted: true,
		},
		Token: contracts.NewSecretString("development-test-token"),
	}})
	if err != nil {
		t.Fatal(err)
	}
	transactionCredentials, err := managedcredentials.NewTransactionLookupFactory(development)
	if err != nil {
		t.Fatal(err)
	}

	limitedConfig := pool.Config()
	limitedConfig.MaxConns = 2
	limited, err := pgxpool.NewWithConfig(ctx, limitedConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer limited.Close()
	service, err := New(Options{Storage: artifactpgtest.NewStorage(),
		Pool: limited, Profiles: profiles,
		TransactionLLMCredentials: transactionCredentials,
		CredentialGuard:           &countingCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}

	project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
		ProjectID: "project-saturated-audit", OwnerID: "owner-saturated-audit",
		Kind: projectstore.KindProject, Name: "Saturated Audit",
		IdempotencyKey: "project-saturated-audit", RequestDigest: serviceTestDigest("saturated-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "saturated", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-saturated", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:          Scope{Objective: "Exercise transaction-bound credential validation"},
		IdempotencyKey: "create-saturated-audit", RequestDigest: serviceTestDigest("saturated-create"),
	})
	if err != nil {
		t.Fatal(err)
	}

	listener, err := limited.Acquire(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Release()
	if _, err := listener.Exec(ctx, `LISTEN contractor_audit_transaction_test`); err != nil {
		t.Fatal(err)
	}
	startCtx, startCancel := context.WithTimeout(ctx, 5*time.Second)
	defer startCancel()
	started, err := service.Start(startCtx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-saturated-audit", RequestDigest: serviceTestDigest("saturated-start"),
	})
	if err != nil {
		t.Fatalf("start Audit with one LISTEN connection and one transaction: %v", err)
	}
	if started.Audit.State != auditstore.AuditActive || len(started.Items) != 0 {
		t.Fatalf("started saturated Audit = %+v", started)
	}
}

func TestAuditStartRechecksProjectHTTPTargetCredentialUse(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, databaseURL)
	profiles := loadAuditServiceProfiles(t)
	gateway, err := profiles.LLMGateway("test-gateway@1")
	if err != nil {
		t.Fatal(err)
	}
	lookup := &switchableCredentialLookup{available: true, gateway: gateway.Ref}
	service, err := New(Options{Storage: artifactpgtest.NewStorage(),
		Pool: pool, Profiles: profiles,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(
			func(pgx.Tx) (resolved.CredentialLookup, error) { return lookup, nil },
		),
		CredentialGuard: &countingCredentialGuard{},
	})
	if err != nil {
		t.Fatal(err)
	}
	cipher, err := managedcredentials.NewTokenCipher([]byte(strings.Repeat("k", 32)))
	if err != nil {
		t.Fatal(err)
	}
	runtimeCredentials, err := managedcredentials.NewRuntimeCredentialService(managedcredentials.RuntimeCredentialServiceOptions{
		Pool: pool, Lease: leasetest.Hold(t, pool).Lease, Cipher: cipher,
		Usage:    managedcredentials.NewRuntimeCredentialRepository(pool),
		Projects: projectstore.NewPostgresStore(pool), Audits: hookstore.NewCredentialHolds(pool, artifactpgtest.NewStorage()), Barrier: managedcredentials.NewLifecycleBarrier(),
	})
	if err != nil {
		t.Fatal(err)
	}
	material, err := managedcredentials.NewHTTPOriginBearerCredential("origin-secret")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtimeCredentials.Create(ctx, managedcredentials.RuntimeCredentialCreateRequest{
		CredentialID: "foreign-origin", Material: material,
		IdempotencyKey: "create-foreign-origin", ActorID: "another-owner",
	}); err != nil {
		t.Fatal(err)
	}

	projects := projectstore.NewPostgresStore(pool)
	project, _, err := projects.Create(ctx, projectstore.CreateParams{
		ProjectID: "project-target-audit", OwnerID: "owner-target-audit", Kind: projectstore.KindProject,
		Name: "Target Audit", IdempotencyKey: "project-target-audit", RequestDigest: serviceTestDigest("target-project"),
	})
	if err != nil {
		t.Fatal(err)
	}
	// Stored before ownership enforcement, or by a principal that has since
	// lost the Operations capability.
	project, err = projects.Update(ctx, projectstore.UpdateParams{
		ProjectID: project.ProjectID, OwnerID: project.OwnerID, ExpectedRevision: project.Revision,
		Name: project.Name, HTTPTarget: &runtimesettings.HTTPOriginTargetRef{
			URL: "https://attacker.example.test",
			Credential: &runtimesettings.RuntimeCredentialRef{
				CredentialID: "foreign-origin", Kind: runtimesettings.RuntimeCredentialOriginBearer,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	projectArtifacts, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)).Project(project.ProjectID)
	if err != nil {
		t.Fatal(err)
	}
	checklist := writeChecklist(t, ctx, projectArtifacts, "target", "automatic")
	draft, _, err := service.CreateDraft(ctx, CreateDraftParams{
		AuditID: "audit-target", OwnerID: project.OwnerID, ProjectID: project.ProjectID,
		Profile:        ProfileSelector{Name: "test-checklist", Version: "1"},
		Inputs:         map[string]contracts.ArtifactRef{"checklist": checklist.Ref},
		Scope:          Scope{Objective: "Exercise Project target credential ownership"},
		IdempotencyKey: "create-target-audit", RequestDigest: serviceTestDigest("target-create"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-target-owner", RequestDigest: serviceTestDigest("target-start-owner"),
	}); !errors.Is(err, managedcredentials.ErrRuntimeCredentialNotFound) {
		t.Fatalf("start with another owner's credential = %v", err)
	}
	started, err := service.Start(ctx, StartParams{
		OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
		IdempotencyKey: "start-target-operations", RequestDigest: serviceTestDigest("target-start-operations"),
		OperationsPrincipal: true,
	})
	if err != nil || started.Audit.State != auditstore.AuditActive {
		t.Fatalf("start by an Operations principal = (%+v, %v)", started.Audit, err)
	}
}

func seedAuditFinding(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, proposedChecks ...auditdomain.ProposedCheck,
) string {
	t.Helper()
	return seedAuditFindingDocument(t, ctx, pool, projectID, ownerID, auditID, suffix,
		auditFindingDocument(suffix, proposedChecks...))
}

func auditFindingDocument(suffix string, proposedChecks ...auditdomain.ProposedCheck) auditdomain.FindingProposal {
	return auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate-" + suffix,
		Title: "Candidate " + suffix, Description: "A retained candidate for review.",
		Subject:       &auditdomain.FindingSubject{Kind: "component", Key: "component-" + suffix},
		Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
		EvidenceIDs: []string{}, ProposedChecks: append([]auditdomain.ProposedCheck{}, proposedChecks...),
		SeveritySuggestion: "medium", Limitations: []string{},
	}
}

func seedAuditFindingDocument(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, document auditdomain.FindingProposal, fingerprints ...auditdomain.FindingFingerprint,
) string {
	t.Helper()
	fingerprint := auditdomain.UnavailableFindingFingerprint("not_recorded")
	if len(fingerprints) == 1 {
		fingerprint = fingerprints[0]
	}
	return seedAuditFindingProjected(t, ctx, pool, projectID, ownerID, auditID, suffix, document, fingerprint, auditdomain.UnavailableFindingAnchors("not_recorded"))
}

func seedAuditFindingProjected(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool,
	projectID, ownerID, auditID, suffix string, document auditdomain.FindingProposal, fingerprint auditdomain.FindingFingerprint, anchors auditdomain.FindingAnchorProjection,
) string {
	t.Helper()
	if len(document.EvidenceIDs) == 0 {
		document.EvidenceIDs = []string{"evidence-1"}
	}
	retainedEvidence := make([]findingintake.ExactArtifact, 0, len(document.EvidenceIDs))
	for _, id := range document.EvidenceIDs {
		body := []byte("Retained evidence " + suffix + " " + id)
		file, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)), projectID, contracts.ArtifactRef{Namespace: artifactpolicy.FindingMaterialsNamespace, Name: suffix + "-" + id}, artifacts.Payload{MediaType: "text/plain", Data: body})
		if err != nil {
			t.Fatal(err)
		}
		retainedEvidence = append(retainedEvidence, findingintake.ExactArtifact{Ref: file.Ref, Digest: auditdomain.DigestBytes(body), MediaType: file.MediaType, SizeBytes: file.Size})
	}
	evidenceJSON, _ := json.Marshal(retainedEvidence)
	payload, err := auditdomain.EncodeFindingProposal(document)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(document))
	if err != nil {
		t.Fatal(err)
	}
	namespace := artifactpolicy.FindingMaterialsNamespace
	written, err := artifactstest.WriteAuditArtifact(ctx, artifacts.NewService(artifactpgtest.NewStorage().Repository(pool)), projectID,
		contracts.ArtifactRef{Namespace: namespace, Name: "candidate-" + suffix},
		artifacts.Payload{MediaType: "application/json", Data: payload})
	if err != nil {
		t.Fatal(err)
	}
	proposal := findingintake.ExactArtifact{
		Ref: written.Ref, Digest: auditdomain.DigestBytes(payload), MediaType: written.MediaType, SizeBytes: written.Size,
	}
	proposalJSON, _ := json.Marshal(proposal)
	proposalRefJSON, _ := json.Marshal(proposal.Ref)
	fingerprintJSON, err := auditdomain.EncodeFindingFingerprint(fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	anchorJSON, err := auditdomain.EncodeFindingAnchors(anchors)
	if err != nil {
		t.Fatal(err)
	}
	receiptID := "receipt-" + suffix
	if _, err := pool.Exec(ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id,
    runtime_instance_id, stage_execution_id, logical_agent_name,
    invocation_id, submission_id, client_key, request_digest,
    run_id, owner_id, project_id,
    workflow_name, workflow_version, workflow_schema_version,
    workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type,
    proposal_size_bytes, evidence, report_projection, fingerprint_projection, anchor_projection,
 retention_state,source_run_deleted_at,project_proposal,project_evidence,publication_requested
, checks_projection) VALUES ($1, $2, $3, 'runtime-review', 'instance-review', 'stage-review', 'worker',
          $4, $5, $6, $7, $8, $9, $10,
          'finding-source', '1', 'contractor/v1alpha1',
          '{"name":"finding-source","version":"1"}'::jsonb, $11,
          $12::jsonb, $13, 'application/json', $14, $18::jsonb, $15, $16,$17,'project-held',clock_timestamp(),$19::jsonb,$18::jsonb,true, json_build_object('schema','contractor.audit.finding-checks-projection.v1','proposal_digest',($13)::text,'size_bytes',($14)::bigint,'discarded',false,'subject',NULL,'limitations',NULL,'checks','[]'::json))`,
		receiptID, "proposal-"+suffix, "allocation-"+suffix,
		"invocation-"+suffix, "submission-"+suffix, document.ClientKey,
		serviceTestDigest("request-"+suffix), "deleted-run-"+suffix, ownerID, projectID,
		serviceTestDigest("workflow-"+suffix), proposalRefJSON, proposal.Digest, proposal.SizeBytes,
		string(projection), string(fingerprintJSON), string(anchorJSON), evidenceJSON, proposalJSON,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES($1,$2,$3,$4::jsonb,$5::jsonb)`, receiptID, auditID, projectID, proposalJSON, evidenceJSON); err != nil {
		t.Fatal(err)
	}
	var findingID string
	err = persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		events := hookstore.NewFindingEventsTx(tx, artifactpgtest.NewStorage())
		var err error
		findingID, err = events.AdmitProjectFinding(ctx, projectID, receiptID, "finding-"+suffix)
		if err != nil {
			return err
		}
		return events.RecordFindingSourceAdded(ctx, auditID, receiptID, "deleted-run-"+suffix)
	})
	if err != nil {
		t.Fatal(err)
	}
	return findingID
}

func decideFindingForTest(
	t *testing.T, ctx context.Context, service *Service,
	ownerID, auditID, findingID string, findingRevision uint64, key string,
	verdict auditstore.AnalystVerdict, severityValue *string, duplicateTarget *string,
) auditstore.ProjectFinding {
	t.Helper()

	var severity *auditstore.FindingSeverity
	if severityValue != nil {
		value := auditstore.FindingSeverity(*severityValue)
		severity = &value
	}
	current, err := service.reads().GetAuditProjectFinding(ctx, ownerID, auditID, findingID)
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.EditProjectFinding(ctx, ProjectFindingEditParams{
		ProjectFindingEdit: auditstore.ProjectFindingEdit{OwnerID: ownerID, ProjectID: current.ProjectID, FindingID: findingID, ExpectedRevision: findingRevision, Verdict: verdict, Severity: severity, Rationale: "Analyst decision for " + key + ".", DuplicateTargetID: duplicateTarget},
		IdempotencyKey:     "decision-" + key, RequestDigest: serviceTestDigest("decision-" + key),
	})
	if err != nil {
		t.Fatal(err)
	}
	return result.Finding
}

func reviewStringPointer(value string) *string { return &value }

type switchableProfileCatalog struct {
	mu        sync.Mutex
	snapshot  *config.Snapshot
	available bool
}

func (c *switchableProfileCatalog) AuditProfile(selector string) (auditprofile.PinnedProgram, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return auditprofile.PinnedProgram{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.AuditProfile(selector)
}

func (c *switchableProfileCatalog) ResolveAuditProfile(_ context.Context, selector string) (auditprofile.PinnedProgram, error) {
	return c.AuditProfile(selector)
}

func (c *switchableProfileCatalog) ResolvePinnedAuditProfile(ctx context.Context, pinned auditprofile.PinnedProgram) (auditprofile.PinnedProgram, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return auditprofile.PinnedProgram{}, config.ErrConfigurationNotFound
	}
	return c.snapshot.ResolvePinnedAuditProfile(ctx, pinned)
}

func (c *switchableProfileCatalog) AuditProfiles() []auditprofile.PinnedProgram {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.available {
		return nil
	}
	return c.snapshot.AuditProfiles()
}

type switchableCredentialLookup struct {
	mu        sync.Mutex
	available bool
	gateway   llmgateway.LLMGatewayConfigRef
}

func (l *switchableCredentialLookup) LookupLLMCredential(
	_ context.Context, id string,
) (resolved.CredentialMetadata, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.available || id != "development-worker" {
		return resolved.CredentialMetadata{}, errors.New("credential unavailable")
	}
	return resolved.CredentialMetadata{
		Ref:          llmgateway.LLMCredentialRef{CredentialID: id},
		LLMGateway:   l.gateway,
		Unrestricted: true,
	}, nil
}

type countingCredentialGuard struct {
	mu    sync.Mutex
	calls int
}

func (g *countingCredentialGuard) WithRunCreation(ctx context.Context, fn func() error) error {
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn()
}

func writeChecklist(
	t *testing.T, ctx context.Context, store artifacts.ScopedStore, name, reviewPolicy string,
) artifacts.WriteResult {
	t.Helper()
	payload := []byte(`{"schema":"contractor.audit.checklist.v1","items":[{"key":"check-` + name + `","version":"1","statement":"Verify ` + name + `.","applicability":"always","allowed_methods":["static"],"required_evidence":[],"review_policy":"` + reviewPolicy + `"}]}`)
	result, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{
		MediaType: "application/json", Data: payload,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func loadAuditServiceProfiles(t *testing.T) *config.Snapshot {
	return loadAuditServiceProfilesWithProfile(t, nil)
}

func loadAuditServiceProfilesWithProfile(t *testing.T, edit func(string) string) *config.Snapshot {
	return loadAuditServiceProfilesWithEdits(t, edit, nil)
}

func loadAuditServiceProfilesWithEdits(
	t *testing.T, editProfile, editWorker func(string) string, customize ...func(map[string]string),
) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"instructions/planner.md": "Execute the selected checklist item.",
		"instructions/worker.md":  "Read the task package and write a result package.",
		"llm-gateways/test.yaml": `apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig
metadata: {name: test-gateway, version: "1"}
spec:
  protocol: openai-compatible@1
  url: http://127.0.0.1:4000/v1
`,
		"model-policies/worker.yaml": `apiVersion: contractor/v1alpha1
kind: ModelPolicy
metadata: {name: worker, version: "1"}
spec:
  model: worker-model
  maxOutputTokens: 1024
  maxModelCalls: 2
  maxToolCalls: 4
  maxTotalTokens: 4096
  temperature: 0
`,
		"agent-templates/worker.yaml": `apiVersion: contractor/v1alpha1
kind: AgentTemplate
metadata: {name: audit-worker, version: "1"}
spec:
  description: Produces one deterministic test result
  runtime: adk@1
  instructions: {ref: instructions/worker.md}
  toolsets:
    - ref: run-artifacts@1
      tools: [read_artifact, write_artifact]
  sandboxProfile: local-workdir@1
`,
		"workflows/check.yaml": `apiVersion: contractor/v1alpha1
kind: Workflow
metadata: {name: audit-check, version: "1"}
spec:
  parameters: {}
  inputs:
    task: {required: true, mediaTypes: [application/json]}
  outputs:
    result: {required: true, mediaTypes: [application/json]}
  executionConfig:
    workers:
      llmGateway: test-gateway@1
      credential: development-worker
      modelPolicy: worker@1
  entryStage: check
  stages:
    check:
      objective: Evaluate one checklist item
      instructions: {ref: instructions/planner.md}
      planner: passthrough@1
      agents:
        worker: {template: audit-worker@1}
      context:
        artifacts:
          task: {namespace: inputs, name: task, required: true}
      result:
        artifacts:
          result: {required: true, mediaTypes: [application/json], from: {namespace: worker, name: result}}
      workflowOutputs: {result: result}
      on:
        succeeded: {succeed: {}}
        failed: {fail: {}}
        interrupted: {fail: {}}
`,
		"audit-profiles/checklist.yaml": `apiVersion: contractor/v1alpha1
kind: AuditProfile
metadata: {name: test-checklist, version: "1"}
spec:
  schema: contractor.audit.program.v1
  inputs:
    checklist: {required: true, media_types: [application/json]}
  workflows:
    check:
      ref: audit-check@1
      inputs:
        task: {source: audit-input, name: checklist}
      parameters: {}
      outputs: {material: result}
  hypotheses: {}
  initial_work:
  - {key: inspect, workflow: check, objective: Inspect selected material}
  limits:
    max_iterations: 1
    batch_size: 1
    max_tasks_per_iteration: 10
    max_tasks: 10
    max_submitted_runs: 20
    max_attempts: 2
    deadline_seconds: 3600
    max_evidence_bytes: 1048576
    max_hypotheses: 1
    max_hypothesis_revisions: 1
    max_hypothesis_bytes: 32768
  policy:
    active_operations: prohibited
`,
	}
	if editProfile != nil {
		files["audit-profiles/checklist.yaml"] = editProfile(files["audit-profiles/checklist.yaml"])
	}
	if editWorker != nil {
		files["agent-templates/worker.yaml"] = editWorker(files["agent-templates/worker.yaml"])
	}
	for _, apply := range customize {
		apply(files)
	}
	for _, directory := range []string{
		"instructions", "llm-gateways", "model-policies",
		"agent-templates", "workflows", "audit-profiles", "skills",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatalf("load Audit service config: %v", err)
	}
	return snapshot
}

func isolatedAuditServicePool(
	t *testing.T, ctx context.Context, databaseURL string,
) *pgxpool.Pool {
	t.Helper()
	pool := postgrestest.NewPool(t, ctx, databaseURL, postgrestest.Options{})
	if _, err := postgres.ApplyMigrations(ctx, pool, postgres.MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	return pool
}

func serviceTestDigest(value string) string {
	return "sha256:" + hex.EncodeToString([]byte(strings.Repeat(value, 64))[:32])
}
