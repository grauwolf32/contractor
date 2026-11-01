//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

type iterativeExactBytes struct {
	Exact contracts.ExactArtifact
	Data  []byte
}

type iterativeObservation struct {
	Consumer     auditdomain.RoundResultsConsumer
	Page         auditdomain.RoundResultsPage
	ResultMember []byte
}

type iterativeProcessJournal struct {
	mu             sync.Mutex
	producer       string
	receiptID      string
	proposal       iterativeExactBytes
	producerResult iterativeExactBytes
	typedTask      auditdomain.ItemTask
	snapshots      map[int]iterativeObservation
}

// The four ordinary Runs are seed check, first assessment, typed follow-up
// check and second assessment. No result or snapshot row is seeded by the test.
func TestAuditIterativeCompositionAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("iterative composition process acceptance cannot run in short mode")
	}
	journal := &iterativeProcessJournal{snapshots: map[int]iterativeObservation{}}
	gateway := newDomainGateway(llmGatewayToken)
	gateway.stages = []domainGatewayStage{}
	t.Cleanup(gateway.close)
	t.Cleanup(func() {
		if t.Failed() {
			t.Logf("Iterative Gateway observations=%+v failures=%v", gateway.Observations(), gateway.Failures())
		}
	})
	h := startConfiguredScanStackWithBudget(t, 10*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), root, gateway.URL())
		h.gatewayToken = llmGatewayToken
		installTypedCheckFixture(t, root)
		installIterativeFixture(t, h, root)
	})
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Joined iterative fixture", "activeChecks": "prohibited"}, 0, http.StatusCreated, &project)
	step := func(name string, run func(*testing.T)) {
		if !t.Run(name, run) {
			t.FailNow()
		}
	}
	step("iterative_two_rounds", func(t *testing.T) {
		completedBefore := gateway.CompletedStages()
		appendIterativeStages(gateway, iterativeProducerStage(), journal.assessmentStage(1), journal.followUpStage(), journal.assessmentStage(2))
		audit := createIterativeAudit(t, h, project.ProjectID, "iterative-fixture", "joined-seed")
		removeFault := installScanFault(t, h, "iterative_snapshot_publication", "artifact_bindings", "INSERT", fmt.Sprintf("IF NEW.scope_kind='project' AND NEW.name LIKE 'round-results-root-%%' THEN RAISE EXCEPTION USING ERRCODE='%s', MESSAGE='controlled joined snapshot publication hold'; END IF;", persistencepostgres.SQLStateSerializationFailure))
		scanAuditPOST(t, h, "/v1/audits/"+audit.AuditID+"/start", nil, audit.Revision, http.StatusOK, nil)
		waitRoundResultsProcess(t, h, "real collected seed and unpublished captured snapshot", func() bool {
			var producer string
			if err := h.pool.QueryRow(h.ctx, `SELECT execution.run_id FROM audit_executions execution JOIN audit_rounds round ON round.round_id=execution.round_id WHERE execution.audit_id=$1 AND execution.role='check' AND execution.state='collected' AND round.ordinal=1 AND round.results_sealed AND round.results_page_count>0 AND round.results_manifest IS NULL`, audit.AuditID).Scan(&producer); err != nil {
				return false
			}
			journal.mu.Lock()
			journal.producer = producer
			journal.mu.Unlock()
			return true
		})
		paused := pauseRoundResultsFixture(t, h, audit.AuditID)
		captureIterativeOriginals(t, h, journal, audit.AuditID)
		deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, journal.producer)
		h.restartServer()
		removeFault()
		var current auditProgramAudit
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID, &current)
		if current.State != "paused" || current.CurrentRoundID != paused.CurrentRoundID {
			t.Fatal("source deletion or restart changed the paused current Round")
		}
		scanAuditPOST(t, h, "/v1/audits/"+audit.AuditID+"/resume", map[string]int{"deadlineSeconds": 3600}, current.Revision, http.StatusOK, nil)
		waitRoundResultsProcess(t, h, "four collected joined Runs", func() bool {
			if failures := gateway.Failures(); len(failures) != 0 {
				t.Fatalf("scripted joined Gateway contract failed: %v", failures)
			}
			var count int
			if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_executions WHERE audit_id=$1 AND state='collected'`, audit.AuditID).Scan(&count); err != nil {
				t.Fatal(err)
			}
			var current auditProgramAudit
			auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID, &current)
			if current.State == "failed" || current.State == "cancelled" || current.State == "completed" && count != 4 {
				t.Fatalf("joined journey failed: state=%s failures=%v", current.State, gateway.Failures())
			}
			return count == 4
		})
		if failures := gateway.Failures(); len(failures) != 0 || gateway.CompletedStages() != completedBefore+4 {
			t.Fatalf("joined Runtime stages=%d failures=%v", gateway.CompletedStages(), failures)
		}
		assertIterativeAttribution(t, h, journal, audit.AuditID, project.ProjectID)
		// Collected Runs precede Round closure and report finalization. Reviews
		// are fenced during finalization, so wait for automatic report acceptance
		// before reading a revision-stable finding page and requesting review.
		waitRoundResultsProcess(t, h, "automatically completed Audit before finding review", func() bool {
			var current auditProgramAudit
			auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID, &current)
			if current.State == "failed" || current.State == "cancelled" {
				t.Fatalf("joined Audit did not complete: state=%s", current.State)
			}
			return current.State == "completed"
		})
		confirmIterativeFinding(t, h, audit.AuditID)
		var reviewed auditProgramAudit
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID, &reviewed)
		if reviewed.State != "completed" {
			t.Fatal("explicit finding review reopened the completed Audit")
		}
		var report auditProgramReport
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID+"/report", &report)
		if report.Status != "ready" || len(report.Machine) == 0 {
			t.Fatal("completed Audit report is not readable after explicit finding review")
		}
	})
	step("direct_model_free_bypass", func(t *testing.T) {
		before := gateway.Calls()
		id := h.createRun(t, "iterative-model-free-fixture@1", map[string]string{"target": h.targetURL}, nil)
		status, _ := h.completedReport(t, id, "scan_nuclei")
		h.assertExecution(t, status, "scan_nuclei", "")
		if gateway.Calls() != before {
			t.Fatal("model-free Run invoked the model Gateway")
		}
		var ordinary bool
		if err := h.pool.QueryRow(h.ctx, `SELECT audit_execution_id IS NULL AND audit_input_materialization IS NULL FROM workflow_runs WHERE run_id=$1`, id).Scan(&ordinary); err != nil || !ordinary {
			t.Fatalf("standalone model-free Run acquired Audit initialization: %t %v", ordinary, err)
		}
		assertIterativeRunHasNoContext(t, h, id)
	})
	step("default_audit_bypass", func(t *testing.T) {
		completedBefore := gateway.CompletedStages()
		appendIterativeStages(gateway, iterativeDefaultStage())
		audit := createIterativeAudit(t, h, project.ProjectID, "iterative-default-fixture", "default-seed")
		scanAuditPOST(t, h, "/v1/audits/"+audit.AuditID+"/start", nil, audit.Revision, http.StatusOK, nil)
		waitRoundResultsProcess(t, h, "unchanged deterministic inventory Audit", func() bool {
			if failures := gateway.Failures(); len(failures) != 0 {
				t.Fatalf("scripted default Gateway contract failed: %v", failures)
			}
			var current auditProgramAudit
			auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+audit.AuditID, &current)
			return current.State == "completed"
		})
		assertIterativeDefaultBypass(t, h, audit.AuditID)
		if gateway.CompletedStages() != completedBefore+1 || len(gateway.Failures()) != 0 {
			t.Fatalf("bypass added an optional Worker: stages=%d failures=%v", gateway.CompletedStages(), gateway.Failures())
		}
	})
}

func appendIterativeStages(gateway *domainGateway, stages ...domainGatewayStage) {
	gateway.mu.Lock()
	defer gateway.mu.Unlock()
	gateway.stages = append(gateway.stages, stages...)
}

func createIterativeAudit(t *testing.T, h *scanProcessHarness, projectID, profile, key string) auditProgramAudit {
	t.Helper()
	checklist := auditdomain.ChecklistDocument{Schema: auditdomain.ChecklistSchema, Items: []auditdomain.ChecklistEntry{{Key: key, Version: "1", Statement: "Record one controlled fixture observation", Applicability: "fixture", AllowedMethods: []string{"verify"}, RequiredEvidence: []string{"observation"}, ReviewPolicy: "automatic"}}}
	data, err := json.Marshal(checklist)
	if err != nil {
		t.Fatal(err)
	}
	input := uploadProjectScopeArtifact(t, h.client, h.baseURL, projectID, "fixture", key, "application/json", data)
	var draft auditProgramAudit
	scanAuditPOST(t, h, "/v1/projects/"+projectID+"/audits", map[string]any{"profile": map[string]string{"name": profile, "version": "1"}, "inputs": map[string]artifactRef{"checklist": input}, "scope": map[string]string{"objective": "Prove exact generic iterative composition without research authority."}}, 0, http.StatusCreated, &draft)
	return draft
}

func captureIterativeOriginals(t *testing.T, h *scanProcessHarness, journal *iterativeProcessJournal, auditID string) {
	t.Helper()
	journal.mu.Lock()
	defer journal.mu.Unlock()
	service := artifacts.NewService(artifacts.NewPostgresRepository(h.pool))
	run, _ := service.Run(journal.producer)
	result, err := run.Read(h.ctx, contracts.ArtifactRef{Namespace: "outputs", Name: "result"})
	if err != nil {
		t.Fatal(err)
	}
	journal.producerResult = iterativeExactBytes{Exact: contracts.ExactContent(result.Ref, result.Payload.MediaType, result.Payload.Data), Data: bytes.Clone(result.Payload.Data)}
	var original []byte
	var proposal contracts.ExactArtifact
	if err := h.pool.QueryRow(h.ctx, `SELECT receipt_id,proposal_ref,proposal_digest,proposal_media_type,proposal_size_bytes FROM finding_proposal_receipts WHERE audit_id=$1 AND run_id=$2`, auditID, journal.producer).Scan(&journal.receiptID, &original, &proposal.Digest, &proposal.MediaType, &proposal.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if err := strictjson.Decode(original, &proposal.Ref); err != nil {
		t.Fatal(err)
	}
	read, err := run.ReadExact(h.ctx, proposal)
	if err != nil {
		t.Fatal(err)
	}
	journal.proposal = iterativeExactBytes{Exact: proposal, Data: bytes.Clone(read.Payload.Data)}
}

func installIterativeFixture(t *testing.T, h *scanProcessHarness, root string) {
	t.Helper()
	source := filepath.Join(repoRoot(t), "tests/e2e/testdata/iterative")
	for _, directory := range []string{"agent-templates", "workflows", "instructions", "audit-profiles"} {
		entries, err := os.ReadDir(filepath.Join(source, directory))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			data, err := os.ReadFile(filepath.Join(source, directory, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, directory, entry.Name()), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	python := os.Getenv("CONTRACTOR_SCAN_PYTHON")
	if python == "" {
		python = filepath.Join(h.repositoryRoot, "runtime/.venv/bin/python")
	}
	binaryRoot := filepath.Join(h.temporaryRoot, "iterative-bin")
	if err := os.Mkdir(binaryRoot, 0700); err != nil {
		t.Fatal(err)
	}
	script := "#!" + python + "\n" + `import json, sys
if '-version' in sys.argv:
    print('Nuclei Engine Version: v3.11.1')
else:
    print(json.dumps({'template-id': 'contractor-local-fixture', 'info': {'severity': 'info'}}))
`
	if err := os.WriteFile(filepath.Join(binaryRoot, "nuclei"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	h.runtimePathPrefix = binaryRoot
}
