//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

// Suggestions are authored by the actual Runtime tool, retained by intake and
// routed by the actual Controller. The deleted producer cannot supply a latest
// revision. Role names deliberately have no method semantics.
func TestAuditTypedCheckRoutingAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("typed routing process acceptance cannot run in short mode")
	}
	reader := &typedCheckProcessReader{seen: map[string]bool{}}
	gateway := newDomainGateway(llmGatewayToken)
	gateway.stages = []domainGatewayStage{
		typedCheckProducerStage(), reader.stage("verify", "second", 0), reader.stage("replay", "first", 1),
	}
	t.Cleanup(gateway.close)
	h := startConfiguredScanStackWithBudget(t, 9*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), root, gateway.URL())
		h.gatewayToken = llmGatewayToken
		installTypedCheckFixture(t, root)
	})
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Typed checks fixture", "activeChecks": "prohibited"}, 0, http.StatusCreated, &project)
	checklist := auditdomain.ChecklistDocument{Schema: auditdomain.ChecklistSchema, Items: []auditdomain.ChecklistEntry{{Key: "seed", Version: "1", Statement: "Record the controlled fixture observation", Applicability: "fixture", AllowedMethods: []string{"verify"}, RequiredEvidence: []string{"observation"}, ReviewPolicy: "automatic"}}}
	data, err := json.Marshal(checklist)
	if err != nil {
		t.Fatal(err)
	}
	input := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", data)
	var draft auditProgramAudit
	scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "typed-check-fixture", "version": "1"}, "inputs": map[string]artifactRef{"checklist": input}, "scope": map[string]string{"objective": "Verify exact receipt routing through renamed ordinary check roles."}}, 0, http.StatusCreated, &draft)
	removeFault := installScanFault(t, h, "typed_round_admission_fence", "audit_rounds", "INSERT", fmt.Sprintf("IF NEW.ordinal=2 THEN RAISE EXCEPTION USING ERRCODE='%s', MESSAGE='controlled typed admission hold'; END IF;", persistencepostgres.SQLStateSerializationFailure))
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
	var producer string
	waitRoundResultsProcess(t, h, "collected source and closed first Round", func() bool {
		return h.pool.QueryRow(h.ctx, `SELECT execution.run_id FROM audit_executions AS execution JOIN audit_rounds AS round ON round.round_id=execution.round_id WHERE execution.audit_id=$1 AND round.ordinal=1 AND round.state='closed' AND execution.state='collected'`, draft.AuditID).Scan(&producer) == nil
	})
	paused := pauseRoundResultsFixture(t, h, draft.AuditID)
	deleteASVSBacktraceRun(t, h.ctx, h.client, h.baseURL, producer)
	h.restartServer()
	removeFault()
	var resumed auditProgramAudit
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID, &resumed)
	if resumed.State != "paused" || resumed.CurrentRoundID != paused.CurrentRoundID {
		t.Fatalf("restart changed the paused accepted Round: %+v", resumed)
	}
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/resume", map[string]int{"deadlineSeconds": 3600}, resumed.Revision, http.StatusOK, nil)
	waitRoundResultsProcess(t, h, "both typed checks collected", func() bool {
		var count int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_executions AS execution JOIN audit_rounds AS round ON round.round_id=execution.round_id WHERE execution.audit_id=$1 AND round.ordinal=2 AND execution.state='collected'`, draft.AuditID).Scan(&count); err != nil {
			t.Fatal(err)
		}
		var current auditProgramAudit
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+draft.AuditID, &current)
		if current.State == "failed" || current.State == "cancelled" {
			t.Fatalf("typed admission failed: %+v, gateway=%v", current, gateway.Failures())
		}
		return count == 2
	})
	if failures := gateway.Failures(); len(failures) != 0 || gateway.CompletedStages() != 3 {
		t.Fatalf("actual Runtime stages=%d, failures=%v", gateway.CompletedStages(), failures)
	}
	reader.mu.Lock()
	seen := len(reader.seen)
	reader.mu.Unlock()
	if seen != 2 {
		t.Fatalf("typed consumers read %d exact proposals", seen)
	}
	assertTypedCheckConsumerBytes(t, h, draft.AuditID, project.ProjectID, producer)
	assertTypedCheckRejection(t, h, draft.AuditID, project.ProjectID)
	// A restart after acceptance must not append a Round, item, decision or
	// child. Both source ordinals already have their durable associations.
	h.restartServer()
	var items, executions, sources, rounds int
	for query, target := range map[string]*int{
		`SELECT count(*) FROM audit_items WHERE audit_id=$1`:                                     &items,
		`SELECT count(*) FROM audit_executions WHERE audit_id=$1`:                                &executions,
		`SELECT count(*) FROM audit_items WHERE audit_id=$1 AND proposal_receipt_id IS NOT NULL`: &sources,
		`SELECT count(*) FROM audit_rounds WHERE audit_id=$1 AND state<>'proposed'`:              &rounds,
	} {
		if err := h.pool.QueryRow(h.ctx, query, draft.AuditID).Scan(target); err != nil {
			t.Fatal(err)
		}
	}
	if items != 3 || executions != 3 || sources != 2 || rounds != 2 {
		t.Fatalf("duplicate or fallback work after restart: items=%d executions=%d sources=%d rounds=%d", items, executions, sources, rounds)
	}
	assertTypedCheckRejection(t, h, draft.AuditID, project.ProjectID)
}

func assertTypedCheckRejection(t *testing.T, h *scanProcessHarness, auditID, projectID string) {
	t.Helper()
	rows, err := h.pool.Query(h.ctx, `SELECT logical_key,artifact_ref,artifact_digest FROM audit_artifact_links WHERE audit_id=$1 AND source_provenance->>'schema'='contractor.audit.admission-decision.v1' AND source_provenance->>'disposition'='rejected' AND source_provenance->>'ordinal'='2'`, auditID)
	if err != nil {
		t.Fatal(err)
	}
	type decision struct {
		Ref    artifactRef
		Digest string
		Key    string
	}
	decisions := []decision{}
	for rows.Next() {
		var ref []byte
		var value decision
		if err := rows.Scan(&value.Key, &ref, &value.Digest); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(ref, &value.Ref); err != nil {
			t.Fatal(err)
		}
		decisions = append(decisions, value)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if len(decisions) != 1 {
		t.Fatalf("unregistered method decisions = %d, want one", len(decisions))
	}
	value := decisions[0]
	if value.Ref.Revision == nil {
		t.Fatal("decision is not exact")
	}
	target := h.baseURL + "/v1/audits/" + auditID + "/artifacts?" + url.Values{"key": {value.Key}, "revision": {*value.Ref.Revision}}.Encode()
	data, _ := download(t, h.client, target)
	var document map[string]any
	if json.Unmarshal(data, &document) != nil || auditdomain.DigestBytes(data) != value.Digest {
		t.Fatal("decision bytes differ")
	}
	if code, ok := lastStringValue(document, "code"); !ok || code != "unsupported_method" {
		t.Fatalf("wrong explicit decision: %s", data)
	}
}

func typedCheckProducerStage() domainGatewayStage {
	return domainGatewayStage{name: "ordinary-check-produces-typed-suggestions", tools: []string{"read_audit_task", "submit_check_result", "finding", "read_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("finding", fixedArguments(map[string]any{
			"title": "Controlled follow-up observation", "description": "The fixture needs independent verification from exact retained context.",
			"proposed_checks": []map[string]string{
				{"objective": "Verify the controlled observation.", "method": "verify"},
				{"objective": "Replay an independent controlled observation.", "method": "replay"},
				{"objective": "This method has no registered authority.", "method": "unregistered-idea"},
			},
		})),
		toolGatewayStep("submit_check_result", findingResultArguments(map[string]any{"assessment": "violated", "summary": "Controlled observation retained for exact follow-up.", "completed": []string{"observation"}, "gaps": []string{}, "evidence": []map[string]string{{"kind": "observation", "summary": "A controlled fixture observation, with no external target activity."}}})),
		finalGatewayStep("Assigned observation recorded", nil),
	}}
}

type typedCheckProcessReader struct {
	mu   sync.Mutex
	seen map[string]bool
}

func (r *typedCheckProcessReader) stage(method, role string, ordinal int) domainGatewayStage {
	return domainGatewayStage{name: "exact-routed-" + method, tools: []string{"read_audit_task", "submit_check_result", "finding", "read_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "read_audit_task")
			if len(responses) != 1 {
				return nil, fmt.Errorf("expected one exact assigned task")
			}
			task, ok := responses[0]["task"].(map[string]any)
			if !ok || task["workflow_role"] != role {
				return nil, fmt.Errorf("renamed role routing differs")
			}
			plan, _ := task["accepted_plan"].(map[string]any)
			route, _ := plan["route"].(map[string]any)
			selected, _ := plan["plan"].(map[string]any)
			source, _ := plan["source"].(map[string]any)
			sourceOrdinal, exactInteger := source["proposed_check_ordinal"].(json.Number)
			if route["workflow_role"] != role || selected["method"] != method || selected["phase"] != method || !exactInteger || sourceOrdinal.String() != strconv.Itoa(ordinal) {
				return nil, fmt.Errorf("accepted source, method or route drifted: role=%v method=%v phase=%v ordinal=%v", route["workflow_role"], selected["method"], selected["phase"], source["proposed_check_ordinal"])
			}
			return map[string]any{"namespace": "inputs", "name": "proposal", "revision": nil}, nil
		}),
		toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			proposal, err := auditdomain.DecodeFindingProposal(data)
			if err != nil || len(proposal.ProposedChecks) != 3 || proposal.ProposedChecks[ordinal].Method != method {
				return nil, fmt.Errorf("Runtime did not read the original exact proposal: %v", err)
			}
			return map[string]any{"namespace": "audit-context", "name": "retained-dependencies", "revision": nil}, nil
		}),
		toolGatewayStep("submit_check_result", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			var context struct {
				Mappings []struct {
					Name                       string
					Source, Retained, Consumer auditstore.ExactArtifact
				}
			}
			if json.Unmarshal(data, &context) != nil || len(context.Mappings) < 2 {
				return nil, fmt.Errorf("consumer closure is absent")
			}
			found := map[string]bool{}
			for _, mapping := range context.Mappings {
				if mapping.Source.Digest != mapping.Retained.Digest || mapping.Source.Digest != mapping.Consumer.Digest || mapping.Consumer.Ref.Namespace != "audit-context" || mapping.Consumer.Ref.Revision == nil {
					return nil, fmt.Errorf("consumer mapping differs from accepted bytes")
				}
				found[mapping.Name] = true
			}
			if !found["proposal"] || !found["source-result"] {
				return nil, fmt.Errorf("missing accepted proposal/result closure")
			}
			r.mu.Lock()
			defer r.mu.Unlock()
			if r.seen[method] {
				return nil, fmt.Errorf("target check repeated")
			}
			r.seen[method] = true
			return map[string]any{"assessment": "supported", "summary": "Exact retained observation independently checked.", "completed": []string{method}, "gaps": []string{}, "evidence": []map[string]string{{"kind": "observation", "summary": "The accepted proposal and exact producer result are available to this ordinary check."}}}, nil
		}),
		finalGatewayStep("Accepted follow-up recorded", nil),
	}}
}

func assertTypedCheckConsumerBytes(t *testing.T, h *scanProcessHarness, auditID, projectID, producer string) {
	t.Helper()
	project, err := artifacts.NewService(artifactpostgres.NewRepository(h.pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := h.pool.Query(h.ctx, `SELECT execution.run_id FROM audit_executions AS execution JOIN audit_rounds AS round ON round.round_id=execution.round_id WHERE execution.audit_id=$1 AND round.ordinal=2 ORDER BY execution.workflow_role`, auditID)
	if err != nil {
		t.Fatal(err)
	}
	runIDs := []string{}
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		runIDs = append(runIDs, id)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if len(runIDs) != 2 || runIDs[0] == runIDs[1] || runIDs[0] == producer || runIDs[1] == producer {
		t.Fatalf("typed child identity: %v", runIDs)
	}
	for _, runID := range runIDs {
		data, _ := download(t, h.client, h.baseURL+"/v1/runs/"+runID+"/artifacts/audit-context/retained-dependencies")
		var context struct {
			Mappings []struct {
				Name               string
				Retained, Consumer auditstore.ExactArtifact
			}
		}
		if json.Unmarshal(data, &context) != nil {
			t.Fatal("invalid accepted mapping context")
		}
		for _, mapping := range context.Mappings {
			retained := mapping.Retained
			local := mapping.Consumer
			localURL := fmt.Sprintf("%s/v1/runs/%s/artifacts/%s/%s?revision=%s", h.baseURL, runID, url.PathEscape(local.Ref.Namespace), url.PathEscape(local.Ref.Name), url.QueryEscape(*local.Ref.Revision))
			// Protected Audit content is read through its exact retained scope;
			// the generic Project HTTP route deliberately hides that namespace.
			original, err := project.ReadExact(h.ctx, retained)
			if err != nil {
				t.Fatal(err)
			}
			consumed, localMedia := download(t, h.client, localURL)
			if !bytes.Equal(original.Payload.Data, consumed) || original.Payload.MediaType != localMedia || auditdomain.DigestBytes(consumed) != retained.Digest || int64(len(consumed)) != retained.SizeBytes {
				t.Fatalf("exact %s bytes differ after source deletion", mapping.Name)
			}
		}
	}
}

func installTypedCheckFixture(t *testing.T, root string) {
	t.Helper()
	source := filepath.Join(repoRoot(t), "tests/e2e/testdata/check-routing")
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
}
