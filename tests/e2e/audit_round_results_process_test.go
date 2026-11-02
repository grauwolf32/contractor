//go:build e2e

package e2e

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

const roundResultsProcessPages = 129

// The source fixture contains one small settled item per page. Production
// capture at 4096/100000 items is covered separately by the store scale test.
// Here the actual Controller, Run Service and Runtime must read every byte
// of >128 local pages through ordinary Run-scoped Artifact tools.
func TestAuditRoundResultsAcrossProductionProcesses(t *testing.T) {
	if testing.Short() {
		t.Fatal("snapshot process acceptance cannot run in short mode")
	}
	reader := &roundResultsProcessReader{seen: make(map[string]bool)}
	gateway := newDomainGateway(llmGatewayToken)
	gateway.stages = []domainGatewayStage{reader.stage()}
	t.Cleanup(gateway.close)
	h := startConfiguredScanStackWithBudget(t, 9*time.Minute, "", func(h *scanProcessHarness, root string) {
		stageE2EConfiguration(t, filepath.Join(h.repositoryRoot, "configs"), root, gateway.URL())
		h.gatewayToken = llmGatewayToken
		installRoundResultsReaderFixture(t, root)
	})
	var project projectResourceResponse
	scanAuditPOST(t, h, "/v1/projects", map[string]string{"kind": "project", "name": "Exact snapshot fixture", "activeChecks": "prohibited"}, 0, http.StatusCreated, &project)
	checklist := auditdomain.ChecklistDocument{Schema: auditdomain.ChecklistSchema, Items: make([]auditdomain.ChecklistEntry, roundResultsProcessPages)}
	for i := range checklist.Items {
		checklist.Items[i] = auditdomain.ChecklistEntry{Key: fmt.Sprintf("snapshot-%03d", i), Version: "1", Statement: "Controlled snapshot transport fixture", Applicability: "manual fixture", AllowedMethods: []string{"code-review"}, RequiredEvidence: []string{"code-review"}, ReviewPolicy: "manual"}
	}
	checklistBytes, _ := json.Marshal(checklist)
	inputs := map[string]artifactRef{"source": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "source", "application/zip", compositionSourceArchive(t)), "checklist": uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "fixture", "checklist", "application/json", checklistBytes)}
	var draft auditProgramAudit
	scanAuditPOST(t, h, "/v1/projects/"+project.ProjectID+"/audits", map[string]any{"profile": map[string]string{"name": "snapshot-reader-fixture", "version": "1"}, "inputs": inputs, "scope": map[string]string{"objective": "Read the complete exact settled Round through one declared input."}}, 0, http.StatusCreated, &draft)
	scanAuditPOST(t, h, "/v1/audits/"+draft.AuditID+"/start", nil, draft.Revision, http.StatusOK, nil)
	paused := pauseRoundResultsFixture(t, h, draft.AuditID)
	source := seedRoundResultsProcessSnapshot(t, h, paused)
	// The first committed slice remains visible while later writes are held.
	removeFault := installScanFault(t, h, "round_results_input_fence", "artifact_bindings", "INSERT", fmt.Sprintf(`IF NEW.scope_kind='run' AND NEW.namespace='audit-context' AND EXISTS (
 SELECT 1 FROM workflow_runs AS run WHERE run.run_id=NEW.scope_id
 AND run.audit_input_materialization IS NOT NULL
 AND NOT (run.audit_input_materialization->>'complete')::boolean
) THEN RAISE EXCEPTION USING ERRCODE='%s', MESSAGE='controlled exact input continuation hold'; END IF;`, persistencepostgres.SQLStateSerializationFailure))
	scanAuditPOST(t, h, "/v1/audits/"+paused.AuditID+"/resume", map[string]int{"deadlineSeconds": 3600}, paused.Revision, http.StatusOK, nil)
	var child string
	waitRoundResultsProcess(t, h, "durable partial input closure", func() bool {
		var state string
		var progress []byte
		if h.pool.QueryRow(h.ctx, `SELECT run.run_id,run.state,run.audit_input_materialization FROM workflow_runs AS run
JOIN audit_executions AS execution ON execution.execution_id=run.audit_execution_id
WHERE execution.audit_id=$1 AND execution.role='assessment'`, paused.AuditID).Scan(&child, &state, &progress) != nil {
			return false
		}
		var value struct {
			Complete bool   `json:"complete"`
			Revision uint32 `json:"revision"`
		}
		if json.Unmarshal(progress, &value) != nil || value.Revision == 0 || value.Complete {
			return false
		}
		if state != "initializing" {
			t.Fatalf("partial closure became schedulable: %s", state)
		}
		var stages int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM stage_executions WHERE run_id=$1`, child).Scan(&stages); err != nil {
			t.Fatal(err)
		}
		if stages != 0 || gateway.Calls() != 0 {
			t.Fatalf("Stage/model ran before complete closure: %d/%d", stages, gateway.Calls())
		}
		return true
	})
	h.restartServer()
	var sameChild string
	if err := h.pool.QueryRow(h.ctx, `SELECT run_id FROM audit_executions WHERE audit_id=$1 AND role='assessment'`, paused.AuditID).Scan(&sameChild); err != nil || sameChild != child {
		t.Fatalf("restart changed child: %s,%v", sameChild, err)
	}
	removeFault()
	waitRoundResultsProcess(t, h, "assessment read every exact local page", func() bool {
		var audit auditProgramAudit
		if auditProgramTryGET(h.ctx, h.client, h.baseURL+"/v1/audits/"+paused.AuditID, &audit) != nil {
			return false
		}
		if audit.State == "failed" || audit.State == "cancelled" {
			t.Fatalf("snapshot Audit failed: %+v, gateway=%v", audit, gateway.Failures())
		}
		return audit.State == "completed"
	})
	if failures := gateway.Failures(); len(failures) != 0 {
		t.Fatalf("ordinary reader failed: %v", failures)
	}
	if gateway.CompletedStages() != 1 || len(reader.seen) != roundResultsProcessPages || reader.readPages != roundResultsProcessPages {
		t.Fatalf("consumer inspected pages=%d items=%d stages=%d", reader.readPages, len(reader.seen), gateway.CompletedStages())
	}
	if !reader.consumer.Source.Ref.SameExact(source.Ref) || reader.consumer.Source.Digest != source.Digest {
		t.Fatal("consumer lost its accepted source root")
	}
	var children, slots, inputBindings int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_executions WHERE audit_id=$1 AND role='assessment'`, paused.AuditID).Scan(&children); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM workflow_runs, jsonb_object_keys(workflow_snapshot->'inputs') WHERE run_id=$1`, child).Scan(&slots); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='inputs'`, child).Scan(&inputBindings); err != nil {
		t.Fatal(err)
	}
	if children != 1 || slots != 1 || inputBindings != 1 {
		t.Fatalf("extra children/Workflow mappings/input bindings: %d/%d/%d", children, slots, inputBindings)
	}
}

func waitRoundResultsProcess(t *testing.T, h *scanProcessHarness, description string, ready func() bool) {
	t.Helper()
	ticker := time.NewTicker(150 * time.Millisecond)
	defer ticker.Stop()
	for !ready() {
		select {
		case <-h.ctx.Done():
			t.Fatalf("waiting for %s: %v\nserver:%s\nruntime:%s", description, h.ctx.Err(), h.server.logs.redacted(publicToken), h.runtime.logs.redacted(publicToken))
		case <-ticker.C:
		}
	}
}

func pauseRoundResultsFixture(t *testing.T, h *scanProcessHarness, id string) auditProgramAudit {
	t.Helper()
	for range 20 {
		var audit auditProgramAudit
		auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+id, &audit)
		request, _ := http.NewRequestWithContext(h.ctx, http.MethodPost, h.baseURL+"/v1/audits/"+id+"/pause", nil)
		request.Header.Set("Authorization", "Bearer "+publicToken)
		request.Header.Set("Idempotency-Key", "snapshot-pause-"+randomHex(t, 8))
		request.Header.Set("If-Match", fmt.Sprintf("\"%d\"", audit.Revision))
		response, err := h.client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode == http.StatusOK {
			var paused auditProgramAudit
			decodeAuditProgramResponse(t, response, &paused)
			response.Body.Close()
			return paused
		}
		response.Body.Close()
		if response.StatusCode != http.StatusPreconditionFailed && response.StatusCode != http.StatusConflict {
			t.Fatalf("pause fixture=%d", response.StatusCode)
		}
	}
	t.Fatal("could not pause manual-only Round")
	return auditProgramAudit{}
}

func seedRoundResultsProcessSnapshot(t *testing.T, h *scanProcessHarness, audit auditProgramAudit) auditstore.ExactArtifact {
	t.Helper()
	tx, err := h.pool.Begin(h.ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(h.ctx) }()
	var revision uint64
	if err = tx.QueryRow(h.ctx, `SELECT revision FROM audits WHERE audit_id=$1 AND state='paused' FOR UPDATE`, audit.AuditID).Scan(&revision); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(h.ctx, `UPDATE audit_items SET state='settled',final_disposition='excluded',coverage_status='excluded',coverage_gaps='["fixture-not-tested"]'::jsonb WHERE audit_id=$1`, audit.AuditID); err != nil {
		t.Fatal(err)
	}
	rows, err := tx.Query(h.ctx, `SELECT item_id,item_key,ordinal,kind,subject_key FROM audit_items WHERE audit_id=$1 AND round_id=$2 ORDER BY ordinal`, audit.AuditID, audit.CurrentRoundID)
	if err != nil {
		t.Fatal(err)
	}
	items := []auditdomain.RoundResultItem{}
	for rows.Next() {
		item := auditdomain.RoundResultItem{FinalDisposition: "excluded", Coverage: auditdomain.RoundResultCoverage{Status: "excluded", Requested: []string{"code-review"}, Completed: []string{}, Gaps: []string{"fixture-not-tested"}}, Attachments: []auditdomain.RoundResultAttachment{}, Findings: []auditdomain.RoundResultFinding{}}
		if err = rows.Scan(&item.ItemID, &item.ItemKey, &item.Ordinal, &item.Kind, &item.SubjectKey); err != nil {
			t.Fatal(err)
		}
		items = append(items, item)
	}
	if err = rows.Err(); err != nil {
		t.Fatal(err)
	}
	rows.Close()
	if len(items) != roundResultsProcessPages {
		t.Fatalf("fixture membership=%d", len(items))
	}
	service := artifacts.NewService(artifactpostgres.NewRepository(tx))
	scope, _ := artifacts.ProjectScope(audit.ProjectID)
	manifest := auditdomain.RoundResultsManifest{Schema: auditdomain.RoundResultsSchema, AuditID: audit.AuditID, RoundID: audit.CurrentRoundID, SourceGeneration: revision, ItemCount: len(items), Pages: []auditdomain.RoundResultsPageRef{}}
	for index, item := range items {
		page := auditdomain.RoundResultsPage{Schema: auditdomain.RoundResultsPageSchema, AuditID: audit.AuditID, RoundID: audit.CurrentRoundID, SourceGeneration: revision, Ordinal: index, Items: []auditdomain.RoundResultItem{item}}
		data, err := auditdomain.EncodeRoundResultsPage(page)
		if err != nil {
			t.Fatal(err)
		}
		result, err := service.CreateFrozen(h.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: fmt.Sprintf("fixture-snapshot-page-%03d", index)}, artifacts.Payload{MediaType: "application/json", Data: data})
		if err != nil {
			t.Fatal(err)
		}
		exact := contracts.ExactContent(result.Ref, result.MediaType, data)
		manifest.Pages = append(manifest.Pages, auditdomain.RoundResultsPageRef{Ordinal: index, ItemCount: 1, Artifact: exact})
		manifest.MetadataBytes += int64(len(data))
		encoded, _ := json.Marshal(exact)
		if _, err = tx.Exec(h.ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
VALUES($1,$2,$3::jsonb,$4,$5,$6,'{"schema":"contractor.audit.round-results-provenance.v1"}'::jsonb)`, audit.AuditID, fmt.Sprintf("round-results/%s/page/%d", audit.CurrentRoundID, index), mustSnapshotFixtureJSON(exact.Ref), exact.Digest, exact.MediaType, exact.SizeBytes); err != nil {
			t.Fatal(err)
		}
		if _, err = tx.Exec(h.ctx, `INSERT INTO audit_round_result_pages(audit_id,round_id,source_generation,ordinal,last_item_ordinal,item_count,content,artifact,content_bytes) VALUES($1,$2,$3,$4,$5,1,$6,$7::jsonb,octet_length($6::bytea))`, audit.AuditID, audit.CurrentRoundID, revision, index, item.Ordinal, data, encoded); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = tx.Exec(h.ctx, `UPDATE audit_rounds SET state='assessing',results_generation=$2,results_sealed=true,results_item_count=$3,results_metadata_bytes=$4,results_page_count=$3 WHERE round_id=$1`, audit.CurrentRoundID, revision, len(items), manifest.MetadataBytes); err != nil {
		t.Fatal(err)
	}
	for _, page := range manifest.Pages {
		data, _ := json.Marshal(page)
		if _, err = tx.Exec(h.ctx, `UPDATE audit_rounds SET results_published_pages=results_published_pages||jsonb_build_array($2::jsonb) WHERE round_id=$1`, audit.CurrentRoundID, data); err != nil {
			t.Fatal(err)
		}
	}
	data, err := auditdomain.EncodeRoundResultsManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.CreateFrozen(h.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(audit.AuditID), Name: "fixture-snapshot-root"}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		t.Fatal(err)
	}
	exact := contracts.ExactContent(result.Ref, result.MediaType, data)
	encoded, _ := json.Marshal(exact)
	if _, err = tx.Exec(h.ctx, `INSERT INTO audit_artifact_links(audit_id,logical_key,artifact_ref,artifact_digest,media_type,size_bytes,source_provenance)
VALUES($1,$2,$3::jsonb,$4,$5,$6,'{"schema":"contractor.audit.round-results-provenance.v1"}'::jsonb)`, audit.AuditID, "round-results/"+audit.CurrentRoundID, mustSnapshotFixtureJSON(exact.Ref), exact.Digest, exact.MediaType, exact.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(h.ctx, `UPDATE audit_rounds SET results_manifest=$2::jsonb WHERE round_id=$1`, audit.CurrentRoundID, encoded); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(h.ctx, `UPDATE audits SET retained_evidence_bytes=retained_evidence_bytes+$2 WHERE audit_id=$1`, audit.AuditID, manifest.MetadataBytes+exact.SizeBytes); err != nil {
		t.Fatal(err)
	}
	if err = tx.Commit(h.ctx); err != nil {
		t.Fatal(err)
	}
	return exact
}

type roundResultsProcessReader struct {
	consumer  auditdomain.RoundResultsConsumer
	seen      map[string]bool
	readPages int
}

func (r *roundResultsProcessReader) stage() domainGatewayStage {
	steps := []domainGatewayStep{toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "snapshot", "revision": nil}))}
	for ordinal := range roundResultsProcessPages {
		ordinal := ordinal
		steps = append(steps, toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			if ordinal == 0 {
				err = strictjson.Decode(data, &r.consumer)
				if err == nil {
					_, err = auditdomain.EncodeRoundResultsConsumer(r.consumer)
				}
				if err != nil {
					return nil, err
				}
				if len(r.consumer.Pages) != roundResultsProcessPages || len(r.consumer.Mappings) != roundResultsProcessPages {
					return nil, fmt.Errorf("incomplete consumer root")
				}
			} else if err = r.inspectPage(data, ordinal-1); err != nil {
				return nil, err
			}
			page := r.consumer.Pages[ordinal]
			if page.Artifact.Ref.Namespace != "audit-context" {
				return nil, fmt.Errorf("page is not a visible trusted local artifact")
			}
			mapping := r.consumer.Mappings[ordinal]
			if !mapping.Consumer.Ref.SameExact(page.Artifact.Ref) || mapping.Consumer.Digest != page.Artifact.Digest {
				return nil, fmt.Errorf("consumer page mapping differs")
			}
			return map[string]any{"namespace": page.Artifact.Ref.Namespace, "name": page.Artifact.Ref.Name, "revision": *page.Artifact.Ref.Revision}, nil
		}))
	}
	steps = append(steps, toolGatewayStep("write_artifact", func(request map[string]any) (map[string]any, error) {
		data, err := roundResultsToolBytes(request)
		if err != nil {
			return nil, err
		}
		if err = r.inspectPage(data, roundResultsProcessPages-1); err != nil {
			return nil, err
		}
		if len(r.seen) != roundResultsProcessPages {
			return nil, fmt.Errorf("incomplete exact item closure")
		}
		return map[string]any{"namespace": "reader", "name": "report", "media_type": "text/plain", "data_base64": base64.StdEncoding.EncodeToString([]byte("Read every exact Run-local snapshot page.")), "expected_revision": nil}, nil
	}), finalGatewayStep("Read every exact snapshot page", map[string]domainArtifactBinding{"report": {namespace: "reader", name: "report"}}))
	return domainGatewayStage{name: "exact-round-results-reader", tools: []string{"read_artifact", "write_artifact"}, steps: steps}
}

func roundResultsToolBytes(request map[string]any) ([]byte, error) {
	value, ok := lastStringValue(request, "dataBase64")
	if !ok {
		return nil, fmt.Errorf("ordinary Artifact read returned no bytes")
	}
	return base64.StdEncoding.DecodeString(value)
}

func (r *roundResultsProcessReader) inspectPage(data []byte, ordinal int) error {
	page, err := auditdomain.DecodeRoundResultsPage(data)
	if err != nil {
		return err
	}
	expected := r.consumer.Pages[ordinal]
	if page.Ordinal != ordinal || page.AuditID != r.consumer.AuditID || page.RoundID != r.consumer.RoundID || page.SourceGeneration != r.consumer.SourceGeneration || auditdomain.DigestBytes(data) != expected.Artifact.Digest || int64(len(data)) != expected.Artifact.SizeBytes {
		return fmt.Errorf("local exact page content/provenance differs")
	}
	for _, item := range page.Items {
		if r.seen[item.ItemID] || item.FinalDisposition != "excluded" || item.Result != nil {
			return fmt.Errorf("duplicate item or invented assessment")
		}
		r.seen[item.ItemID] = true
	}
	r.readPages++
	return nil
}

func installRoundResultsReaderFixture(t *testing.T, root string) {
	t.Helper()
	source := filepath.Join(repoRoot(t), "tests/e2e/testdata/round-results")
	for _, folder := range []string{"agent-templates", "workflows", "instructions"} {
		entries, err := os.ReadDir(filepath.Join(source, folder))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			data, err := os.ReadFile(filepath.Join(source, folder, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(root, folder, entry.Name()), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	data, err := os.ReadFile(filepath.Join(root, "audit-profiles/source_checklist.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	profile := strings.Replace(string(data), "name: source-checklist", "name: snapshot-reader-fixture", 1)
	profile = strings.ReplaceAll(profile, "maxItemsPerRound: 128", "maxItemsPerRound: 256")
	profile = strings.ReplaceAll(profile, "maxItemsTotal: 128", "maxItemsTotal: 256")
	profile = strings.Replace(profile, "  execution:\n", `    assess:
      kind: assessment
      ref: snapshot-reader-fixture@1
      inputs:
        snapshot: {source: round-results}
      parameters: {}
      outputs: {result: result}
  execution:
`, 1)
	if err = os.WriteFile(filepath.Join(root, "audit-profiles/snapshot_reader_fixture.yaml"), []byte(profile), 0600); err != nil {
		t.Fatal(err)
	}
}

func mustSnapshotFixtureJSON(value any) []byte { data, _ := json.Marshal(value); return data }
