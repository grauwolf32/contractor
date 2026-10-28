//go:build e2e

package e2e

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func assertIterativeAttribution(t *testing.T, h *scanProcessHarness, j *iterativeProcessJournal, auditID, projectID string) {
	t.Helper()
	j.mu.Lock()
	defer j.mu.Unlock()
	if len(j.snapshots) != 2 || j.typedTask.AcceptedPlan == nil || j.typedTask.Dependencies == nil {
		t.Fatal("the production chain did not consume both snapshots and the accepted typed task")
	}
	project, err := artifacts.NewService(artifacts.NewPostgresRepository(h.pool)).Project(projectID)
	if err != nil {
		t.Fatal(err)
	}
	for ordinal := 1; ordinal <= 2; ordinal++ {
		observation := j.snapshots[ordinal]
		consumer := observation.Consumer
		var rootJSON []byte
		var generation uint64
		var assessmentRun string
		if err := h.pool.QueryRow(h.ctx, `SELECT round.results_manifest,round.results_generation,execution.run_id FROM audit_rounds round JOIN audit_executions execution ON execution.round_id=round.round_id AND execution.role='assessment' WHERE round.audit_id=$1 AND round.ordinal=$2`, auditID, ordinal).Scan(&rootJSON, &generation, &assessmentRun); err != nil {
			t.Fatal(err)
		}
		var source contracts.ExactArtifact
		if strictjson.Decode(rootJSON, &source) != nil || consumer.Source.Verify(source) != nil || consumer.SourceGeneration != generation || consumer.AuditID != auditID {
			t.Fatal("ordinary assessment did not consume the accepted coherent source root")
		}
		root, err := project.ReadExact(h.ctx, source)
		if err != nil {
			t.Fatal(err)
		}
		manifest, err := auditdomain.DecodeRoundResultsManifest(root.Payload.Data)
		if err != nil || manifest.ItemCount != 1 || len(manifest.Pages) != 1 || manifest.Pages[0].Artifact.Verify(consumer.Mappings[0].Source) != nil {
			t.Fatalf("incomplete captured source membership: %v", err)
		}
		pageRead, err := project.ReadExact(h.ctx, manifest.Pages[0].Artifact)
		if err != nil {
			t.Fatal(err)
		}
		page, err := auditdomain.DecodeRoundResultsPage(pageRead.Payload.Data)
		if err != nil || len(page.Items) != 1 || page.Items[0].ItemID != observation.Page.Items[0].ItemID {
			t.Fatalf("consumer changed the selected item: %v", err)
		}
		item := page.Items[0]
		attempt := item.SelectedAttempt
		if attempt == nil || item.Result == nil || attempt.RunID == nil || attempt.SourceResult == nil {
			t.Fatal("accepted source snapshot has no authoritative result attribution")
		}
		var itemID, executionID, roundID, receiptID, runID, manifestDigest string
		var number int
		if err := h.pool.QueryRow(h.ctx, `SELECT member.item_id,member.execution_id,execution.round_id,execution.collection_receipt_id,execution.run_id,member.item_attempt,execution.manifest_digest FROM audit_execution_items member JOIN audit_executions execution USING(execution_id) WHERE member.execution_item_id=$1 AND execution.audit_id=$2`, attempt.ExecutionItemID, auditID).Scan(&itemID, &executionID, &roundID, &receiptID, &runID, &number, &manifestDigest); err != nil {
			t.Fatal(err)
		}
		if itemID != item.ItemID || executionID != attempt.ExecutionID || roundID != consumer.RoundID || receiptID != attempt.ReceiptID || runID != *attempt.RunID || number != attempt.Attempt {
			t.Fatal("snapshot guessed item, execution, attempt or receipt attribution")
		}
		retained, err := project.ReadExact(h.ctx, *item.Result)
		if err != nil {
			t.Fatal(err)
		}
		setPackage, err := auditdomain.ValidatePackage(retained.Payload.Data)
		if err != nil {
			t.Fatal(err)
		}
		member, _ := setPackage.MemberByID(auditdomain.CheckResultsMemberID)
		if !bytes.Equal(bytes.TrimSpace(member.Data()), bytes.TrimSpace(observation.ResultMember)) {
			t.Fatal("Runtime snapshot consumer read a different accepted result member")
		}
		set, err := auditdomain.DecodeCheckResultSet(member.Data())
		if err != nil || set.ExecutionManifestDigest != manifestDigest {
			t.Fatalf("accepted result belongs to another execution manifest: %v", err)
		}
		if ordinal == 1 && (!bytes.Equal(j.producerResult.Data, retained.Payload.Data) || attempt.SourceResult.Verify(j.producerResult.Exact) != nil) {
			t.Fatal("first snapshot lost original-to-retained result bytes after producer deletion")
		}
		// Page content is projected because exact refs change scope. Each
		// declared result/attachment mapping still preserves the exact bytes.
		for _, mapping := range observation.Page.Mappings {
			original, err := project.ReadExact(h.ctx, mapping.Source)
			if err != nil {
				t.Fatal(err)
			}
			assertIterativeLocalBytes(t, h, assessmentRun, mapping.Consumer, original.Payload.Data)
		}
		var slots, bindings int
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM workflow_runs,jsonb_object_keys(workflow_snapshot->'inputs') WHERE run_id=$1`, assessmentRun).Scan(&slots); err != nil {
			t.Fatal(err)
		}
		if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='inputs'`, assessmentRun).Scan(&bindings); err != nil || slots != 1 || bindings != 1 {
			t.Fatalf("assessment required extra Workflow slots: %d/%d %v", slots, bindings, err)
		}
	}
	first := j.snapshots[1].Page.Items[0]
	var child string
	if err := h.pool.QueryRow(h.ctx, `SELECT execution.run_id FROM audit_executions execution JOIN audit_rounds round ON round.round_id=execution.round_id WHERE execution.audit_id=$1 AND round.ordinal=2 AND execution.role='check'`, auditID).Scan(&child); err != nil {
		t.Fatal(err)
	}
	contextBytes, _ := download(t, h.client, h.baseURL+"/v1/runs/"+child+"/artifacts/audit-context/retained-dependencies")
	var context struct {
		Mappings []struct {
			Name                       string
			Source, Retained, Consumer contracts.ExactArtifact
		}
	}
	if err := json.Unmarshal(contextBytes, &context); err != nil || len(context.Mappings) != len(j.typedTask.Dependencies.Dependencies) {
		t.Fatalf("consumer did not materialize the entire admitted dependency closure: %v", err)
	}
	for _, mapping := range context.Mappings {
		var accepted *auditdomain.RetainedDependency
		for i := range j.typedTask.Dependencies.Dependencies {
			if j.typedTask.Dependencies.Dependencies[i].Name == mapping.Name {
				accepted = &j.typedTask.Dependencies.Dependencies[i]
			}
		}
		if accepted == nil || accepted.Source.Verify(mapping.Source) != nil || accepted.Retained.Verify(mapping.Retained) != nil || mapping.Consumer.Digest != mapping.Source.Digest || mapping.Consumer.MediaType != mapping.Source.MediaType || mapping.Consumer.SizeBytes != mapping.Source.SizeBytes {
			t.Fatal("consumer closure differs from its exact accepted manifest")
		}
		retained, err := project.ReadExact(h.ctx, mapping.Retained)
		if err != nil {
			t.Fatal(err)
		}
		assertIterativeLocalBytes(t, h, child, mapping.Consumer, retained.Payload.Data)
	}
	for _, dependency := range j.typedTask.Dependencies.Dependencies {
		if dependency.Name != "proposal" && dependency.Name != "source-result" {
			continue
		}
		expected := j.proposal
		if dependency.Name == "source-result" {
			expected = j.producerResult
			origin := dependency.Origin
			if origin.ItemID != first.ItemID || origin.RoundID != j.snapshots[1].Page.RoundID || origin.ExecutionID != first.SelectedAttempt.ExecutionID || origin.Attempt != 1 || origin.RunID != j.producer {
				t.Fatal("retained result dependency acquired fabricated producer attribution")
			}
		} else if dependency.Origin.ReceiptID != j.receiptID || dependency.Origin.ProposedCheckOrdinal == nil || *dependency.Origin.ProposedCheckOrdinal != 0 {
			t.Fatal("proposal dependency acquired fabricated receipt or ordinal")
		}
		retained, err := project.ReadExact(h.ctx, dependency.Retained)
		if err != nil || dependency.Source.Verify(expected.Exact) != nil || !bytes.Equal(retained.Payload.Data, expected.Data) {
			t.Fatalf("exact dependency original/retained bytes differ: %s %v", dependency.Name, err)
		}
	}
	var rounds, checks, assessments, items int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_rounds WHERE audit_id=$1 AND state<>'proposed'`, auditID).Scan(&rounds); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FILTER(WHERE role='check'),count(*) FILTER(WHERE role='assessment') FROM audit_executions WHERE audit_id=$1`, auditID).Scan(&checks, &assessments); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM audit_items WHERE audit_id=$1`, auditID).Scan(&items); err != nil || rounds != 2 || checks != 2 || assessments != 2 || items != 2 {
		t.Fatalf("joined membership changed: rounds=%d checks=%d assessments=%d items=%d %v", rounds, checks, assessments, items, err)
	}
	var completed bool
	if err := h.pool.QueryRow(h.ctx, `SELECT (audit_input_materialization->>'complete')::boolean FROM workflow_runs WHERE run_id=$1`, child).Scan(&completed); err != nil || !completed {
		t.Fatalf("typed consumer closure never completed: %t %v", completed, err)
	}
}

func assertIterativeLocalBytes(t *testing.T, h *scanProcessHarness, runID string, exact contracts.ExactArtifact, expected []byte) {
	t.Helper()
	if exact.Ref.Revision == nil || exact.Ref.Namespace != "audit-context" {
		t.Fatal("consumer dependency is not an exact local context ref")
	}
	endpoint := fmt.Sprintf("%s/v1/runs/%s/artifacts/%s/%s?revision=%s", h.baseURL, runID, url.PathEscape(exact.Ref.Namespace), url.PathEscape(exact.Ref.Name), url.QueryEscape(*exact.Ref.Revision))
	data, media := download(t, h.client, endpoint)
	if !bytes.Equal(data, expected) || media != exact.MediaType || int64(len(data)) != exact.SizeBytes || auditdomain.DigestBytes(data) != exact.Digest {
		t.Fatal("original-to-retained-to-consumer bytes or exact metadata differ")
	}
}

func confirmIterativeFinding(t *testing.T, h *scanProcessHarness, auditID string) {
	t.Helper()
	var page struct {
		Items []auditProgramFinding `json:"items"`
	}
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/findings?limit=100", &page)
	if len(page.Items) != 1 || page.Items[0].State != "proposed" || page.Items[0].AnalystVerdict != nil {
		t.Fatal("successful typed check or assessment was mistaken for human confirmation")
	}
	finding := page.Items[0]
	var review auditProgramReview
	scanAuditPOST(t, h, "/v1/audits/"+auditID+"/findings/"+finding.FindingID+"/reviews", map[string]any{}, finding.Revision, http.StatusCreated, &review)
	scanAuditPOST(t, h, "/v1/audits/"+auditID+"/reviews/"+review.RequestID+"/decisions", map[string]string{"verdict": "true_positive", "severity": "medium", "rationale": "Owner review of the exact controlled fixture observation."}, review.Revision, http.StatusOK, nil)
	var confirmed auditProgramFinding
	auditProgramGET(t, h.client, h.baseURL+"/v1/audits/"+auditID+"/findings/"+finding.FindingID, &confirmed)
	if confirmed.State != "confirmed" || confirmed.AnalystVerdict == nil || *confirmed.AnalystVerdict != "true_positive" {
		t.Fatal("explicit owner finding confirmation was not retained")
	}
}

func assertIterativeRunHasNoContext(t *testing.T, h *scanProcessHarness, runID string) {
	t.Helper()
	var count int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='run' AND scope_id=$1 AND namespace='audit-context'`, runID).Scan(&count); err != nil || count != 0 {
		t.Fatalf("bypass Run acquired iterative input context: %d %v", count, err)
	}
}

func assertIterativeDefaultBypass(t *testing.T, h *scanProcessHarness, auditID string) {
	t.Helper()
	var rounds, checkpoints, snapshots, items, plans, dependencies, checks, others int
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*),count(admission_checkpoint),count(results_manifest) FROM audit_rounds WHERE audit_id=$1`, auditID).Scan(&rounds, &checkpoints, &snapshots); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*),count(prepared_execution->'accepted_plan'),count(retained_dependencies) FROM audit_items WHERE audit_id=$1 AND item_key='default-seed' AND final_disposition='accepted-result' AND coverage_status='satisfied'`, auditID).Scan(&items, &plans, &dependencies); err != nil {
		t.Fatal(err)
	}
	if err := h.pool.QueryRow(h.ctx, `SELECT count(*) FILTER(WHERE role='check'),count(*) FILTER(WHERE role<>'check') FROM audit_executions WHERE audit_id=$1`, auditID).Scan(&checks, &others); err != nil {
		t.Fatal(err)
	}
	if rounds != 1 || items != 1 || checks != 1 || checkpoints+snapshots+plans+dependencies+others != 0 {
		t.Fatalf("default membership/coverage or bypass changed: rounds=%d items=%d checks=%d optional=%d", rounds, items, checks, checkpoints+snapshots+plans+dependencies+others)
	}
	var runID string
	var empty bool
	if err := h.pool.QueryRow(h.ctx, `SELECT run.run_id,run.audit_input_materialization IS NULL FROM workflow_runs run JOIN audit_executions execution ON execution.run_id=run.run_id WHERE execution.audit_id=$1`, auditID).Scan(&runID, &empty); err != nil || !empty {
		t.Fatalf("default check acquired iterative initialization: %t %v", empty, err)
	}
	assertIterativeRunHasNoContext(t, h, runID)
}
