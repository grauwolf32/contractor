package public

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestSharedEvalPostgresHumanReviewParametersRejectedBeforeEffects(t *testing.T) {
	for _, kind := range []string{"native-create", "draft-update", "external-create"} {
		t.Run(kind, func(t *testing.T) {
			h := newEvalAPIHarness(t)
			method, path, etag, branch, accepted := "POST", "/v1/projects/evaluation/eval-experiments", "", "draft", 201
			var supported any
			if kind == "external-create" {
				create, _, _ := h.externalCreateRequest(t, "workflow")
				supported, branch = create, "registration"
			} else {
				draft, _ := h.dataset(t, "workflow")
				create := evaldomain.CreateExperiment{Name: "Parameter contract", ControlMode: "server", Draft: &draft}
				supported = create
				if kind == "draft-update" {
					receipt := apiDecode[evalstore.ExperimentReceipt](t, h.request(t, "POST", path, create, "base-experiment", "", 201))
					method, path, etag, accepted = "PATCH", "/v1/eval-experiments/"+receipt.ExperimentID, `"1"`, 200
					supported = evaldomain.DraftUpdate{Name: "Updated criterion", Draft: draft}
				}
			}
			raw, err := json.Marshal(supported)
			if err != nil {
				t.Fatal(err)
			}
			for key, value := range map[string]string{"output": "report", "mediaType": "text/plain", "schema": "json@1"} {
				var unsupported map[string]any
				if err := json.Unmarshal(raw, &unsupported); err != nil {
					t.Fatal(err)
				}
				check := unsupported[branch].(map[string]any)["checks"].([]any)[0].(map[string]any)
				if check["evaluator"] != "human-review@1" {
					t.Fatal("regression requires the registered human-review evaluator")
				}
				check["parameters"] = map[string]string{key: value}
				before := evalReceiptEffects(t, h)
				answer := apiDecode[errorResponse](t, h.request(t, method, path, unsupported, "human-parameters-"+key, etag, 422))
				if answer.Code != "eval_invalid" || answer.Retryable {
					t.Fatalf("ignored human parameters accepted: %+v", answer)
				}
				if after := evalReceiptEffects(t, h); !bytes.Equal(before, after) {
					t.Fatal("rejected human parameters changed authority or recorded a receipt")
				}
			}
			// A corrected request can reuse a rejected key; there was no receipt.
			h.request(t, method, path, supported, "human-parameters-output", etag, accepted)
		})
	}
}

func TestSharedEvalPostgresRetainedHumanParametersRejectReviewAndAssessment(t *testing.T) {
	h, e, members := completedEvalFixture(t)
	member := members[0]
	base := "/v1/eval-experiments/" + e.ID
	memberPath := base + "/members/" + member.Member.ID
	review := ownerEvalReview(t, h, memberPath+"/review")
	check := review.Policy[0]
	assessment := evaldomain.AssessmentInput{SchemaVersion: evaldomain.AssessmentSchemaVersion, Source: evaldomain.AssessmentSource{Kind: "human"}, ResultSHA256: review.ResultSHA256, PreviousAssessmentSHA256: member.AssessmentSHA256, Checks: []evaldomain.CheckResult{{ID: check.ID, Evaluator: check.Evaluator, ImplementationSHA256: check.ImplementationSHA256, Status: "pass", Reason: "Owner decision", EvidenceRefs: []string{}}}}
	var original []byte
	if err := h.pool.QueryRow(t.Context(), `SELECT plan_setup FROM eval_experiments WHERE experiment_id=$1`, e.ID).Scan(&original); err != nil {
		t.Fatal(err)
	}
	var setup map[string]any
	if err := json.Unmarshal(original, &setup); err != nil {
		t.Fatal(err)
	}
	policy := setup["checks"].([]any)[0].(map[string]any)
	if policy["evaluator"] != "human-review@1" {
		t.Fatal("regression requires human review")
	}
	policy["parameters"] = map[string]string{"output": "report"}
	retained, err := json.Marshal(setup)
	if err != nil {
		t.Fatal(err)
	}
	// Only test DDL can seed an invalid retained projection. The transaction
	// restores the immutability trigger on every outcome; no production repair
	// or reader is permitted to rewrite these historical bytes.
	tx, err := h.pool.Begin(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback(t.Context()) }()
	if _, err = tx.Exec(t.Context(), `ALTER TABLE eval_experiments DISABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments DISABLE TRIGGER eval_experiments_protect`); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(t.Context(), `UPDATE eval_experiments SET plan_setup=$2 WHERE experiment_id=$1`, e.ID, retained); err != nil {
		t.Fatal(err)
	}
	if _, err = tx.Exec(t.Context(), `ALTER TABLE eval_experiments ENABLE TRIGGER eval_plans_immutable; ALTER TABLE eval_experiments ENABLE TRIGGER eval_experiments_protect`); err != nil {
		t.Fatal(err)
	}
	if err = tx.Commit(t.Context()); err != nil {
		t.Fatal(err)
	}
	snapshot := func() []byte {
		t.Helper()
		var records []byte
		if err := h.pool.QueryRow(t.Context(), `SELECT jsonb_build_object(
    'records', (SELECT jsonb_agg(to_jsonb(r) ORDER BY experiment_id,member_id,kind,record_sha256) FROM eval_records r),
    'selections', (SELECT jsonb_agg(to_jsonb(s) ORDER BY experiment_id,member_id) FROM eval_members s WHERE selected_result_sha256 IS NOT NULL),
    'plans', (SELECT jsonb_agg(to_jsonb(p) ORDER BY experiment_id) FROM eval_experiments p)
)`).Scan(&records); err != nil {
			t.Fatal(err)
		}
		return append(evalReceiptEffects(t, h), records...)
	}
	before := snapshot()
	if _, err = h.service.Review(t.Context(), "user-1", e.ID, member.Member.ID, ""); !evaldomain.IsCode(err, "eval_invalid") {
		t.Fatalf("direct review ignored retained parameters: %v", err)
	}
	for _, path := range []string{base, base + "/members", memberPath + "/review"} {
		answer := apiDecode[errorResponse](t, h.request(t, "GET", path, nil, "", "", 422))
		if answer.Code != "eval_invalid" || answer.Retryable {
			t.Fatalf("frozen policy read accepted unsupported parameters: %+v", answer)
		}
	}
	for name, body := range map[string]any{
		"human":  assessment,
		"native": evaldomain.CheckRequest{SchemaVersion: "contractor.eval-check-request/v1", Source: evaldomain.AssessmentSource{Kind: "native"}, ResultSHA256: review.ResultSHA256, CheckIDs: []string{check.ID}},
	} {
		answer := apiDecode[errorResponse](t, h.request(t, "POST", memberPath+"/assessments", body, "retained-"+name, "", 422))
		if answer.Code != "eval_invalid" || answer.Retryable {
			t.Fatalf("assessment ignored retained parameters: %+v", answer)
		}
	}
	if after := snapshot(); !bytes.Equal(before, after) {
		t.Fatal("rejected frozen policy changed a record, selection, receipt or retained plan")
	}
}
