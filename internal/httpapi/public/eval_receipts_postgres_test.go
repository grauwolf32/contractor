package public

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestEvalPostgresRejectsLegacyReceiptReplayWithoutNewEffects(t *testing.T) {
	for _, kind := range []string{"dataset", "experiment", "command", "submission"} {
		t.Run(kind, func(t *testing.T) {
			h := newEvalAPIHarness(t)
			var path, key, etag, identityField, revisionField string
			var body any
			status := http.StatusCreated
			if kind == "submission" {
				e, manifest, _ := h.registerExternal(t, "workflow")
				path = "/v1/eval-experiments/" + e.ID + "/members/" + manifest.Members[0].MemberID + "/submissions"
				body = evaldomain.Submission{PlanSHA256: *e.PlanSHA256}
				key, identityField, revisionField = "submit", "submissionKey", "experimentRevision"
				status = http.StatusAccepted
			} else {
				draft, data := h.dataset(t, "workflow")
				path, key, body = "/v1/projects/evaluation/eval-datasets", "import", data
				identityField, revisionField = "datasetId", "datasetRevision"
				if kind != "dataset" {
					path, key = "/v1/projects/evaluation/eval-experiments", "receipt-create"
					body = evaldomain.CreateExperiment{Name: "Receipt replay", ControlMode: "server", Draft: &draft}
					identityField, revisionField = "experimentId", "revision"
				}
				if kind == "command" {
					created := h.request(t, "POST", path, body, key, "", http.StatusCreated)
					experiment := apiDecode[evalstore.ExperimentReceipt](t, created)
					path, key, etag = "/v1/eval-experiments/"+experiment.ExperimentID+"/commands", "prepare", `"1"`
					body = evaldomain.Command{Kind: "prepare"}
					identityField, revisionField = "commandId", "experimentRevision"
					status = http.StatusAccepted
				}
			}
			accepted := h.request(t, "POST", path, body, key, etag, status)
			var current []byte
			if err := h.pool.QueryRow(t.Context(), `
SELECT response FROM eval_mutation_receipts
WHERE owner_id='user-1' AND project_id='evaluation' AND operation_key=$1`, key).Scan(&current); err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err := json.Unmarshal(current, &fields); err != nil {
				t.Fatal(err)
			}
			legacy := map[string]any{"id": fields[identityField], "revision": fields[revisionField], "state": fields["state"]}
			if kind == "dataset" {
				legacy["revision"], legacy["state"] = 0, fields["datasetRevision"]
			}
			for _, shape := range []string{"legacy", "mixed"} {
				t.Run(shape, func(t *testing.T) {
					payload := make(map[string]any)
					for k, v := range legacy {
						payload[k] = v
					}
					if shape == "mixed" {
						for k, v := range fields {
							payload[k] = v
						}
					}
					stored, err := json.Marshal(payload)
					if err != nil {
						t.Fatal(err)
					}
					// Seed historical bytes under a retained key without updating an immutable receipt.
					retainedKey := key + "-" + shape
					inserted, err := h.pool.Exec(t.Context(), `
INSERT INTO eval_mutation_receipts(owner_id,project_id,resource_id,operation,operation_key,request_sha256,expected_revision,response)
SELECT owner_id,project_id,resource_id,operation,$2,request_sha256,expected_revision,$3
FROM eval_mutation_receipts
WHERE owner_id='user-1' AND project_id='evaluation' AND operation_key=$1`, key, retainedKey, stored)
					if err != nil || inserted.RowsAffected() != 1 {
						t.Fatalf("seed retained receipt: %v %v", inserted, err)
					}
					before := evalReceiptEffects(t, h)
					for range 2 {
						h.request(t, "POST", path, body, retainedKey, etag, http.StatusInternalServerError)
					}
					if after := evalReceiptEffects(t, h); !bytes.Equal(before, after) {
						t.Fatal("invalid receipt replay changed accepted mutations or stored receipt bytes")
					}
				})
			}
			// Current receipt replay still returns the original response after the rejected reads.
			replayed := h.request(t, "POST", path, body, key, etag, status)
			if !bytes.Equal(accepted.Body.Bytes(), replayed.Body.Bytes()) || replayed.Header().Get("Idempotency-Replayed") != "true" {
				t.Fatal("current receipt replay changed")
			}
		})
	}
}

func evalReceiptEffects(t *testing.T, h *evalAPIHarness) []byte {
	t.Helper()
	// Compare exact receipts and affected authority rows, including revision and clocks.
	var snapshot []byte
	if err := h.pool.QueryRow(t.Context(), `SELECT jsonb_build_object(
    'receipts', (SELECT jsonb_agg(to_jsonb(r) ORDER BY operation_key) FROM eval_mutation_receipts r),
    'datasets', (SELECT jsonb_agg(to_jsonb(d) ORDER BY dataset_id,revision) FROM eval_dataset_revisions d),
    'experiments', (SELECT jsonb_agg(to_jsonb(e) ORDER BY experiment_id) FROM eval_experiments e),
    'commands', (SELECT jsonb_agg(to_jsonb(c) ORDER BY command_id) FROM eval_commands c),
    'submissions', (SELECT jsonb_agg(to_jsonb(s) ORDER BY experiment_id,member_id) FROM eval_submissions s)
)`).Scan(&snapshot); err != nil {
		t.Fatal(err)
	}
	return snapshot
}
