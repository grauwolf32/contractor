package evalstore

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestReceiptsRequireCurrentTypedFields(t *testing.T) {
	t.Run("dataset", func(t *testing.T) {
		testTypedReceipt(t, Receipt.Dataset,
			`{"datasetId":"dataset-1","datasetRevision":"r1"}`,
			`{"id":"dataset-1","revision":0,"state":"r1"}`,
			DatasetReceipt{DatasetID: "dataset-1", DatasetRevision: "r1"},
			"datasetId", "datasetRevision")
	})
	t.Run("experiment", func(t *testing.T) {
		testTypedReceipt(t, Receipt.Experiment,
			`{"experimentId":"experiment-1","revision":3,"state":"draft"}`,
			`{"id":"experiment-1","revision":3,"state":"draft"}`,
			ExperimentReceipt{ExperimentID: "experiment-1", Revision: 3, State: "draft"},
			"experimentId", "revision")
	})
	t.Run("command", func(t *testing.T) {
		testTypedReceipt(t, Receipt.Command,
			`{"commandId":"command-1","experimentRevision":4,"state":"accepted"}`,
			`{"id":"command-1","revision":4,"state":"accepted"}`,
			AcceptedCommandReceipt{CommandID: "command-1", ExperimentRevision: 4, State: "accepted"},
			"commandId", "experimentRevision")
	})
	t.Run("submission", func(t *testing.T) {
		testTypedReceipt(t, Receipt.Submission,
			`{"submissionKey":"submission-1","experimentRevision":5,"state":"intent"}`,
			`{"id":"submission-1","revision":5,"state":"intent"}`,
			AcceptedSubmissionReceipt{SubmissionKey: "submission-1", ExperimentRevision: 5, State: "intent"},
			"submissionKey", "experimentRevision")
	})
}

func testTypedReceipt[T comparable](t *testing.T, decode func(Receipt) (T, error), current, legacy string, want T, identity, revision string) {
	t.Helper()
	for _, replayed := range []bool{false, true} {
		r := Receipt{Response: json.RawMessage(current), Replayed: replayed}
		got, err := decode(r)
		if err != nil || got != want || string(r.Response) != current || r.Replayed != replayed {
			t.Fatalf("current receipt (replayed=%t): %+v %v", replayed, got, err)
		}
	}
	mutate := func(fn func(map[string]any)) string {
		var body map[string]any
		if err := json.Unmarshal([]byte(current), &body); err != nil {
			t.Fatal(err)
		}
		fn(body)
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	invalid := map[string]string{
		"legacy":                legacy,
		"mixed":                 mutate(func(body map[string]any) { body["id"] = "old-identity" }),
		"empty legacy identity": mutate(func(body map[string]any) { body["id"] = "" }),
		"missing identity":      mutate(func(body map[string]any) { delete(body, identity) }),
		"null identity":         mutate(func(body map[string]any) { body[identity] = nil }),
		"empty identity":        mutate(func(body map[string]any) { body[identity] = "" }),
		"wrong identity type":   mutate(func(body map[string]any) { body[identity] = 42 }),
		"missing revision":      mutate(func(body map[string]any) { delete(body, revision) }),
		"null revision":         mutate(func(body map[string]any) { body[revision] = nil }),
		"unknown field":         mutate(func(body map[string]any) { body["unknown"] = true }),
		"empty":                 "", "null": "null", "array": "[]", "empty object": "{}",
		"trailing object": current + `{}`, "trailing null": current + `null`,
		"malformed": current[:len(current)-1],
	}
	if revision != "datasetRevision" {
		invalid["zero revision"] = mutate(func(body map[string]any) { body[revision] = 0 })
		invalid["negative revision"] = mutate(func(body map[string]any) { body[revision] = -1 })
	}
	for name, raw := range invalid {
		t.Run(name, func(t *testing.T) {
			r := Receipt{Response: json.RawMessage(raw), Replayed: true}
			original := bytes.Clone(r.Response)
			var zero T
			if got, err := decode(r); err == nil || got != zero {
				t.Fatalf("invalid receipt returned (%+v, %v), want empty result and error", got, err)
			}
			if !bytes.Equal(r.Response, original) || !r.Replayed {
				t.Fatal("invalid receipt was rewritten during replay")
			}
		})
	}
}

func TestReceiptTypesCannotBeInterchanged(t *testing.T) {
	bodies := []json.RawMessage{
		json.RawMessage(`{"datasetId":"dataset-1","datasetRevision":"r1"}`),
		json.RawMessage(`{"experimentId":"experiment-1","revision":3,"state":"draft"}`),
		json.RawMessage(`{"commandId":"command-1","experimentRevision":4,"state":"accepted"}`),
		json.RawMessage(`{"submissionKey":"submission-1","experimentRevision":5,"state":"intent"}`),
	}
	for index, body := range bodies {
		r := Receipt{Response: body}
		_, datasetErr := r.Dataset()
		_, experimentErr := r.Experiment()
		_, commandErr := r.Command()
		_, submissionErr := r.Submission()
		for decoder, err := range []error{datasetErr, experimentErr, commandErr, submissionErr} {
			if (err == nil) != (index == decoder) {
				t.Fatalf("receipt %d through decoder %d: %v", index, decoder, err)
			}
		}
	}
}
