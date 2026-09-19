package evalstore

import (
	"encoding/json"
	"testing"
)

func TestLegacyReceiptsRemainReadableWithoutRewritingStoredBytes(t *testing.T) {
	dataset := Receipt{Response: json.RawMessage(`{"id":"dataset-1","revision":0,"state":"r1"}`), Replayed: true}
	got, err := dataset.Dataset()
	if err != nil || got.DatasetID != "dataset-1" || got.DatasetRevision != "r1" {
		t.Fatalf("dataset receipt: %+v %v", got, err)
	}
	experiment := Receipt{Response: json.RawMessage(`{"id":"experiment-1","revision":3,"state":"draft"}`), Replayed: true}
	exp, err := experiment.Experiment()
	if err != nil || exp.ExperimentID != "experiment-1" || exp.Revision != 3 || exp.State != "draft" {
		t.Fatalf("experiment receipt: %+v %v", exp, err)
	}
	command := Receipt{Response: json.RawMessage(`{"id":"command-1","revision":4,"state":"accepted"}`), Replayed: true}
	cmd, err := command.Command()
	if err != nil || cmd.CommandID != "command-1" || cmd.ExperimentRevision != 4 || cmd.State != "accepted" {
		t.Fatalf("command receipt: %+v %v", cmd, err)
	}
	submission := Receipt{Response: json.RawMessage(`{"id":"submission-1","revision":5,"state":"intent"}`), Replayed: true}
	sub, err := submission.Submission()
	if err != nil || sub.SubmissionKey != "submission-1" || sub.ExperimentRevision != 5 || sub.State != "intent" {
		t.Fatalf("submission receipt: %+v %v", sub, err)
	}
	if string(dataset.Response) != `{"id":"dataset-1","revision":0,"state":"r1"}` {
		t.Fatal("receipt mutated during decoding")
	}
	if _, err = (Receipt{Response: json.RawMessage(`{"datasetId":"dataset-1","datasetRevision":"r1"}`)}).Experiment(); err == nil {
		t.Fatal("dataset decoded as experiment")
	}
}
