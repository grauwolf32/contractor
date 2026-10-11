package public

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestRunDetailReadsEveryArtifactMetadataPage(t *testing.T) {
	fixture := newHandlerFixture(t)
	repository := fixture.repository
	fixture.runs.runs["run-1"] = queryRun("run-1", "user-1", runstore.RunSucceeded, time.Now().UTC())
	scope, err := artifacts.RunScope("run-1")
	if err != nil {
		t.Fatal(err)
	}
	const outputs = 2*pagelimit.MaxRows + 1
	want := make(map[string]string, outputs)
	for index := range outputs {
		name := fmt.Sprintf("slot-%03d", index)
		written, writeErr := repository.Write(t.Context(), scope,
			artifacts.ArtifactRef{Namespace: "outputs", Name: name},
			artifacts.Payload{MediaType: "text/plain", Data: []byte(name)}, nil)
		if writeErr != nil {
			t.Fatal(writeErr)
		}
		want[name] = *written.Ref.Revision
	}
	if _, err := repository.Write(t.Context(), scope,
		artifacts.ArtifactRef{Namespace: "inputs", Name: "source"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("in")}, nil); err != nil {
		t.Fatal(err)
	}
	repository.queryReads = 0

	response := serveQuery(t, fixture.handler, "/v1/runs/run-1")
	if response.Code != http.StatusOK {
		t.Fatalf("Run detail = %d %s", response.Code, response.Body.String())
	}
	var detail runStatusResponse
	decodeQueryResponse(t, response, &detail)
	if len(detail.Outputs) != len(want) || len(detail.Inputs) != 1 || detail.Inputs["source"].Namespace != "inputs" {
		t.Fatalf("Run detail returned %d outputs and inputs %+v, want %d outputs and one input",
			len(detail.Outputs), detail.Inputs, len(want))
	}
	for name, revision := range want {
		ref, ok := detail.Outputs[name]
		if !ok || ref.Namespace != "outputs" || ref.Name != name || ref.Revision == nil || *ref.Revision != revision {
			t.Fatalf("ref %s = %+v, want revision %s", name, ref, revision)
		}
	}
	// Three keyset pages of outputs and one page of inputs, never one lookup
	// per binding.
	if repository.queryReads != 4 {
		t.Fatalf("metadata queries = %d, want 4 keyset pages", repository.queryReads)
	}
}
