package public

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestRunArtifactsByNamespaceReadsEveryMetadataPage(t *testing.T) {
	repository := newFakeArtifactRepository()
	scope, err := artifacts.RunScope("run-1")
	if err != nil {
		t.Fatal(err)
	}
	const outputs = 2*maxPageLimit + 1
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
	h := &handler{dependencies: Dependencies{Artifacts: artifacts.NewService(repository)}}
	request := httptest.NewRequest(http.MethodGet, "/", nil)

	got, err := h.runArtifactsByNamespace(request, "run-1", "outputs")
	if err != nil {
		t.Fatalf("runArtifactsByNamespace: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("runArtifactsByNamespace returned %d refs, want %d", len(got), len(want))
	}
	for name, revision := range want {
		ref, ok := got[name]
		if !ok || ref.Namespace != "outputs" || ref.Name != name || ref.Revision == nil || *ref.Revision != revision {
			t.Fatalf("ref %s = %+v, want revision %s", name, ref, revision)
		}
	}
	if repository.queryReads != 3 {
		t.Fatalf("metadata queries = %d, want 3 keyset pages", repository.queryReads)
	}
}
