package evalservice

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/evaldomain"
)

func TestEvalNativeStructuralChecksAndPinnedFailures(t *testing.T) {
	pool := serviceTestPool(t)
	scope, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).User("owner")
	if err != nil {
		t.Fatal(err)
	}
	write := func(name, media, body string) evaldomain.Artifact {
		written, err := scope.Write(t.Context(), contracts.ArtifactRef{Namespace: "fixtures", Name: name}, artifacts.Payload{MediaType: media, Data: []byte(body)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return evaldomain.Artifact{Scope: "user", ScopeID: "owner", Namespace: written.Ref.Namespace, Name: written.Ref.Name, Revision: *written.Ref.Revision, SHA256: evaldomain.Digest([]byte(body)), MediaType: media, SizeBytes: int64(len(body))}
	}
	valid := write("valid", "application/json", `{"observed":true}`)
	invalid := write("invalid", "application/json", `not json`)
	result := evaldomain.ResultInput{Outputs: map[string]evaldomain.Artifact{"valid": valid, "invalid": invalid}, Evidence: []evaldomain.Evidence{{ID: "valid-proof", Artifact: valid}, {ID: "invalid-proof", Artifact: invalid}}}
	checks := []evaldomain.Check{
		{ID: "shape", Evaluator: "json-schema@1", Required: true, ImplementationSHA256: NativePolicySHA256(), Parameters: map[string]string{"output": "valid", "schema": "json@1"}},
		{ID: "invalid-shape", Evaluator: "json-schema@1", Required: true, ImplementationSHA256: NativePolicySHA256(), Parameters: map[string]string{"output": "invalid", "schema": "json@1"}},
		{ID: "media", Evaluator: "media-type@1", Required: true, ImplementationSHA256: NativePolicySHA256(), Parameters: map[string]string{"output": "valid", "mediaType": "text/plain"}},
		{ID: "missing", Evaluator: "required-artifact@1", Required: true, ImplementationSHA256: NativePolicySHA256(), Parameters: map[string]string{"output": "missing"}},
		{ID: "unavailable", Evaluator: "json-schema@1", Required: true, ImplementationSHA256: evaldomain.Digest([]byte("old implementation")), Parameters: map[string]string{"output": "valid", "schema": "json@1"}},
	}
	results, err := runNativeChecks(t.Context(), pool, checks, nil, result)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"pass", "fail", "fail", "incomplete", "error"}
	for i, expected := range want {
		if results[i].Status != expected {
			t.Fatalf("%s: got %s want %s", checks[i].ID, results[i].Status, expected)
		}
	}
	if len(results[0].EvidenceRefs) != 1 || results[0].EvidenceRefs[0] != "valid-proof" {
		t.Fatal("check lost exact evidence")
	}
	if selectedCollectionComplete(evaldomain.ResultInput{Collection: evaldomain.Collection{Status: "complete"}}, &evaldomain.ResultInput{Collection: evaldomain.Collection{Status: "complete"}}, map[string]evaldomain.Output{"required": {Required: true}}) {
		t.Fatal("missing required result output was collection-complete")
	}
	// Database failure must be retried, not persisted as an unavailable artifact.
	if _, err = pool.Exec(t.Context(), `ALTER TABLE artifact_binding_revisions RENAME TO unavailable_revisions`); err != nil {
		t.Fatal(err)
	}
	if _, err = runNativeChecks(t.Context(), pool, checks[:1], nil, result); err == nil {
		t.Fatal("database failure became a check result")
	}
}
