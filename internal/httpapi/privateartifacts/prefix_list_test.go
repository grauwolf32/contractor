package privateartifacts

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPrivateArtifactPrefixListIsBoundedAndValidated(t *testing.T) {
	repository := &prefixRepository{memoryRepository: newMemoryRepository()}
	for i := 0; i < 7000; i++ {
		repository.seed("run-a", "builder", fmt.Sprintf("artifact_%06d_%s", i, strings.Repeat("x", 110)), "r", []byte("ordinary"))
	}
	repository.seed("run-a", "builder", "memory.one", "r", []byte("note"))
	repository.seed("run-a", "foreign", "memory.foreign", "r", []byte("foreign"))
	repository.seed("run-b", "builder", "memory.foreign", "r", []byte("foreign"))
	handler, err := NewHandler(Dependencies{Registry: &fakeRegistry{grant: testGrant("run-a")}, Artifacts: artifacts.NewService(repository)})
	if err != nil {
		t.Fatal(err)
	}
	target := "/private/v1/allocations/allocation-1/artifacts"
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, trustedRequest(http.MethodGet, target+"?namespace=builder&namePrefix=memory.&limit=129", nil))
	var result contracts.ArtifactListResult
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || response.Code != 200 || len(result.Artifacts) != 1 || result.Artifacts[0].Name != "memory.one" || repository.prefixCalls != 1 {
		t.Fatalf("filtered response: %d %s; %v", response.Code, response.Body.String(), err)
	}
	if len(response.Body.Bytes()) > 1024 {
		t.Fatal("unrelated bindings reached the response")
	}
	for _, query := range []string{
		"namePrefix=memory.&limit=129", "namespace=builder&limit=129", "namespace=builder&namePrefix=memory.",
		"namespace=builder&namePrefix=&limit=129", "namespace=builder&namePrefix=memory%25&limit=129",
		"namespace=builder&namePrefix=memory.&limit=0", "namespace=builder&namePrefix=memory.&limit=257",
		"namespace=builder&namePrefix=memory.&limit=bad", "namespace=builder&namePrefix=memory.&limit=1&limit=2",
		"namespace=builder&namePrefix=memory.&limit=129&runId=run-b",
	} {
		t.Run(query, func(t *testing.T) {
			rejected := httptest.NewRecorder()
			handler.ServeHTTP(rejected, trustedRequest(http.MethodGet, target+"?"+query, nil))
			if rejected.Code != 400 {
				t.Fatalf("invalid query response: %d %s", rejected.Code, rejected.Body.String())
			}
		})
	}
	if repository.prefixCalls != 1 {
		t.Fatal("invalid query reached prefix repository")
	}
	legacy := httptest.NewRecorder()
	handler.ServeHTTP(legacy, trustedRequest(http.MethodGet, target+"?namespace=builder", nil))
	if legacy.Code != 200 || len(legacy.Body.Bytes()) <= 1<<20 {
		t.Fatal("legacy list changed")
	}
}

type prefixRepository struct {
	*memoryRepository
	prefixCalls int
}

func (r *prefixRepository) ListPrefix(_ context.Context, scope artifacts.Scope, namespace, prefix string, limit int) ([]artifacts.ArtifactRef, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prefixCalls++
	keyPrefix := string(scope.Kind()) + "\x00" + scope.ID() + "\x00" + namespace + "\x00" + prefix
	refs := make([]artifacts.ArtifactRef, 0)
	for key := range r.bindings {
		if strings.HasPrefix(key, keyPrefix) {
			parts := strings.Split(key, "\x00")
			refs = append(refs, artifacts.ArtifactRef{Namespace: parts[2], Name: parts[3]})
		}
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].Name < refs[j].Name })
	if len(refs) > limit {
		refs = refs[:limit]
	}
	return refs, nil
}
