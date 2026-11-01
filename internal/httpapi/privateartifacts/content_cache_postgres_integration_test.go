//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// Forks have independent binding revisions but share an immutable version.
// Purging either consuming Run removes authority without pinning cached bytes.
func TestPostgresPrivateContentCacheIdentityAcrossForksAndPurge(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	createPrivateArtifactRun(t, ctx, pool, "run-b")
	service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
	user, err := service.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	source, err := user.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "source"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("immutable")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	forkA, err := artifactstest.ForkInput(ctx, service, "user-1", source.Ref, "run-a", "source")
	if err != nil {
		t.Fatal(err)
	}
	forkB, err := artifactstest.ForkInput(ctx, service, "user-1", source.Ref, "run-b", "source")
	if err != nil {
		t.Fatal(err)
	}
	read := func(runID string, ref artifacts.ArtifactRef, status int) contracts.ArtifactMetadataResult {
		t.Helper()
		handler, err := NewHandler(Dependencies{
			Registry: &fakeRegistry{grant: testGrant(runID)}, Artifacts: service,
			NewRequestID: func() (string, error) { return "cache-identity", nil },
		})
		if err != nil {
			t.Fatal(err)
		}
		path := "/private/v1/allocations/allocation-1/artifacts/inputs/source/metadata"
		if ref.Revision != nil {
			path += "?revision=" + url.QueryEscape(*ref.Revision)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, trustedRequest(http.MethodGet, path, nil))
		if response.Code != status {
			t.Fatalf("metadata status=%d body=%s", response.Code, response.Body.String())
		}
		if status != http.StatusOK {
			return contracts.ArtifactMetadataResult{}
		}
		var result contracts.ArtifactMetadataResult
		if err := strictjson.Decode(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if err := result.Validate(); err != nil {
			t.Fatal(err)
		}
		if !result.Artifact.Equal(ref) {
			t.Fatalf("metadata ref=%+v want %+v", result.Artifact, ref)
		}
		return result
	}
	first := read("run-a", forkA.TargetRef, http.StatusOK)
	second := read("run-b", forkB.TargetRef, http.StatusOK)
	if *first.Artifact.Revision == *second.Artifact.Revision || first.ContentCacheKey != second.ContentCacheKey {
		t.Fatalf("fork identity did not preserve version with independent refs: %+v %+v", first, second)
	}
	ownerMetadata, err := user.Metadata(ctx, source.Ref)
	if err != nil || ownerMetadata.ContentCacheKey != first.ContentCacheKey {
		t.Fatal("fork changed owner version identity", err)
	}
	publicMetadata, err := json.Marshal(ownerMetadata)
	if err != nil || bytes.Contains(publicMetadata, []byte(first.ContentCacheKey)) || bytes.Contains(publicMetadata, []byte("contentCacheKey")) {
		t.Fatal("internal version identity leaked into ordinary metadata")
	}
	independent, err := user.Write(ctx, artifacts.ArtifactRef{Namespace: "sources", Name: "independent"},
		artifacts.Payload{MediaType: "application/zip", Data: []byte("immutable")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	independentMetadata, err := user.Metadata(ctx, independent.Ref)
	if err != nil || independentMetadata.ContentCacheKey == first.ContentCacheKey {
		t.Fatal("independent versions aliased", err)
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		t.Fatal(err)
	}
	if err := purger.PurgeRun(ctx, "run-a"); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	read("run-a", forkA.TargetRef, http.StatusNotFound)
	surviving := read("run-b", forkB.TargetRef, http.StatusOK)
	if surviving.ContentCacheKey != first.ContentCacheKey {
		t.Fatal("purge changed surviving version identity")
	}
}
