//go:build integration

package privateartifacts

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
)

// The read_artifact tool of the Python Runtime refuses an archive from the
// metadata read alone: the Server answers the metadata route and never the
// raw read of the archive (ART-R116, ART-R117).
func TestPostgresBinaryReadIsRefusedBeforeDownload(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	var mu sync.Mutex
	var reads []string
	record := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
			if request.Method == http.MethodGet || request.URL.Path == pb.Allocation_ReadArtifactMetadata_FullMethodName {
				mu.Lock()
				reads = append(reads, request.URL.Path)
				mu.Unlock()
			}
			next.ServeHTTP(w, request)
		})
	}
	harness := newCrossLanguageHarness(t, artifactpgtest.NewStorage().Repository(pool), record)
	result := harness.probe("--mode", "binary-read")
	if message, _ := result["error"].(string); !strings.Contains(message, "binary (application/zip)") {
		t.Fatalf("binary read through read_artifact = %+v", result)
	}
	mu.Lock()
	defer mu.Unlock()
	const archive = pb.Allocation_ReadArtifactMetadata_FullMethodName
	if len(reads) != 1 || reads[0] != archive {
		t.Fatalf("binary read requested %v, want only the metadata of %s", reads, archive)
	}
}
