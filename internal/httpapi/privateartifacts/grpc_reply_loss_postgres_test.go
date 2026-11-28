//go:build integration

package privateartifacts

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/grauwolf32/contractor/api/v1alpha1/runtimepb"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"
)

// The real Go/Python/mTLS/PG path commits once, loses its gRPC terminal reply,
// rejects the same consumed CAS and lets the existing Runtime owner confirm
// exact bytes by reading current. There is no transport replay or new receipt.
func TestPostgresNativeArtifactReplyLossReconcilesOriginalCAS(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	pool := isolatedPrivateArtifactPool(t, ctx)
	createPrivateArtifactRun(t, ctx, pool, "run-a")
	storage := artifactpgtest.NewStorage()
	var writes atomic.Int32
	fixture := newCrossLanguageHarness(t, storage.Repository(pool), func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == pb.Artifact_Write_FullMethodName && writes.Add(1) == 1 {
				held := &heldNativeReceipt{ResponseRecorder: httptest.NewRecorder(), outer: w}
				next.ServeHTTP(held, r)
				if held.Header().Get("Grpc-Status") != "0" {
					t.Errorf("lost-reply injection did not follow a successful commit: %v", held.Header())
				}
				panic(http.ErrAbortHandler)
			}
			next.ServeHTTP(w, r)
		})
	})
	result := fixture.probe("--mode", "reconcile")
	if result["revision"] == "" || result["size"] != float64(256*2049) || writes.Load() != 2 {
		t.Fatalf("native CAS recovery=%v writes=%d", result, writes.Load())
	}
	run, err := artifacts.NewService(storage.Repository(pool)).Run("run-a")
	if err != nil {
		t.Fatal(err)
	}
	current, err := run.Read(ctx, artifacts.ArtifactRef{Namespace: "builder", Name: "reconciled"})
	if err != nil || current.Ref.Revision == nil || *current.Ref.Revision != result["revision"] || len(current.Payload.Data) != 256*2049 {
		t.Fatalf("committed exact recovery=%+v %v", current.Ref, err)
	}
	var revisions int
	if err = pool.QueryRow(ctx, `SELECT count(*) FROM artifact_versions`).Scan(&revisions); err != nil || revisions != 1 {
		t.Fatalf("lost reply created %d revisions: %v", revisions, err)
	}
}

type heldNativeReceipt struct {
	*httptest.ResponseRecorder
	outer http.ResponseWriter
}

func (w *heldNativeReceipt) Unwrap() http.ResponseWriter { return w.outer }
