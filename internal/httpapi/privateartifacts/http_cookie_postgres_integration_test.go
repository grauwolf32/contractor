//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

// The Server treats the Runtime-owned cookie format as ordinary opaque bytes.
// Both production blob backends exercise mTLS, lost acknowledgements, exact CAS,
// fresh toolset restoration, generic reads, allocation fences and Run isolation.
func TestPostgresHTTPCookieContinuation(t *testing.T) {
	for _, backend := range []artifacts.BlobBackend{artifacts.BlobPostgres, artifacts.BlobFilesystem} {
		for _, proxy := range []bool{false, true} {
			name := string(backend) + "/direct"
			if proxy {
				name = string(backend) + "/proxy"
			}
			t.Run(name, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
				defer cancel()
				pool := isolatedPrivateArtifactPool(t, ctx)
				createPrivateArtifactRun(t, ctx, pool, "run-a")
				createPrivateArtifactRun(t, ctx, pool, "run-b")
				_, store, err := artifacts.OpenBlobStore(ctx, backend, t.TempDir(), 0)
				if err != nil {
					t.Fatal(err)
				}
				if closer, ok := store.(io.Closer); ok {
					t.Cleanup(func() { _ = closer.Close() })
				}
				runtime := artifacts.NewBlobRuntime(store, nil)
				var lost atomic.Bool
				wrap := func(next http.Handler) http.Handler {
					return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						r = r.WithContext(artifacts.WithBlobRuntime(r.Context(), runtime))
						if r.Method == http.MethodPut && strings.HasSuffix(r.URL.Path, "/http-cookie-session") && lost.CompareAndSwap(false, true) {
							recorder := httptest.NewRecorder()
							next.ServeHTTP(recorder, r)
							if recorder.Code != http.StatusCreated {
								t.Errorf("first checkpoint status = %d", recorder.Code)
							}
							// The commit succeeded; its acknowledgement was lost.
							w.Header().Set("Content-Type", "application/json")
							w.WriteHeader(http.StatusInternalServerError)
							_, _ = w.Write([]byte(`{"error":{"code":"internal_error","message":"storage acknowledgement lost","retryable":true,"requestId":"cookie-fault"}}`))
							return
						}
						next.ServeHTTP(w, r)
					})
				}
				harness := newCrossLanguageHarness(t, artifacts.NewPostgresRepository(pool), wrap)
				probe := func(action string) map[string]any {
					args := []string{"--mode", "cookies", "--cookie-action", action}
					if proxy {
						args = append(args, "--cookie-proxy")
					}
					return harness.probe(args...)
				}
				result := probe("login")
				if result["status"] != float64(200) || result["calls"] != float64(2) || result["count"] != float64(1) {
					t.Fatalf("cookie continuation = %+v", result)
				}
				var revisions int
				if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_binding_revisions").Scan(&revisions); err != nil || revisions != 1 {
					t.Fatalf("unchanged/lost/stale writes created %d revisions: %v", revisions, err)
				}
				var actual string
				if err := pool.QueryRow(ctx, "SELECT backend FROM artifact_blobs").Scan(&actual); err != nil || actual != string(backend) {
					t.Fatalf("cookie backend = %q, %v", actual, err)
				}
				harness.registry.mu.Lock()
				harness.registry.grant.WriteFenced = true
				harness.registry.mu.Unlock()
				if result := probe("fenced"); result["code"] != "http_cookie_write_denied" || result["retryable"] != false || result["calls"] != float64(0) {
					t.Fatalf("fenced cookie checkpoint = %+v", result)
				}
				harness.registry.mu.Lock()
				harness.registry.grant.WriteFenced = false
				harness.registry.grant.RunID = "run-b"
				harness.registry.mu.Unlock()
				if result := probe("read"); result["count"] != float64(0) {
					t.Fatalf("cross-Run cookie leak = %+v", result)
				}
				harness.registry.mu.Lock()
				harness.registry.grant.RunID = "run-a"
				harness.registry.mu.Unlock()
				if result := probe("delete"); result["count"] != float64(0) {
					t.Fatalf("cookie deletion = %+v", result)
				}
				if result := probe("read"); result["count"] != float64(0) {
					t.Fatalf("deleted cookie restored = %+v", result)
				}
				if result := probe("clear"); result["count"] != float64(0) {
					t.Fatalf("cookie clear = %+v", result)
				}
				if result := probe("faults"); result["storage"] != "http_cookie_storage_unavailable" || result["joined"] != true {
					t.Fatalf("cookie storage/cancellation fault = %+v", result)
				}
				tx, err := pool.Begin(ctx)
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = tx.Rollback(context.Background()) }()
				purger, err := artifacts.NewPostgresPurger(tx)
				if err != nil {
					t.Fatal(err)
				}
				if err := purger.PurgeRun(artifacts.WithBlobRuntime(ctx, runtime), "run-a"); err != nil {
					t.Fatal(err)
				}
				if err := tx.Commit(ctx); err != nil {
					t.Fatal(err)
				}
				if err := pool.QueryRow(ctx, "SELECT count(*) FROM artifact_binding_revisions").Scan(&revisions); err != nil || revisions != 0 {
					t.Fatalf("Run purge retained %d cookie revisions: %v", revisions, err)
				}

			})
		}
	}
}

// The same allocation gate used at Stage termination joins a cookie write
// across the real PostgreSQL commit on both physical storage backends.
func TestPostgresCookieTerminalWriteBarrier(t *testing.T) {
	data := []byte(`{"schema_version":1,"target_origin":null,"cookies":[]}`)
	for _, backend := range []artifacts.BlobBackend{artifacts.BlobPostgres, artifacts.BlobFilesystem} {
		t.Run(string(backend), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
			defer cancel()
			pool := isolatedPrivateArtifactPool(t, ctx)
			createPrivateArtifactRun(t, ctx, pool, "run-production-race")
			_, blobs, err := artifacts.OpenBlobStore(ctx, backend, t.TempDir(), 0)
			if err != nil {
				t.Fatal(err)
			}
			if closer, ok := blobs.(io.Closer); ok {
				t.Cleanup(func() { _ = closer.Close() })
			}
			runtime := artifacts.NewBlobRuntime(blobs, nil)
			repository := &cookieBarrierRepository{
				Repository: artifacts.NewPostgresRepository(pool),
				entered:    make(chan struct{}), release: make(chan struct{}),
			}
			handler, registry, allocation, _, _ := newProductionGrantHandler(t, repository)
			write := func() *httptest.ResponseRecorder {
				request := trustedRequest(http.MethodPut,
					"/private/v1/allocations/"+allocation+"/artifacts/builder/http-cookie-session",
					bytes.NewReader(data))
				request.Header.Set("Content-Type", "application/vnd.contractor.http-cookie-session+json")
				request.Header.Set("If-None-Match", "*")
				request = request.WithContext(artifacts.WithBlobRuntime(request.Context(), runtime))
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				return response
			}
			written := make(chan *httptest.ResponseRecorder, 1)
			go func() { written <- write() }()
			select {
			case <-repository.entered:
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			fenced := make(chan error, 1)
			go func() {
				err := registry.SetWriteFence(allocation)
				if err == nil && !repository.committed.Load() {
					err = errors.New("write fence overtook cookie commit")
				}
				fenced <- err
			}()
			close(repository.release)
			select {
			case response := <-written:
				if response.Code != http.StatusCreated {
					t.Fatalf("cookie write status = %d", response.Code)
				}
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			select {
			case err := <-fenced:
				if err != nil {
					t.Fatal(err)
				}
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			response := write()
			if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), "allocation_write_fenced") {
				t.Fatalf("old cookie writer status = %d", response.Code)
			}
			run, err := artifacts.NewService(repository.Repository).Run("run-production-race")
			if err != nil {
				t.Fatal(err)
			}
			value, err := run.Read(artifacts.WithBlobRuntime(ctx, runtime), artifacts.ArtifactRef{Namespace: "builder", Name: "http-cookie-session"})
			if err != nil || !bytes.Equal(value.Payload.Data, data) {
				t.Fatal("post-barrier cookie bytes changed", err)
			}
		})
	}
}

type cookieBarrierRepository struct {
	artifacts.Repository
	entered   chan struct{}
	release   chan struct{}
	committed atomic.Bool
}

func (r *cookieBarrierRepository) Write(ctx context.Context, scope artifacts.Scope, target artifacts.ArtifactRef, payload artifacts.Payload, expected *string) (artifacts.WriteResult, error) {
	close(r.entered)
	select {
	case <-r.release:
	case <-ctx.Done():
		return artifacts.WriteResult{}, ctx.Err()
	}
	result, err := r.Repository.Write(ctx, scope, target, payload, expected)
	if err == nil {
		r.committed.Store(true)
	}
	return result, err
}
