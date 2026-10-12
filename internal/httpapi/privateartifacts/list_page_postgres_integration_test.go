//go:build integration

package privateartifacts

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

const (
	pagedRunVisible = 7000
	pagedRunHidden  = 15000
)

// listResponse is one recorded private list response.
type listResponse struct {
	status int
	limit  int
	bytes  int
	refs   int
}

type listRecorder struct {
	mu        sync.Mutex
	responses []listResponse
}

func (r *listRecorder) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodGet || !strings.HasSuffix(request.URL.Path, "/artifacts") {
			next.ServeHTTP(w, request)
			return
		}
		capture := &capturingWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(capture, request)
		limit, _ := strconv.Atoi(request.URL.Query().Get("limit"))
		var page contracts.ArtifactListResult
		_ = json.Unmarshal(capture.body.Bytes(), &page)
		r.mu.Lock()
		defer r.mu.Unlock()
		r.responses = append(r.responses, listResponse{
			status: capture.status, limit: limit, bytes: capture.body.Len(), refs: len(page.Artifacts),
		})
	})
}

type capturingWriter struct {
	http.ResponseWriter
	status int
	body   bytes.Buffer
}

func (w *capturingWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *capturingWriter) Write(data []byte) (int, error) {
	w.body.Write(data)
	return w.ResponseWriter.Write(data)
}

// TestPostgresPagedListThroughPythonClientAndTool lists a Run whose hidden
// HTTP-body and Memory bindings alone exceed the Runtime's 1 MiB list response
// limit. Every page must stay within its limit and the byte bound, and the
// Python client and the list_artifacts tool must each return every visible
// binding exactly once.
func TestPostgresPagedListThroughPythonClientAndTool(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
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
		artifacts.Payload{MediaType: "text/plain", Data: []byte("source")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, runID := range []string{"run-a", "run-b"} {
		// Only the trusted Workflow input fork creates inputs/<slot>.
		if _, err := service.ForkInput(ctx, "user-1", source.Ref, runID, "source"); err != nil {
			t.Fatal(err)
		}
	}

	want := map[string]bool{"inputs/source": true}
	var namespaces, names []string
	add := func(namespace, name string, visible bool) {
		namespaces = append(namespaces, namespace)
		names = append(names, name)
		if visible {
			want[namespace+"/"+name] = true
		}
	}
	padding := strings.Repeat("x", 110)
	for i := 0; i < pagedRunVisible-1; i++ {
		// Visible names sort on both sides of the hidden ones in "builder".
		prefix := "artifact"
		if i%2 == 1 {
			prefix = "zz_report"
		}
		add("builder", fmt.Sprintf("%s_%06d_%s", prefix, i, padding)[:126], true)
	}
	for i := 0; i < pagedRunHidden; i++ {
		switch {
		case i%10 == 0:
			add("builder", fmt.Sprintf("memory.note_%06d", i), false)
		case i%25 == 1:
			add("skills", fmt.Sprintf("skill_%06d", i), false)
		default:
			add("builder", fmt.Sprintf("http.body.%016x.%06d_%s", i, i, padding)[:126], false)
		}
	}
	if len(want) != pagedRunVisible || len(names)-len(want)+1 != pagedRunHidden {
		t.Fatalf("seeded %d visible and %d hidden bindings", len(want), len(names)-len(want)+1)
	}
	seedPrivateRunBindings(t, ctx, pool, "run-a", namespaces, names)
	seedPrivateRunBindings(t, ctx, pool, "run-b", []string{"builder"}, []string{"foreign_only"})
	var hiddenBytes int
	if err := pool.QueryRow(ctx, `
SELECT coalesce(sum(length(namespace) + length(name) + 30), 0)::int FROM artifact_bindings
WHERE scope_kind = 'run' AND scope_id = 'run-a' AND (namespace = 'skills' OR name LIKE 'http.body.%' OR name LIKE 'memory.%')`,
	).Scan(&hiddenBytes); err != nil || hiddenBytes <= 1<<20 {
		t.Fatalf("hidden bindings must exceed the 1 MiB list response limit alone: %d, %v", hiddenBytes, err)
	}

	recorder := &listRecorder{}
	harness := newCrossLanguageHarness(t, artifacts.NewPostgresRepository(pool), recorder.wrap)
	result := harness.probe("--mode", "page", "--limit", strconv.Itoa(artifacts.MaxBindingListLimit))
	for _, view := range []string{"client", "tool"} {
		listed, _ := result[view].([]any)
		seen := make(map[string]bool, len(listed))
		for _, item := range listed {
			pair, _ := item.([]any)
			if len(pair) != 2 {
				t.Fatalf("%s returned a malformed ref %v", view, item)
			}
			key := fmt.Sprint(pair[0]) + "/" + fmt.Sprint(pair[1])
			if seen[key] || !want[key] {
				t.Fatalf("%s listed %q unexpectedly or more than once", view, key)
			}
			seen[key] = true
		}
		if len(seen) != len(want) {
			t.Fatalf("%s listed %d of %d visible bindings", view, len(seen), len(want))
		}
	}
	clientPages := (pagedRunVisible + artifacts.MaxBindingListLimit - 1) / artifacts.MaxBindingListLimit
	if result["clientPages"] != float64(clientPages) {
		t.Fatalf("client pages = %v, want %d", result["clientPages"], clientPages)
	}

	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	if len(recorder.responses) < clientPages+1 {
		t.Fatalf("recorded %d list responses", len(recorder.responses))
	}
	for index, response := range recorder.responses {
		if response.status != http.StatusOK || response.limit < 1 || response.refs > response.limit ||
			response.bytes >= 1<<20 {
			t.Fatalf("list response %d = %+v", index, response)
		}
	}
}

func seedPrivateRunBindings(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string, namespaces, names []string,
) {
	t.Helper()
	// Current bindings share one valid immutable version; both relations are
	// written in one statement for the deferred foreign key.
	_, err := pool.Exec(ctx, `WITH bindings AS (
 INSERT INTO artifact_bindings (scope_kind, scope_id, namespace, name, current_revision)
 SELECT 'run', $1, input.namespace, input.name, 'bulk-rev'
 FROM unnest($2::text[], $3::text[]) AS input(namespace, name)
 RETURNING scope_kind, scope_id, namespace, name, current_revision
) INSERT INTO artifact_binding_revisions (scope_kind, scope_id, namespace, name, revision, version_id)
 SELECT scope_kind, scope_id, namespace, name, current_revision, (SELECT version_id FROM artifact_versions LIMIT 1)
 FROM bindings`, runID, namespaces, names)
	if err != nil {
		t.Fatalf("seed bindings: %v", err)
	}
}

func createPrivateArtifactRun(t *testing.T, ctx context.Context, pool *pgxpool.Pool, runID string) {
	t.Helper()
	store := runstore.NewPostgresStore(pool)
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-1", WorkflowName: "artifact-list", WorkflowVersion: "1",
		WorkflowSchemaVersion: "contractor/v1alpha1",
		WorkflowSnapshot:      json.RawMessage(`{"ref":{"name":"artifact-list","version":"1"}}`),
		RuntimeConfig:         runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatalf("create WorkflowRun: %v", err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
	); err != nil {
		t.Fatalf("start WorkflowRun: %v", err)
	}
	if _, err := store.TransitionRun(
		ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
	); err != nil {
		t.Fatalf("start WorkflowRun: %v", err)
	}
}

func isolatedPrivateArtifactPool(t *testing.T, ctx context.Context) *pgxpool.Pool {
	t.Helper()
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatalf("open test admin pool: %v", err)
	}
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Fatalf("ping test database: %v", err)
	}
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	identifier := pgx.Identifier{"contractor_test_" + hex.EncodeToString(random)}.Sanitize()
	if _, err := admin.Exec(ctx, `CREATE SCHEMA `+identifier); err != nil {
		admin.Close()
		t.Fatalf("create isolated schema: %v", err)
	}
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = "contractor_test_" + hex.EncodeToString(random)
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("open isolated pool: %v", err)
	}
	t.Cleanup(func() {
		pool.Close()
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, err := admin.Exec(cleanupCtx, `DROP SCHEMA `+identifier+` CASCADE`); err != nil {
			t.Logf("drop isolated schema: %v", err)
		}
		admin.Close()
	})
	if _, err := persistencepostgres.ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}
	return pool
}
