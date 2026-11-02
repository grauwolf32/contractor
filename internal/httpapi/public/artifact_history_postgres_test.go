package public

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/helpers/sourcezip"
	"golang.org/x/crypto/ssh"
)

// The public Artifact API answers retained-history admission with the shared
// codes: a write that held revisions alone cannot fit is the permanent 409
// artifact_history_protected, and one that needs more maintenance than one
// transaction may do is the retryable 503 artifact_history_maintenance_pending
// with Retry-After, after a committed maintenance pass. Exact reads of a
// removed revision stay 404 not_found.
func TestPostgresPublicArtifactHistoryOutcomes(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	service := artifacts.NewService(artifactpostgres.NewRepository(pool))
	fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t),
		mustTestOrigins(t), false, nil, func(dependencies *testServices) { dependencies.Artifacts = service })
	policy := func(revisions int64) context.Context {
		configured, err := artifactpostgres.NewHistoryPolicy(revisions, artifactpostgres.DefaultHistoryMaxBytes)
		if err != nil {
			t.Fatal(err)
		}
		return artifactpostgres.WithHistoryPolicy(ctx, configured)
	}
	put := func(requestContext context.Context, name, header, value, body string) *httptest.ResponseRecorder {
		request := authenticatedRequest(http.MethodPut, "/v1/artifacts/notes/"+name, bytes.NewReader([]byte(body)))
		request = request.WithContext(requestContext)
		request.Header.Set("Content-Type", "text/plain")
		request.Header.Set(header, value)
		response := httptest.NewRecorder()
		fixture.handler.ServeHTTP(response, request)
		return response
	}
	write := func(requestContext context.Context, name string, count int) []string {
		t.Helper()
		var tags []string
		header, value := "If-None-Match", "*"
		for index := range count {
			response := put(requestContext, name, header, value, fmt.Sprintf("%s %d", name, index))
			if response.Code != http.StatusCreated && response.Code != http.StatusOK {
				t.Fatalf("write %s %d = %d %s", name, index, response.Code, response.Body.String())
			}
			header, value = "If-Match", response.Header().Get("ETag")
			tags = append(tags, value)
		}
		return tags
	}

	// Seventy revisions fit a limit of 100; at a limit of 2 the next write
	// needs 69 removals, more than one transaction's 64.
	tags := write(policy(100), "long", 70)
	response := put(policy(2), "long", "If-Match", tags[69], "next")
	assertErrorEnvelope(t, response, http.StatusServiceUnavailable, "artifact_history_maintenance_pending", true)
	if response.Header().Get("Retry-After") != "1" {
		t.Fatalf("Retry-After = %q", response.Header().Get("Retry-After"))
	}
	var retained int
	if err := pool.QueryRow(ctx, `SELECT retained_revisions FROM artifact_bindings
WHERE scope_kind = 'user' AND namespace = 'notes' AND name = 'long'`).Scan(&retained); err != nil {
		t.Fatal(err)
	}
	if retained != 70-64 {
		t.Fatalf("binding retains %d revisions after the committed pass, want 6", retained)
	}
	if response := put(policy(2), "long", "If-Match", tags[69], "next"); response.Code != http.StatusOK {
		t.Fatalf("retry after maintenance = %d %s", response.Code, response.Body.String())
	}
	removed := authenticatedRequest(http.MethodGet, "/v1/artifacts/notes/long?revision="+trimETag(tags[0]), bytes.NewReader(nil))
	recorder := httptest.NewRecorder()
	fixture.handler.ServeHTTP(recorder, removed)
	assertErrorEnvelope(t, recorder, http.StatusNotFound, "not_found", false)

	// Two held revisions leave no room under a limit of two.
	held := write(policy(2), "held", 2)
	if _, err := pool.Exec(ctx, `UPDATE artifact_binding_revisions SET hold_count = 1
WHERE scope_kind = 'user' AND namespace = 'notes' AND name = 'held'`); err != nil {
		t.Fatal(err)
	}
	response = put(policy(2), "held", "If-Match", held[1], "third")
	assertErrorEnvelope(t, response, http.StatusConflict, "artifact_history_protected", false)
	if response.Header().Get("Retry-After") != "" {
		t.Fatalf("protected history carries Retry-After %q", response.Header().Get("Retry-After"))
	}
}

// A Git import owns its transaction: a pending outcome rolls the import back,
// and the importer then commits the binding's maintenance pass on the pool
// before it answers, so the same import succeeds on a later attempt.
func TestPostgresGitImportCommitsHistoryMaintenanceAfterItsRollback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedPublicPool(t, ctx)
	limits := sourcezip.Limits{MaxExpandedBytes: 1 << 20, MaxFileBytes: 1 << 20, MaxEntries: 16}
	archive, err := sourcezip.Encode(ctx, []sourcezip.Member{{Name: "README", Size: 6, Write: func(w io.Writer) error {
		_, err := w.Write([]byte("readme"))
		return err
	}}}, sourcezip.Options{Limits: &limits})
	if err != nil {
		t.Fatal(err)
	}
	client := &snapshotClientFixture{fetch: func(context.Context, gitimport.Remote, string, gitimport.Mode, ssh.Signer) (gitimport.Snapshot, error) {
		return gitimport.Snapshot{Data: archive, RepositoryURL: "https://example.test:443/repo.git", Commit: strings.Repeat("b", 40)}, nil
	}}
	cipher, err := credentials.NewTokenCipher(bytes.Repeat([]byte{1}, 32))
	if err != nil {
		t.Fatal(err)
	}
	importer, err := gitimport.NewImporter(pool, client, credentials.NewGitKeys(pool, cipher), gitimport.Deadline)
	if err != nil {
		t.Fatal(err)
	}
	user, err := artifacts.NewService(artifactpostgres.NewRepository(pool)).User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	target := artifacts.ArtifactRef{Namespace: "source", Name: "repository"}
	var current *string
	for index := range 70 {
		written, err := user.Write(ctx, target, artifacts.Payload{MediaType: "text/plain", Data: fmt.Appendf(nil, "seed %d", index)}, current)
		if err != nil {
			t.Fatal(err)
		}
		current = written.Ref.Revision
	}
	configured, err := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	limited := artifactpostgres.WithHistoryPolicy(ctx, configured)
	request := gitimport.ImportRequest{OwnerID: "user-1", Target: target, ExpectedRevision: current, RepositoryURL: "https://example.test/repo.git"}
	err = importer.DoImport(limited, request, func(gitimport.ImportResult) { t.Fatal("a pending import responded") })
	var pending *artifacts.HistoryMaintenanceError
	if !errors.As(err, &pending) || !pending.Maintained || pending.Removed != 64 {
		t.Fatalf("import = %v, want a committed 64-revision maintenance pass", err)
	}
	var retained, sources int
	if err := pool.QueryRow(ctx, `SELECT binding.retained_revisions,
       (SELECT count(*) FROM artifact_binding_revisions AS revision
        JOIN artifact_versions AS version ON version.version_id = revision.version_id
        WHERE revision.scope_kind = binding.scope_kind AND revision.scope_id = binding.scope_id
          AND revision.namespace = binding.namespace AND revision.name = binding.name
          AND version.git_repository_url IS NOT NULL)
FROM artifact_bindings AS binding
WHERE binding.scope_kind = 'user' AND binding.namespace = 'source' AND binding.name = 'repository'`).Scan(&retained, &sources); err != nil {
		t.Fatal(err)
	}
	if retained != 70-64 || sources != 0 {
		t.Fatalf("after the rolled-back import: %d revisions, %d Git sources, want 6 and 0", retained, sources)
	}
	var result gitimport.ImportResult
	if err := importer.DoImport(limited, request, func(value gitimport.ImportResult) { result = value }); err != nil {
		t.Fatal(err)
	}
	if result.Artifact.Revision == nil || result.GitSource.ResolvedCommit != strings.Repeat("b", 40) {
		t.Fatalf("import after maintenance = %+v", result)
	}
}

func trimETag(tag string) string {
	if len(tag) >= 2 && tag[0] == '"' && tag[len(tag)-1] == '"' {
		return tag[1 : len(tag)-1]
	}
	return tag
}
