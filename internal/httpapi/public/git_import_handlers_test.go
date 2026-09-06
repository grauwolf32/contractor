package public

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/gitimport"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"golang.org/x/crypto/ssh"
)

type snapshotClientFixture struct {
	fetch func(context.Context, gitimport.Remote, string, ssh.Signer) (gitimport.Snapshot, error)
}

func (c *snapshotClientFixture) Allowed(gitimport.Remote) bool { return true }
func (c *snapshotClientFixture) Fetch(ctx context.Context, remote gitimport.Remote, ref string, key ssh.Signer) (gitimport.Snapshot, error) {
	return c.fetch(ctx, remote, ref, key)
}

func TestGitImportPublicationAndAdmission(t *testing.T) {
	for _, backend := range []artifacts.BlobBackend{artifacts.BlobPostgres, artifacts.BlobFilesystem} {
		t.Run(string(backend), func(t *testing.T) {
			pool := isolatedPublicPool(t, t.Context())
			var blob artifacts.BlobStore = artifacts.PostgresBlobStore{}
			if err := artifacts.ClaimBlobBackend(t.Context(), pool, backend); err != nil {
				t.Fatal(err)
			}
			if backend == artifacts.BlobFilesystem {
				fs, err := artifacts.OpenFilesystemBlobStore(t.Context(), t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer fs.Close()
				blob = fs
			}
			ctx := artifacts.WithBlobRuntime(t.Context(), artifacts.NewBlobRuntime(blob, nil))
			cipher, _ := credentials.NewTokenCipher(bytes.Repeat([]byte{1}, 32))
			keys := credentials.NewGitKeys(pool, cipher)
			var count atomic.Int32
			client := &snapshotClientFixture{fetch: func(ctx context.Context, remote gitimport.Remote, ref string, _ ssh.Signer) (gitimport.Snapshot, error) {
				count.Add(1)
				if pool.Stat().AcquiredConns() != 0 {
					t.Error("SQL connection held during fetch")
				}
				return gitimport.Snapshot{Data: []byte("fixture ZIP bytes"), RepositoryURL: remote.URL, Commit: strings.Repeat("a", 40)}, nil
			}}
			importer, err := gitimport.NewImporter(pool, client, keys)
			if err != nil {
				t.Fatal(err)
			}
			service := artifacts.NewService(artifacts.NewPostgresRepository(pool))
			projects := projectstore.NewPostgresStore(pool)
			fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", newTestAuthentication(t), mustTestOrigins(t), false, nil, func(d *Dependencies) { d.GitImports = importer; d.Artifacts = service; d.Projects = projects })
			call := func(path, repository string, expected *string) *httptest.ResponseRecorder {
				body, _ := json.Marshal(map[string]string{"repositoryUrl": repository})
				request := authenticatedRequest(http.MethodPost, path, bytes.NewReader(body)).WithContext(ctx)
				request.Header.Set("Content-Type", "application/json")
				if expected == nil {
					request.Header.Set("If-None-Match", "*")
				} else {
					request.Header.Set("If-Match", quotedETag(expected))
				}
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, request)
				return response
			}
			path := "/v1/artifacts/source/repository/git-import"
			response := call(path, "https://example.test/first.git", nil)
			if response.Code != 201 {
				t.Fatalf("create: %d %s", response.Code, response.Body)
			}
			var first gitimport.ImportResult
			if err := json.Unmarshal(response.Body.Bytes(), &first); err != nil {
				t.Fatal(err)
			}
			if first.GitSource.RepositoryURL != "https://example.test:443/first.git" || first.Artifact.Revision == nil {
				t.Fatalf("result: %+v", first)
			}
			if response := call(path, "https://example.test/second.git", nil); response.Code != 409 {
				t.Fatalf("conflict: %d %s", response.Code, response.Body)
			}
			if count.Load() != 1 {
				t.Fatal("CAS conflict fetched Git")
			}
			if response := call(path, "https://example.test/second.git", first.Artifact.Revision); response.Code != 200 {
				t.Fatalf("replace: %d %s", response.Code, response.Body)
			}
			user, _ := service.User("user-1")
			old, err := user.Metadata(ctx, first.Artifact)
			if err != nil || old.GitSource == nil || old.GitSource.RepositoryURL != first.GitSource.RepositoryURL {
				t.Fatalf("old source: %+v %v", old, err)
			}
			var blobs, origins int
			if err := pool.QueryRow(ctx, `SELECT (SELECT count(*) FROM artifact_blobs),(SELECT count(*) FROM artifact_git_sources)`).Scan(&blobs, &origins); err != nil {
				t.Fatal(err)
			}
			if blobs != 1 || origins != 2 {
				t.Fatalf("dedup: %d %d", blobs, origins)
			}
			project, _, err := projects.Create(ctx, projectstore.CreateParams{ProjectID: "git-project", OwnerID: "user-1", Kind: projectstore.KindProject, Name: "Git", IdempotencyKey: "git-project", RequestDigest: "sha256:" + strings.Repeat("a", 64)})
			if err != nil {
				t.Fatal(err)
			}
			projectPath := "/v1/projects/" + project.ProjectID + "/artifacts/source/repository/git-import"
			if response := call(projectPath, "https://example.test/first.git", nil); response.Code != 201 {
				t.Fatalf("project: %d %s", response.Code, response.Body)
			}
			if response := call("/v1/projects/foreign/artifacts/source/repository/git-import", "https://example.test/first.git", nil); response.Code != 404 {
				t.Fatalf("foreign: %d %s", response.Code, response.Body)
			}
			originalFetch := client.fetch
			client.fetch = func(fetchCtx context.Context, remote gitimport.Remote, ref string, signer ssh.Signer) (gitimport.Snapshot, error) {
				if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{ProjectID: project.ProjectID, OwnerID: "user-1", ExpectedRevision: project.Revision}); err != nil {
					t.Fatal(err)
				}
				return originalFetch(fetchCtx, remote, ref, signer)
			}
			if response := call(strings.Replace(projectPath, "/repository/", "/racing/", 1), "https://example.test/first.git", nil); response.Code != 409 {
				t.Fatalf("deletion race: %d %s", response.Code, response.Body)
			}
			client.fetch = originalFetch
			// Hold the successful response callback: both admission slots must remain
			// occupied while metadata can still be queried through the pool.
			request := gitimport.ImportRequest{OwnerID: "user-1", Target: artifacts.ArtifactRef{Namespace: "source", Name: "admission"}, RepositoryURL: "https://example.test/first.git"}
			err = importer.DoImport(ctx, request, func(gitimport.ImportResult) {
				if err := importer.DoImport(ctx, request, func(gitimport.ImportResult) {}); !errors.Is(err, gitimport.ErrCapacity) {
					t.Errorf("capacity: %v", err)
				}
				if _, err := user.Metadata(ctx, first.Artifact); err != nil {
					t.Error(err)
				}
				for range 3 {
					_, release, err := artifacts.AcquireTransfer(ctx)
					if err != nil {
						t.Error(err)
						return
					}
					defer release()
				}
				if _, release, err := artifacts.AcquireTransfer(ctx); !errors.Is(err, artifacts.ErrTransferCapacity) {
					release()
					t.Errorf("Git response lost transfer lease: %v", err)
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			// A failed fetch leaves no revision and releases admission for the next one.
			client.fetch = func(context.Context, gitimport.Remote, string, ssh.Signer) (gitimport.Snapshot, error) {
				return gitimport.Snapshot{}, context.Canceled
			}
			if response := call(strings.Replace(path, "/repository/", "/cancelled/", 1), "https://example.test/first.git", nil); response.Code != 408 {
				t.Fatalf("cancelled: %d %s", response.Code, response.Body)
			}
			if _, err := user.Metadata(ctx, artifacts.ArtifactRef{Namespace: "source", Name: "cancelled"}); !errors.Is(err, artifacts.ErrArtifactNotFound) {
				t.Fatalf("partial artifact: %v", err)
			}
			client.fetch = originalFetch
			if response := call(strings.Replace(path, "/repository/", "/cancelled/", 1), "https://example.test/first.git", nil); response.Code != 201 {
				t.Fatalf("released admission: %d %s", response.Code, response.Body)
			}
		})
	}
}
