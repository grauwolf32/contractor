package public

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func archiveFixtureBytes(t *testing.T, path, content string) []byte {
	t.Helper()
	var data bytes.Buffer
	w := zip.NewWriter(&data)
	file, err := w.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func TestArtifactArchiveScopesAndExactRevision(t *testing.T) {
	fixture := newHandlerFixture(t)
	fixture.projects.projects["project-one"] = projectstore.Project{ProjectID: "project-one", OwnerID: "user-1"}
	fixture.runs.runs["run-one"] = runstore.WorkflowRun{RunID: "run-one", OwnerID: "user-1", State: runstore.RunSucceeded}
	user, _ := fixture.artifacts.User("user-1")
	project, _ := fixture.artifacts.Project("project-one")
	run, _ := fixture.artifacts.Run("run-one")
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		path      string
		store     artifacts.ScopedStore
		mediaType string
	}{
		{"/v1/artifacts/skills/package", user, "application/vnd.contractor.agent-skill+zip"},
		{"/v1/projects/project-one/artifacts/skills/package", project, "application/zip"},
		{"/v1/runs/run-one/artifacts/inputs/package", run, "application/x-zip-compressed"},
	} {
		t.Run(tc.path, func(t *testing.T) {
			namespace := "skills"
			if tc.store == run {
				namespace = "inputs"
			}
			first, err := tc.store.Write(t.Context(), contracts.ArtifactRef{Namespace: namespace, Name: "package"},
				artifacts.Payload{MediaType: tc.mediaType, Data: archiveFixtureBytes(t, "folder/SKILL.md", "# First\n<script>untrusted()</script>")}, nil)
			if err != nil {
				t.Fatal(err)
			}
			_, err = tc.store.Write(t.Context(), contracts.ArtifactRef{Namespace: namespace, Name: "package"},
				artifacts.Payload{MediaType: tc.mediaType, Data: archiveFixtureBytes(t, "folder/SKILL.md", "# Second")}, first.Ref.Revision)
			if err != nil {
				t.Fatal(err)
			}
			query := "?revision=" + url.QueryEscape(*first.Ref.Revision)
			index := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, tc.path+"/archive"+query, nil), true)
			if index.Code != http.StatusOK || index.Header().Get("ETag") != quotedETag(first.Ref.Revision) {
				t.Fatalf("index = %d %s", index.Code, index.Body.String())
			}
			var listing artifactArchiveResponse
			if err := json.Unmarshal(index.Body.Bytes(), &listing); err != nil || len(listing.Entries) != 2 {
				t.Fatalf("listing = %+v, %v", listing, err)
			}
			file := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, tc.path+"/archive/file"+query+"&path=folder%2FSKILL.md", nil), true)
			var body artifactArchiveFileResponse
			if err := json.Unmarshal(file.Body.Bytes(), &body); file.Code != http.StatusOK || err != nil || !strings.HasPrefix(body.Text, "# First") {
				t.Fatalf("file = %d %+v %v", file.Code, body, err)
			}
			if file.Header().Get("X-Content-Type-Options") != "nosniff" || strings.Contains(file.Body.String(), "<script>") {
				t.Fatal("unsafe JSON response")
			}
			for _, suffix := range []string{"/archive", "/archive/file?revision=revision-1", "/archive?revision=x&revision=y", "/archive?revision=x&path=a", "/archive/file?revision=x&path=../secret", "/archive/file?revision=x&path=a&path=b", "/archive?revision=x&scope=other"} {
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, tc.path+suffix, bytes.NewReader(nil)))
				if response.Code != http.StatusBadRequest {
					t.Errorf("%s = %d %s", suffix, response.Code, response.Body.String())
				}
			}
			for _, endpoint := range []string{"/archive", "/archive/file"} {
				for _, method := range []string{http.MethodHead, http.MethodPost} {
					response := httptest.NewRecorder()
					fixture.handler.ServeHTTP(response, authenticatedRequest(method, tc.path+endpoint+query, bytes.NewReader(nil)))
					if response.Code != http.StatusMethodNotAllowed {
						t.Errorf("%s %s = %d", method, endpoint, response.Code)
					}
				}
				response := httptest.NewRecorder()
				fixture.handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, tc.path+endpoint+query, nil))
				if response.Code != http.StatusUnauthorized {
					t.Errorf("unauthenticated = %d", response.Code)
				}
			}
		})
	}
}

func TestArtifactArchiveRejectsForeignAndHiddenScopes(t *testing.T) {
	fixture := newHandlerFixture(t)
	readsBefore := fixture.repository.reads
	fixture.projects.projects["foreign"] = projectstore.Project{ProjectID: "foreign", OwnerID: "user-2"}
	fixture.projects.projects["owned"] = projectstore.Project{ProjectID: "owned", OwnerID: "user-1"}
	fixture.runs.runs["foreign"] = runstore.WorkflowRun{RunID: "foreign", OwnerID: "user-2", State: runstore.RunSucceeded}
	fixture.runs.runs["owned"] = runstore.WorkflowRun{RunID: "owned", OwnerID: "user-1", State: runstore.RunSucceeded}
	for _, prefix := range []string{
		"/v1/projects/foreign/artifacts/files/package",
		"/v1/runs/foreign/artifacts/files/package",
		"/v1/artifacts/" + artifactpolicy.AuditStandardCatalogNamespace + "/package",
		"/v1/projects/owned/artifacts/" + artifactpolicy.AuditManagedProjectNamespacePrefix + "test/package",
		"/v1/runs/owned/artifacts/" + artifactpolicy.RunSystemNamespace + "/package",
	} {
		for _, endpoint := range []string{"/archive?revision=r1", "/archive/file?revision=r1&path=SKILL.md"} {
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, prefix+endpoint, bytes.NewReader(nil)))
			if response.Code != http.StatusNotFound {
				t.Errorf("%s = %d %s", prefix, response.Code, response.Body.String())
			}
		}
	}
	if fixture.repository.reads != readsBefore {
		t.Fatal("foreign or hidden archive request reached artifact storage")
	}
}

func TestArtifactArchiveErrorsAreBounded(t *testing.T) {
	fixture := newHandlerFixture(t)
	store, _ := fixture.artifacts.User("user-1")
	for _, tc := range []struct {
		name, mediaType, path, text, endpoint, code string
		status                                      int
	}{
		{"invalid", "application/zip", "../unsafe-private-name", "", "/archive", "archive_invalid", 422},
		{"media", "text/plain", "file", "text", "/archive", "archive_media_type", 415},
		{"binary", "application/zip", "file", "\x00\xff", "/archive/file", "archive_file_not_text", 422},
		{"large", "application/zip", "file", strings.Repeat("a", 256*1024+1), "/archive/file", "archive_preview_limit", 422},
		{"missing", "application/zip", "another", "text", "/archive/file", "not_found", 404},
	} {
		t.Run(tc.name, func(t *testing.T) {
			written, err := store.Write(t.Context(), contracts.ArtifactRef{Namespace: "files", Name: tc.name},
				artifacts.Payload{MediaType: tc.mediaType, Data: archiveFixtureBytes(t, tc.path, tc.text)}, nil)
			if err != nil {
				t.Fatal(err)
			}
			path := "/v1/artifacts/files/" + tc.name + tc.endpoint + "?revision=" + url.QueryEscape(*written.Ref.Revision)
			if tc.endpoint == "/archive/file" {
				path += "&path=file"
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, authenticatedRequest(http.MethodGet, path, bytes.NewReader(nil)))
			var body errorResponse
			if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil || response.Code != tc.status || body.Code != tc.code {
				t.Fatalf("response = %d %s (%v)", response.Code, response.Body.String(), err)
			}
			if strings.Contains(response.Body.String(), "unsafe-private-name") {
				t.Fatal("parser details leaked")
			}
		})
	}
}
