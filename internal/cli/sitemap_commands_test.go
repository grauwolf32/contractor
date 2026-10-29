package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestProjectSitemapCLIUsesGeneratedOwnerClientAndPinnedIntent(t *testing.T) {
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.Path)
		w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		w.Header().Set("Content-Type", "application/json")
		if r.Method == "GET" {
			if r.URL.Query().Get("revision") != "rev-selected" {
				t.Error("exact selection changed")
			}
			w.Header().Set("Content-Type", projectsitemap.MediaType)
			w.Header().Set("ETag", `"rev-selected"`)
			if err := json.NewEncoder(w).Encode(projectsitemap.Empty()); err != nil {
				t.Error(err)
			}
			return
		}
		if r.Header.Get("Idempotency-Key") != "original-key" {
			t.Error("mutation key changed")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["expectedRevision"] != "rev-original" {
			t.Error("CLI refreshed stale edit intent")
		}
		_, _ = io.WriteString(w, `{"artifact":{"namespace":"sitemap","name":"map","revision":"rev-result"},"changed":true,"replayed":false}`)
	}))
	defer server.Close()
	file := filepath.Join(t.TempDir(), "mutation.json")
	if err := os.WriteFile(file, []byte(`{"operation":"remove","expectedRevision":"rev-original","markerId":"hidden","memberIds":["selected"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	var output, stderr bytes.Buffer
	command := New(strings.NewReader(""), &output, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "test-token"
		}
		return ""
	})
	for _, args := range [][]string{
		{"get", "project-one", "--revision", "rev-selected"},
		{"mutate", "project-one", "--file", file, "--idempotency-key", "original-key"},
	} {
		prefix := []string{"--server", server.URL, "--output", "json", "project", "sitemap"}
		if err := command.Run(context.Background(), append(prefix, args...)); err != nil {
			t.Fatalf("command: %v; %s", err, stderr.String())
		}
	}
	if len(calls) != 2 || calls[0] != "GET /v1/projects/project-one/artifacts/sitemap/map" || calls[1] != "POST /v1/projects/project-one/sitemap/mutations" {
		t.Fatalf("unexpected extra reads or calls: %v", calls)
	}
}

func TestProjectSitemapCLIArtifactReadBoundaries(t *testing.T) {
	valid, err := json.Marshal(projectsitemap.Empty())
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, revision, mediaType, body, code, lifecycle, wantRevision, wantCode string
		tags                                                                     []string
		artifactStatus, projectStatus, projectReads                              int
		wantError                                                                bool
	}{
		{name: "current", mediaType: projectsitemap.MediaType, tags: []string{`"current-r1"`}, body: string(valid), wantRevision: "current-r1"},
		{name: "exact", revision: "retained-r1", mediaType: projectsitemap.MediaType, tags: []string{`"retained-r1"`}, body: string(valid), wantRevision: "retained-r1"},
		{name: "absent-current-active-project", artifactStatus: 404, code: "not_found", projectStatus: 200, lifecycle: "active", projectReads: 1},
		{name: "absent-exact", revision: "evicted-r1", artifactStatus: 404, code: "not_found", wantError: true, wantCode: "not_found"},
		{name: "absent-project", artifactStatus: 404, code: "not_found", projectStatus: 404, projectReads: 1, wantError: true, wantCode: "not_found"},
		{name: "foreign-project", artifactStatus: 404, code: "not_found", projectStatus: 404, projectReads: 1, wantError: true, wantCode: "not_found"},
		{name: "deleting-project-without-map", artifactStatus: 404, code: "not_found", projectStatus: 200, lifecycle: "deleting", projectReads: 1, wantError: true, wantCode: "project_deleting"},
		{name: "corrupt-content", artifactStatus: 500, code: "artifact_content_corrupt", wantError: true, wantCode: "artifact_content_corrupt"},
		{name: "wrong-media-type", mediaType: "application/json", tags: []string{`"current-r1"`}, body: string(valid), wantError: true},
		{name: "malformed-map", mediaType: projectsitemap.MediaType, tags: []string{`"current-r1"`}, body: `{}`, wantError: true},
		{name: "wrong-exact-etag", revision: "retained-r1", mediaType: projectsitemap.MediaType, tags: []string{`"other-r1"`}, body: string(valid), wantError: true},
		{name: "missing-etag", mediaType: projectsitemap.MediaType, body: string(valid), wantError: true},
		{name: "weak-etag", mediaType: projectsitemap.MediaType, tags: []string{`W/"current-r1"`}, body: string(valid), wantError: true},
		{name: "duplicate-etag", mediaType: projectsitemap.MediaType, tags: []string{`"current-r1"`, `"current-r1"`}, body: string(valid), wantError: true},
		{name: "oversized-map", mediaType: projectsitemap.MediaType, tags: []string{`"current-r1"`}, body: strings.Repeat(" ", projectsitemap.MaxDocumentBytes+1), wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			projectReads := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
				w.Header().Set("Content-Type", "application/json")
				if r.URL.Path == "/v1/projects/project-one" {
					projectReads++
					w.WriteHeader(test.projectStatus)
					if test.projectStatus == 200 {
						_, _ = fmt.Fprintf(w, `{"projectId":"project-one","lifecycle":%q}`, test.lifecycle)
					} else {
						_, _ = io.WriteString(w, `{"code":"not_found","message":"not found","retryable":false,"requestId":"test"}`)
					}
					return
				}
				if r.Method != http.MethodGet || r.URL.Path != "/v1/projects/project-one/artifacts/sitemap/map" || r.URL.Query().Get("revision") != test.revision {
					t.Errorf("unexpected read: %s %s", r.Method, r.URL)
				}
				if test.artifactStatus != 0 {
					w.WriteHeader(test.artifactStatus)
					_, _ = fmt.Fprintf(w, `{"code":%q,"message":"unavailable","retryable":false,"requestId":"test"}`, test.code)
					return
				}
				w.Header().Set("Content-Type", test.mediaType)
				for _, tag := range test.tags {
					w.Header().Add("ETag", tag)
				}
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "fixture-token", AllowHTTP: true, Timeout: 5 * time.Second})
			if err != nil {
				t.Fatal(err)
			}
			view, err := readProjectSitemap(t.Context(), client, "project-one", test.revision)
			if (err != nil) != test.wantError || projectReads != test.projectReads {
				t.Fatalf("read = %+v, %v; Project reads %d want %d", view, err, projectReads, test.projectReads)
			}
			if test.wantCode != "" {
				var failure *publicclient.APIError
				if !errors.As(err, &failure) || failure.Code != test.wantCode {
					t.Fatalf("error = %v want %s", err, test.wantCode)
				}
			}
			if !test.wantError && (view.Revision != test.wantRevision || view.Document.Version != projectsitemap.Empty().Version) {
				t.Fatalf("invalid read result: %+v", view)
			}
		})
	}
}
