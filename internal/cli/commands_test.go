package cli

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/grauwolf32/contractor/internal/localpki"
	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestWorkflowListUsesBearerAndSupportsFlagsAfterCommand(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/workflows" || request.URL.Query().Get("limit") != "1" {
			t.Errorf("request URL = %s", request.URL.String())
		}
		if request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("Authorization = %q", request.Header.Get("Authorization"))
		}
		writer.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"items":[{"ref":{"name":"review","version":"2"},"entryStage":"inspect","inputs":{},"outputs":{},"parameters":{}}],"page":{"hasMore":false}}`)
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "secret"
		}
		return ""
	})
	err := command.Run(context.Background(), []string{"--server", server.URL, "--output", "name", "workflow", "list", "--limit", "1"})
	if err != nil {
		t.Fatalf("Run: %v (stderr %s)", err, stderr.String())
	}
	if stdout.String() != "review@2\n" {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestGlobalHelpPrintsOnceWithoutConfiguration(t *testing.T) {
	var stdout, stderr bytes.Buffer
	command := New(strings.NewReader(""), &stdout, &stderr, func(string) string { return "" })
	if err := command.Run(context.Background(), []string{"--help"}); err != nil {
		t.Fatal(err)
	}
	if count := strings.Count(stderr.String(), "Usage: contractor"); count != 1 {
		t.Fatalf("usage count = %d in %q", count, stderr.String())
	}
}

func TestServerGroupHelpPrintsUsageWithoutConfiguration(t *testing.T) {
	for _, args := range [][]string{
		{"server", "--help"}, {"server", "-h"}, {"server", "config", "--help"}, {"server", "auth", "-h"},
	} {
		var stderr bytes.Buffer
		command := New(strings.NewReader(""), io.Discard, &stderr, func(string) string { return "" })
		if err := command.Run(context.Background(), args); err != nil {
			t.Fatalf("%v = %v, want usage and success", args, err)
		}
		if args[1] == "--help" && !strings.Contains(stderr.String(), "config validate") {
			t.Fatalf("server usage = %q", stderr.String())
		}
	}
	command := New(strings.NewReader(""), io.Discard, io.Discard, func(string) string { return "" })
	if err := command.Run(context.Background(), []string{"server", "config"}); err == nil {
		t.Fatal("server config without validate succeeded")
	}
}

func TestSourcePushPackagesDirectoryAndCreatesArtifact(t *testing.T) {
	source := filepath.Join(t.TempDir(), "source tree")
	if err := os.Mkdir(source, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(source, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
		writer.Header().Set("Content-Type", "application/json")
		switch request.Method {
		case http.MethodGet:
			if request.URL.Path != "/v1/artifacts/projects/repo/metadata" {
				t.Errorf("metadata path = %q", request.URL.Path)
			}
			writer.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(writer, `{"code":"not_found","message":"missing","retryable":false,"requestId":"req-1"}`)
		case http.MethodPut:
			if request.URL.Path != "/v1/artifacts/projects/repo" {
				t.Errorf("write path = %q", request.URL.Path)
			}
			if request.Header.Get("If-None-Match") != "*" || request.Header.Get("Content-Type") != "application/zip" {
				t.Errorf("write headers = %#v", request.Header)
			}
			payload, err := io.ReadAll(request.Body)
			if err != nil {
				t.Error(err)
			}
			archive, err := zip.NewReader(bytes.NewReader(payload), int64(len(payload)))
			if err != nil || len(archive.File) != 1 || archive.File[0].Name != "main.go" {
				t.Errorf("archive = %#v, err %v", archive, err)
			}
			writer.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(writer).Encode(map[string]any{
				"artifact":  map[string]string{"namespace": "projects", "name": "repo", "revision": "rev-1"},
				"mediaType": "application/zip", "size": len(payload),
			})
		default:
			t.Errorf("method = %s", request.Method)
		}
	}))
	defer server.Close()

	var stdout, stderr bytes.Buffer
	command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
		if name == "CONTRACTOR_API_TOKEN" {
			return "secret"
		}
		return ""
	})
	err := command.Run(context.Background(), []string{"--server", server.URL, "--output", "name", "source", "push", source, "--name", "repo"})
	if err != nil {
		t.Fatalf("Run: %v (stderr %s)", err, stderr.String())
	}
	if stdout.String() != "projects/repo@rev-1\n" {
		t.Fatalf("stdout = %q", stdout.String())
	}
}

func TestSourcePushRejectsEmptyBundleBeforeArtifactRequest(t *testing.T) {
	for _, ignored := range []bool{false, true} {
		name := "empty"
		if ignored {
			name = "fully ignored"
		}
		t.Run(name, func(t *testing.T) {
			source := t.TempDir()
			if ignored {
				if err := os.WriteFile(filepath.Join(source, ".contractorignore"), []byte("*\n"), 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(source, "main.go"), []byte("package main"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				requests.Add(1)
			}))
			defer server.Close()
			var stdout, stderr bytes.Buffer
			command := New(strings.NewReader(""), &stdout, &stderr, func(name string) string {
				if name == "CONTRACTOR_API_TOKEN" {
					return "secret"
				}
				return ""
			})
			err := command.Run(context.Background(), []string{
				"--server", server.URL, "source", "push", source, "--name", "repo",
			})
			if err == nil || !strings.Contains(err.Error(), "no files to package") || requests.Load() != 0 {
				t.Fatalf("empty source push = %v, Artifact requests = %d", err, requests.Load())
			}
		})
	}
}

func TestSourcePushRejectsRuntimeUnopenableBundleBeforeArtifactRequest(t *testing.T) {
	for _, test := range []struct {
		name  string
		files map[string]string
		want  string
	}{
		{"binary and ignored only", map[string]string{"assets/logo.png": "png", "node_modules/pkg/index.js": "js"}, "no file the Runtime can read"},
		{"file and directory collide after NFC", map[string]string{"cafe\u0301": "file", "caf\u00e9/notes.txt": "nested"}, "is both a file and a directory"},
	} {
		t.Run(test.name, func(t *testing.T) {
			source := t.TempDir()
			for path, content := range test.files {
				target := filepath.Join(source, filepath.FromSlash(path))
				if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
					t.Skipf("filesystem cannot hold the fixture names: %v", err)
				}
				if err := os.WriteFile(target, []byte(content), 0o600); err != nil {
					t.Skipf("filesystem cannot hold the fixture names: %v", err)
				}
			}
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				requests.Add(1)
			}))
			defer server.Close()
			command := New(strings.NewReader(""), io.Discard, io.Discard, func(name string) string {
				if name == "CONTRACTOR_API_TOKEN" {
					return "secret"
				}
				return ""
			})
			err := command.Run(context.Background(), []string{
				"--server", server.URL, "source", "push", source, "--name", "repo", "--include-ignored",
			})
			if err == nil || !strings.Contains(err.Error(), test.want) || requests.Load() != 0 {
				t.Fatalf("source push = %v, Artifact requests = %d; want %q before any request", err, requests.Load(), test.want)
			}
		})
	}
}

func TestPKICommandsIssueRuntimeCertificateWithoutServerContext(t *testing.T) {
	root := filepath.Join(t.TempDir(), "pki")
	var output bytes.Buffer
	command := New(strings.NewReader(""), &output, io.Discard, func(string) string { return "" })
	if err := command.Run(context.Background(), []string{"--output", "name", "pki", "init-ca", "--root", root}); err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if err := command.Run(context.Background(), []string{"--output", "json", "pki", "issue-runtime", "--root", root, "--name", "worker-1"}); err != nil {
		t.Fatal(err)
	}
	paths, err := localpki.AgentPaths(root, "worker-1")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{paths.Certificate, paths.PrivateKey} {
		if _, err := os.Stat(path); err != nil {
			t.Fatal(err)
		}
	}
	info, err := os.Stat(paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("private key mode = %v", info.Mode().Perm())
	}
	if !strings.Contains(output.String(), fmt.Sprintf(`"certificate": %q`, paths.Certificate)) {
		t.Fatalf("JSON output = %s", output.String())
	}
	if !strings.Contains(output.String(), fmt.Sprintf(`"caCertificate": %q`, localpki.CAPaths(root).Certificate)) {
		t.Fatalf("JSON output has no CA path: %s", output.String())
	}
	beforeCertificate, err := os.ReadFile(paths.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	beforeKey, err := os.ReadFile(paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if err := command.Run(context.Background(), []string{
		"--output", "name", "pki", "renew-runtime", "--root", root, "--name", "worker-1",
	}); err != nil {
		t.Fatal(err)
	}
	afterCertificate, err := os.ReadFile(paths.Certificate)
	if err != nil {
		t.Fatal(err)
	}
	afterKey, err := os.ReadFile(paths.PrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(beforeCertificate, afterCertificate) || !bytes.Equal(beforeKey, afterKey) ||
		output.String() != paths.Certificate+"\n" {
		t.Fatalf("Runtime renewal changed the key or did not replace the certificate: %q", output.String())
	}
}
