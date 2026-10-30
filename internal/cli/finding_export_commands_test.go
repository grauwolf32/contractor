package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/publicclient"
)

func TestAuditExportCLIForwardsDuplicateOptionAndExactDownload(t *testing.T) {
	for _, include := range []bool{false, true} {
		t.Run(map[bool]string{false: "default", true: "include"}[include], func(t *testing.T) {
			payload := []byte("{\"version\":\"2.1.0\"}\n")
			exportID := "export-" + strings.Repeat("a", 64)
			posts, gets := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set(publicclient.APIVersionHeader, publicclient.APIVersion)
				w.Header().Set("Content-Type", "application/json")
				switch r.Method {
				case http.MethodPost:
					posts++
					if r.URL.Path != "/v1/audits/audit-one/findings/exports" {
						t.Errorf("path %s", r.URL.Path)
					}
					var request findingexport.Request
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Error(err)
					}
					if request.IncludeDuplicates != include || request.ClientKey != "retry-key" || len(request.Findings) != 1 {
						t.Errorf("request: %+v", request)
					}
					w.WriteHeader(http.StatusCreated)
					_ = json.NewEncoder(w).Encode(map[string]any{"exportId": exportID, "artifact": map[string]any{"ref": map[string]string{"namespace": "audit-held", "name": "export", "revision": "r1"}, "digest": contentdigest.Bytes(payload), "mediaType": "application/json", "sizeBytes": len(payload)}, "snapshotAt": "2026-10-30T12:00:00Z", "selectedCount": 1, "resultCount": 1, "warningCount": 0, "replayed": false})
				case http.MethodGet:
					gets++
					if r.URL.Path != "/v1/audits/audit-one/findings/exports/"+exportID+"/content" {
						t.Errorf("download path %s", r.URL.Path)
					}
					_, _ = w.Write(payload)
				default:
					t.Errorf("unexpected method %s", r.Method)
				}
			}))
			defer server.Close()
			client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second, Transfer: true})
			if err != nil {
				t.Fatal(err)
			}
			var stdout bytes.Buffer
			c := New(strings.NewReader(`[{"findingId":"finding-one","revision":1}]`), &stdout, io.Discard, nil)
			printer, _ := NewPrinter(string(OutputJSON), &stdout)
			args := []string{"findings", "export", "audit-one", "--client-key", "retry-key"}
			if include {
				args = append(args, "--include-duplicates")
			}
			if err := c.runAudit(context.Background(), client, printer, args); err != nil {
				t.Fatal(err)
			}
			if posts != 1 || gets != 1 || !bytes.Equal(stdout.Bytes(), payload) {
				t.Fatalf("request/download: %d %d %q", posts, gets, stdout.Bytes())
			}
		})
	}
	if !transferCommand("audit", []string{"findings", "export"}) || !transferCommand("audit", []string{"findings", "download"}) {
		t.Fatal("export did not use Artifact transfer timeout")
	}
}

func TestAuditExportCLIRejectsInvalidSelectionBeforePublication(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests++; http.Error(w, "unexpected", 500) }))
	defer server.Close()
	client, err := publicclient.New(publicclient.Options{Server: server.URL, Token: "token", AllowHTTP: true, Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{`null`, `[{"findingId":"one","revision":1,"unknown":true}]`, `[{"findingId":"one","revision":1},{"findingId":"one","revision":1}]`, `[{"findingId":"one","revision":0}]`} {
		c := New(strings.NewReader(input), io.Discard, io.Discard, nil)
		printer, _ := NewPrinter(string(OutputJSON), io.Discard)
		if err := c.runAudit(context.Background(), client, printer, []string{"findings", "export", "audit-one", "--client-key", "key"}); err == nil {
			t.Fatalf("accepted %s", input)
		}
	}
	if requests != 0 {
		t.Fatal("invalid selection reached server")
	}
}
