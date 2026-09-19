package publicclient

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func TestGeneratedFindingDecisionRequestsPreserveVerdictFields(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/public/finding-decision-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name    string
		Valid   bool
		Request json.RawMessage
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		if !tc.Valid {
			continue
		}
		t.Run(tc.Name, func(t *testing.T) {
			var fields map[string]any
			if err := json.Unmarshal(tc.Request, &fields); err != nil {
				t.Fatal(err)
			}
			var finding publicapi.DecideAuditFindingRequest
			switch fields["verdict"] {
			case "true_positive":
				err = finding.FromDecideAuditFindingTruePositiveRequest(publicapi.DecideAuditFindingTruePositiveRequest{Verdict: "true_positive", Severity: "high", Rationale: "Observed evidence"})
			case "duplicate":
				err = finding.FromDecideAuditFindingDuplicateRequest(publicapi.DecideAuditFindingDuplicateRequest{Verdict: "duplicate", DuplicateTargetId: "finding-original", Rationale: "Observed evidence"})
			default:
				err = finding.FromDecideAuditFindingOtherRequest(publicapi.DecideAuditFindingOtherRequest{Verdict: publicapi.DecideAuditFindingOtherRequestVerdict(fields["verdict"].(string)), Rationale: "Observed evidence"})
			}
			if err != nil {
				t.Fatal(err)
			}
			var body publicapi.DecideAuditReviewRequest
			if err := body.FromDecideAuditFindingRequest(finding); err != nil {
				t.Fatal(err)
			}
			request, err := publicapi.NewDecideAuditReviewRequest("https://contractor.example", "audit-1", "review-1", &publicapi.DecideAuditReviewParams{IdempotencyKey: "decision-1", IfMatch: `"1"`}, body)
			if err != nil {
				t.Fatal(err)
			}
			defer request.Body.Close()
			var actual map[string]any
			if err := json.NewDecoder(request.Body).Decode(&actual); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, fields) {
				t.Fatalf("typed body = %#v, want %#v", actual, fields)
			}
			if request.Header.Get("If-Match") != `"1"` || request.Header.Get("Idempotency-Key") != "decision-1" {
				t.Fatal("decision concurrency headers changed")
			}
		})
	}
}

func TestGeneratedGitRequestsExposeConditionalBrowserHeaders(t *testing.T) {
	const server = "https://contractor.example"
	body := publicapi.GitImportRequest{RepositoryUrl: "https://example.test/repository.git"}
	privateKey := "fixture"
	for _, cookie := range []bool{false, true} {
		var origin, csrf *string
		if cookie {
			o, c := server, strings.Repeat("c", 43)
			origin, csrf = &o, &c
		}
		builders := map[string]func() (*http.Request, error){
			"replace-key": func() (*http.Request, error) {
				return publicapi.NewReplaceGitKeyRequest(server, &publicapi.ReplaceGitKeyParams{Origin: origin, XCSRFToken: csrf}, publicapi.ReplaceGitKeyJSONRequestBody{PrivateKey: &privateKey})
			},
			"delete-key": func() (*http.Request, error) {
				return publicapi.NewDeleteGitKeyRequest(server, &publicapi.DeleteGitKeyParams{Origin: origin, XCSRFToken: csrf})
			},
			"user-import": func() (*http.Request, error) {
				return publicapi.NewImportGitArtifactRequest(server, "source", "repository", &publicapi.ImportGitArtifactParams{Origin: origin, XCSRFToken: csrf}, body)
			},
			"project-import": func() (*http.Request, error) {
				return publicapi.NewImportProjectGitArtifactRequest(server, "project-1", "source", "repository", &publicapi.ImportProjectGitArtifactParams{Origin: origin, XCSRFToken: csrf}, body)
			},
		}
		for name, build := range builders {
			request, err := build()
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if request.Body != nil {
				_, _ = io.Copy(io.Discard, request.Body)
				_ = request.Body.Close()
			}
			if cookie {
				if request.Header.Get("Origin") != *origin || request.Header.Get("X-CSRF-Token") != *csrf {
					t.Fatalf("%s lost browser headers", name)
				}
			} else if request.Header.Get("Origin") != "" || request.Header.Get("X-CSRF-Token") != "" {
				t.Fatalf("%s added browser headers to bearer request", name)
			}
			if request.Header.Get("If-Match") != "" || request.Header.Get("If-None-Match") != "" {
				t.Fatalf("%s changed implicit create to an explicit precondition", name)
			}
		}
	}
}
