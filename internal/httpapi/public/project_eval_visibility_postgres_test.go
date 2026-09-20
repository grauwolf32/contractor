package public

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func TestProjectCollectionExcludesEvalExecutionWorkspacesBeforePagination(t *testing.T) {
	h := newEvalAPIHarness(t)
	projects := projectstore.NewPostgresStore(h.pool)
	// A user-created Project with an Eval-looking ID and name must remain visible.
	ordinary, _, err := projects.Create(t.Context(), projectstore.CreateParams{
		OwnerID: "user-1", ProjectID: "project-eval-user-created", Kind: projectstore.KindProject,
		Name: "Eval user-created", IdempotencyKey: "eval-looking-project",
		RequestDigest: evaldomain.Digest([]byte("eval-looking-project")),
	})
	if err != nil {
		t.Fatal(err)
	}

	e, manifest, _ := h.registerExternal(t, "audit")
	memberPath := "/v1/eval-experiments/" + e.ID + "/members/" + manifest.Members[0].MemberID
	h.request(t, "POST", memberPath+"/submissions", evaldomain.Submission{PlanSHA256: *e.PlanSHA256}, "submit", "", 202)
	h.tick(t)
	submission, err := evalstore.NewPostgresStore(h.pool).Submission(t.Context(), "user-1", e.ID, manifest.Members[0].MemberID)
	if err != nil || submission.ExecutionProjectID == nil || submission.ExecutionID == nil {
		t.Fatalf("Audit workspace was not created: %+v, %v", submission, err)
	}
	workspaceID := *submission.ExecutionProjectID
	// The association must keep hiding a renamed workspace as well.
	h.request(t, "PATCH", "/v1/projects/"+workspaceID, map[string]string{"name": "Renamed workspace"}, "", `"1"`, 200)

	assertCollections := func(t *testing.T) {
		t.Helper()
		for _, tc := range []struct {
			query string
			want  []string
		}{
			{"kind=project&", []string{ordinary.ProjectID, "ordinary"}},
			{"kind=evaluation&", []string{"evaluation"}},
			{"", []string{ordinary.ProjectID, "ordinary", "evaluation"}},
		} {
			path := "/v1/projects?" + tc.query + "limit=1"
			var got []string
			for {
				page := apiDecode[projectPageResponse](t, h.request(t, "GET", path, nil, "", "", 200))
				if len(page.Items) != 1 {
					t.Fatalf("%s returned an empty or oversized page: %+v", path, page)
				}
				got = append(got, page.Items[0].ProjectID)
				if len(got) > len(tc.want) {
					t.Fatalf("%s returned unexpected Projects: %v", tc.query, got)
				}
				if !page.Page.HasMore {
					if page.Page.NextCursor != nil {
						t.Fatal("last page retained a cursor")
					}
					break
				}
				if page.Page.NextCursor == nil {
					t.Fatal("next page cursor is missing")
				}
				path = "/v1/projects?" + tc.query + "limit=1&cursor=" + url.QueryEscape(*page.Page.NextCursor)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("%s Projects = %v, want %v", tc.query, got, tc.want)
			}
		}
	}
	assertCollections(t)
	workspace := apiDecode[projectResponse](t, h.request(t, "GET", "/v1/projects/"+workspaceID, nil, "", "", 200))
	if workspace.ProjectID != workspaceID || workspace.Name != "Renamed workspace" {
		t.Fatalf("direct workspace access changed: %+v", workspace)
	}
	auditResponse := httptest.NewRecorder()
	h.handler.ServeHTTP(auditResponse, newPublicContractRequest("GET", "/v1/audits/"+*submission.ExecutionID, nil))
	if auditResponse.Code != http.StatusOK {
		t.Fatalf("direct Audit access = %d: %s", auditResponse.Code, auditResponse.Body.String())
	}
	audit := apiDecode[struct {
		ProjectID string `json:"projectId"`
	}](t, auditResponse)
	if audit.ProjectID != workspaceID {
		t.Fatalf("Audit workspace = %s, want %s", audit.ProjectID, workspaceID)
	}
	inventory := apiDecode[struct {
		Items []evalstore.InventoryEntry `json:"items"`
	}](t, h.request(t, "GET", memberPath+"/executions", nil, "", "", 200))
	if len(inventory.Items) != 1 || !inventory.Items[0].Available ||
		inventory.Items[0].Execution == nil || inventory.Items[0].Execution.ID != *submission.ExecutionID ||
		inventory.Items[0].Execution.Kind != "audit" {
		t.Fatalf("Eval evidence navigation lost its workspace: %+v", inventory)
	}
	if _, err := projects.Get(t.Context(), "user-2", workspaceID); !errors.Is(err, projectstore.ErrNotFound) {
		t.Fatalf("foreign workspace access = %v", err)
	}
	h.request(t, "DELETE", "/v1/projects/"+workspaceID, nil, "", `"2"`, 202)
	assertCollections(t)
}
