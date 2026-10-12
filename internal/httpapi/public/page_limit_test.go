package public

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// Every fake behind a public paged list applies the bound of the real store
// it stands for. Fakes that ignored Limit let handlers fetch limit+1 rows from
// stores that allowed only 200, and nothing failed until PostgreSQL.
func TestPublicListFakesRejectReadsBeyondThePageFetchBound(t *testing.T) {
	ctx := t.Context()
	fixture := newHandlerFixture(t)
	scope, err := artifacts.UserScope("user-1")
	if err != nil {
		t.Fatal(err)
	}
	userArtifacts, err := fixture.artifacts.User("user-1")
	if err != nil {
		t.Fatal(err)
	}
	written, err := userArtifacts.Write(ctx, contracts.ArtifactRef{Namespace: "projects", Name: "page"},
		artifacts.Payload{MediaType: "text/plain", Data: []byte("page")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	audits := &fakeAuditManagement{}
	proposals := &fakeFindingProposalManagement{}
	ids := func(count int) []string {
		result := make([]string, count)
		for index := range result {
			result[index] = fmt.Sprintf("id-%03d", index)
		}
		return result
	}
	reads := []struct {
		name    string
		bound   int
		invalid error
		read    func(rows int) error
	}{
		{"projects", pagelimit.MaxFetchRows, projectstore.ErrInvalid, func(rows int) error {
			_, err := fixture.projects.List(ctx, projectstore.ListParams{OwnerID: "user-1", Limit: rows})
			return err
		}},
		{"artifact bindings", pagelimit.MaxFetchRows, artifacts.ErrInvalidName, func(rows int) error {
			_, err := fixture.repository.ListMetadata(ctx, scope, artifacts.BindingPageQuery{Limit: rows})
			return err
		}},
		{"artifact versions", pagelimit.MaxFetchRows, artifacts.ErrInvalidName, func(rows int) error {
			_, err := fixture.repository.ListVersions(ctx, scope, written.Ref, artifacts.VersionPageQuery{Limit: rows})
			return err
		}},
		{"artifact lineage", pagelimit.MaxFetchRows, artifacts.ErrInvalidName, func(rows int) error {
			_, err := fixture.repository.ListLineage(ctx, scope, written.Ref, artifacts.LineagePageQuery{Limit: rows})
			return err
		}},
		{"runs", pagelimit.MaxFetchRows, runstore.ErrInvalid, func(rows int) error {
			_, err := fixture.runs.ListRuns(ctx, runstore.ListRunsParams{OwnerID: "user-1", Limit: rows})
			return err
		}},
		{"run queue", pagelimit.MaxFetchRows, runstore.ErrInvalid, func(rows int) error {
			_, err := fixture.runs.ListRunQueue(ctx, runstore.ListRunQueueParams{OwnerID: "user-1", Limit: rows})
			return err
		}},
		{"run deletion gates", pagelimit.MaxRows, runstore.ErrInvalid, func(rows int) error {
			_, err := fixture.runs.DeletableRuns(ctx, "user-1", ids(rows))
			return err
		}},
		{"audits", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.List(ctx, auditstore.ListParams{Limit: rows})
			return err
		}},
		{"audit events", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListEventsPage(ctx, auditstore.EventListParams{Limit: rows})
			return err
		}},
		{"audit items", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListItems(ctx, auditstore.ListItemsParams{Limit: rows})
			return err
		}},
		{"audit item attempts", pagelimit.MaxRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListItemAttempts(ctx, "user-1", "audit-1", ids(rows))
			return err
		}},
		{"audit coverage", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListCoverage(ctx, "user-1", "audit-1", "round-1", -1, rows, nil)
			return err
		}},
		{"audit findings", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListFindingsPage(ctx, auditstore.FindingListParams{Limit: rows})
			return err
		}},
		{"audit reviews", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListReviewsPage(ctx, auditstore.ReviewListParams{Limit: rows})
			return err
		}},
		{"audit finding provenance", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListFindingProvenance(ctx, auditstore.ProvenanceListParams{Limit: rows})
			return err
		}},
		{"owner findings", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListOwnerFindings(ctx, auditstore.OwnerFindingListParams{Limit: rows})
			return err
		}},
		{"owner reviews", pagelimit.MaxFetchRows, auditstore.ErrInvalid, func(rows int) error {
			_, err := audits.ListOwnerReviews(ctx, auditstore.OwnerReviewListParams{Limit: rows})
			return err
		}},
		{"run finding proposals", pagelimit.MaxFetchRows, findingintake.ErrInvalid, func(rows int) error {
			_, err := proposals.ListRun(ctx, "user-1", "run-1", findingintake.ListQuery{Limit: rows})
			return err
		}},
		{"audit finding proposals", pagelimit.MaxFetchRows, findingintake.ErrInvalid, func(rows int) error {
			_, err := proposals.ListAuditInbox(ctx, "user-1", "audit-1", findingintake.ListQuery{Limit: rows})
			return err
		}},
		{"LLM credentials", pagelimit.MaxFetchRows, credentials.ErrInvalid, func(rows int) error {
			_, err := fixture.credentials.ListCredentials(ctx, "", rows)
			return err
		}},
		{"RuntimeConfig versions", pagelimit.MaxFetchRows, runtimeconfig.ErrInvalid, func(rows int) error {
			_, err := fixture.runtimeConfigs.ListVersions(ctx, "", "", rows)
			return err
		}},
		{"Runtime labels", pagelimit.MaxFetchRows, runtimeconfig.ErrInvalid, func(rows int) error {
			_, err := fixture.runtimeConfigs.ListBindings(ctx, "", rows)
			return err
		}},
		{"Runtime credentials", pagelimit.MaxFetchRows, credentials.ErrRuntimeCredentialInvalid, func(rows int) error {
			_, err := fixture.runtimeCredentials.List(ctx, "", rows)
			return err
		}},
		// The principal service takes the page size and adds the look-ahead row.
		{"Runtime Agent principals", pagelimit.MaxRows, runtimeconfig.ErrInvalid, func(rows int) error {
			_, _, err := fixture.runtimePrincipals.List(ctx, "", rows)
			return err
		}},
	}
	for _, read := range reads {
		t.Run(read.name, func(t *testing.T) {
			if err := read.read(read.bound); err != nil {
				t.Fatalf("read of %d rows = %v, want accepted", read.bound, err)
			}
			if err := read.read(read.bound + 1); !errors.Is(err, read.invalid) {
				t.Fatalf("read of %d rows = %v, want %v", read.bound+1, err, read.invalid)
			}
		})
	}
}

// The OpenAPI Limit parameter is what the UI and Go clients are generated
// from; its range and default are the Server's.
func TestPublicOpenAPILimitParameterIsTheSharedPageBound(t *testing.T) {
	parameter := loadPublicOpenAPI(t).Components.Parameters["Limit"]
	if parameter == nil || parameter.Value == nil || parameter.Value.Schema == nil || parameter.Value.Schema.Value == nil {
		t.Fatal("OpenAPI declares no #/components/parameters/Limit schema")
	}
	schema := parameter.Value.Schema.Value
	if schema.Min == nil || *schema.Min != 1 || schema.Max == nil || *schema.Max != pagelimit.MaxRows ||
		fmt.Sprint(schema.Default) != fmt.Sprint(defaultPageLimit) {
		t.Fatalf("OpenAPI Limit = min %v max %v default %v, want 1..%d default %d",
			schema.Min, schema.Max, schema.Default, pagelimit.MaxRows, defaultPageLimit)
	}
}

// pagedListOperations returns the operationIds that take the shared Limit
// parameter.
func pagedListOperations(t *testing.T) map[string]bool {
	t.Helper()
	result := make(map[string]bool)
	for _, item := range loadPublicOpenAPI(t).Paths.Map() {
		for _, operation := range item.Operations() {
			for _, parameter := range operation.Parameters {
				if parameter.Ref == "#/components/parameters/Limit" {
					result[operation.OperationID] = true
				}
			}
		}
	}
	if len(result) == 0 {
		t.Fatal("OpenAPI has no operation with the shared Limit parameter")
	}
	return result
}

// checkLargestPublicPage calls every paged list operation with the largest
// documented page and one beyond it. paths maps each operationId that takes
// the shared Limit parameter to a request path; an operation without a path
// fails, so a new paged list cannot skip the check. fullPages names the
// operations whose collection holds more than one page, which must return a
// full page and a cursor.
func checkLargestPublicPage(t *testing.T, handler http.Handler, paths map[string]string, fullPages ...string) {
	t.Helper()
	operations := pagedListOperations(t)
	for operation := range operations {
		if _, ok := paths[operation]; !ok {
			t.Errorf("paged list %s has no request path in this test", operation)
		}
	}
	for operation := range paths {
		if !operations[operation] {
			t.Errorf("%s does not take the shared Limit parameter", operation)
		}
	}
	for operation, path := range paths {
		t.Run(operation, func(t *testing.T) {
			separator := "?"
			if strings.Contains(path, "?") {
				separator = "&"
			}
			largest := httptest.NewRecorder()
			handler.ServeHTTP(largest, newPublicContractRequest(http.MethodGet,
				fmt.Sprintf("%s%slimit=%d", path, separator, pagelimit.MaxRows), nil))
			if largest.Code != http.StatusOK {
				t.Fatalf("limit=%d = %d: %s", pagelimit.MaxRows, largest.Code, largest.Body.String())
			}
			var page struct {
				Items *[]json.RawMessage `json:"items"`
				Page  *pageInfoResponse  `json:"page"`
			}
			if err := json.Unmarshal(largest.Body.Bytes(), &page); err != nil || page.Items == nil || page.Page == nil {
				t.Fatalf("limit=%d page is not an items page (%v): %s", pagelimit.MaxRows, err, largest.Body.String())
			}
			if len(*page.Items) > pagelimit.MaxRows || page.Page.HasMore != (page.Page.NextCursor != nil) {
				t.Fatalf("limit=%d returned %d items and page %+v", pagelimit.MaxRows, len(*page.Items), *page.Page)
			}
			if slices.Contains(fullPages, operation) && (len(*page.Items) != pagelimit.MaxRows || !page.Page.HasMore) {
				t.Fatalf("limit=%d over a larger collection returned %d items and page %+v",
					pagelimit.MaxRows, len(*page.Items), *page.Page)
			}
			beyond := httptest.NewRecorder()
			handler.ServeHTTP(beyond, newPublicContractRequest(http.MethodGet,
				fmt.Sprintf("%s%slimit=%d", path, separator, pagelimit.MaxRows+1), nil))
			if beyond.Code != http.StatusBadRequest {
				t.Fatalf("limit=%d = %d: %s", pagelimit.MaxRows+1, beyond.Code, beyond.Body.String())
			}
		})
	}
}
