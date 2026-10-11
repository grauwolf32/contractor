package auditstore

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestPostgresListFiltersStatesProfilesCursorAndListedProjects drives the one
// Audit list statement through a Project list with a one-state filter, the
// owner-wide list with a state set, a profile filter and keyset pages; the
// owner-wide list leaves out the Audits of a deleting Project.
func TestPostgresListFiltersStatesProfilesCursorAndListedProjects(t *testing.T) {
	databaseURL := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditPool(t, ctx, databaseURL)
	projects := projectstore.NewPostgresStore(pool)
	store := NewPostgresStore(pool)
	const owner = "owner-list"
	for _, projectID := range []string{"project-listed", "project-deleting"} {
		if _, _, err := projects.Create(ctx, projectstore.CreateParams{
			ProjectID: projectID, OwnerID: owner, Kind: projectstore.KindProject, Name: projectID,
			IdempotencyKey: projectID, RequestDigest: testDigest("a"),
		}); err != nil {
			t.Fatal(err)
		}
	}
	create := func(auditID, projectID, profile string) {
		t.Helper()
		if _, _, err := store.CreateDraft(ctx, CreateDraftParams{
			AuditID: auditID, OwnerID: owner, ProjectID: projectID,
			Profile:         ProfileIdentity{Name: profile, Version: "1", Digest: testDigest("b")},
			ProfileSnapshot: json.RawMessage(`{}`), InputSelection: json.RawMessage(`{}`),
			Limits:         Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1024},
			IdempotencyKey: auditID, RequestDigest: testDigest("c"),
		}); err != nil {
			t.Fatal(err)
		}
	}
	create("audit-draft", "project-listed", "checklist")
	create("audit-other-profile", "project-listed", "other")
	create("audit-failed", "project-listed", "checklist")
	create("audit-hidden", "project-deleting", "checklist")
	if _, err := pool.Exec(ctx, `UPDATE audits SET state = 'failed', baseline_snapshot = '{}'::jsonb,
       started_at = clock_timestamp(), finished_at = clock_timestamp()
 WHERE audit_id = 'audit-failed'`); err != nil {
		t.Fatal(err)
	}
	if _, _, err := projects.BeginDeletion(ctx, projectstore.BeginDeletionParams{
		OwnerID: owner, ProjectID: "project-deleting", ExpectedRevision: 1,
	}); err != nil {
		t.Fatal(err)
	}

	list := func(params ListParams) []string {
		t.Helper()
		params.OwnerID = owner
		if params.Limit == 0 {
			params.Limit = 10
		}
		audits, err := store.List(ctx, params)
		if err != nil {
			t.Fatalf("List(%+v) = %v", params, err)
		}
		ids := make([]string, len(audits))
		for index, audit := range audits {
			ids[index] = audit.AuditID
		}
		return ids
	}
	listed, deleting := "project-listed", "project-deleting"
	profile, version := "checklist", "1"
	for name, test := range map[string]struct {
		params ListParams
		want   []string
	}{
		"Project":                  {ListParams{ProjectID: &listed}, []string{"audit-failed", "audit-other-profile", "audit-draft"}},
		"Project draft":            {ListParams{ProjectID: &listed, States: []AuditState{AuditDraft}}, []string{"audit-other-profile", "audit-draft"}},
		"Project failed":           {ListParams{ProjectID: &listed, States: []AuditState{AuditFailed}}, []string{"audit-failed"}},
		"Project profile":          {ListParams{ProjectID: &listed, ProfileName: &profile, ProfileVersion: &version}, []string{"audit-failed", "audit-draft"}},
		"deleting Project":         {ListParams{ProjectID: &deleting}, []string{"audit-hidden"}},
		"owner-wide":               {ListParams{VisibleProjects: true}, []string{"audit-failed", "audit-other-profile", "audit-draft"}},
		"owner-wide state set":     {ListParams{VisibleProjects: true, States: []AuditState{AuditDraft, AuditFailed}}, []string{"audit-failed", "audit-other-profile", "audit-draft"}},
		"owner-wide one state":     {ListParams{VisibleProjects: true, States: []AuditState{AuditFailed}}, []string{"audit-failed"}},
		"every Project and state":  {ListParams{}, []string{"audit-hidden", "audit-failed", "audit-other-profile", "audit-draft"}},
		"owner-wide missing state": {ListParams{VisibleProjects: true, States: []AuditState{AuditActive}}, []string{}},
	} {
		t.Run(name, func(t *testing.T) {
			if got := list(test.params); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("listed %v, want %v", got, test.want)
			}
		})
	}

	// Keyset pages of one row each walk the owner-wide list in order.
	var paged []string
	params := ListParams{OwnerID: owner, VisibleProjects: true, Limit: 1}
	for range 4 {
		page, err := store.List(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if len(page) == 0 {
			break
		}
		paged = append(paged, page[0].AuditID)
		params.BeforeCreatedAt, params.BeforeAuditID = &page[0].CreatedAt, page[0].AuditID
	}
	if want := []string{"audit-failed", "audit-other-profile", "audit-draft"}; !reflect.DeepEqual(paged, want) {
		t.Fatalf("pages %v, want %v", paged, want)
	}
	if _, err := store.List(ctx, ListParams{OwnerID: owner, States: []AuditState{"archived"}, Limit: 10}); err == nil {
		t.Fatal("an unknown state filter was accepted")
	}
}
