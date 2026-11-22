//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

func TestProcessDatabaseIsolation(t *testing.T) {
	base := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if base == "" {
		t.Fatal("CONTRACTOR_TEST_DATABASE_URL is required")
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	admin, err := pgx.Connect(ctx, base)
	if err != nil {
		t.Fatal(err)
	}
	defer admin.Close(context.Background())
	names := []string{}
	// The nested cleanup must finish before the parent verifies actual deletion.
	t.Run("owned-databases", func(t *testing.T) {
		for range 2 {
			parsed, err := url.Parse(base)
			if err != nil {
				t.Fatal(err)
			}
			query := parsed.Query()
			query.Set("search_path", "unrelated_schema")
			query.Set("application_name", "isolation-regression")
			parsed.RawQuery = query.Encode()
			dsn := isolatedDatabase(t, ctx, parsed.String())
			actual, err := url.Parse(dsn)
			if err != nil {
				t.Fatal(err)
			}
			if actual.Query().Has("search_path") || actual.Query().Get("application_name") != "isolation-regression" {
				t.Fatal("database options were not isolated correctly")
			}
			conn, err := pgx.Connect(ctx, dsn)
			if err != nil {
				t.Fatal(err)
			}
			var name string
			err = conn.QueryRow(ctx, `SELECT current_database()`).Scan(&name)
			_ = conn.Close(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			names = append(names, name)
		}
		if names[0] == names[1] {
			t.Fatal("independent process harnesses share a database-wide Server lease")
		}
	})
	for _, name := range names {
		var exists bool
		if err := admin.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=$1)`, name).Scan(&exists); err != nil || exists {
			t.Fatalf("owned database was not removed: exists=%t error=%v", exists, err)
		}
	}
}

// Both process groups must remain live together before either starts work.
// This exercises the actual database-wide lease without duplicating its key.
func TestIndependentFindingsServersAcrossProcesses(t *testing.T) {
	var ready atomic.Int32
	allReady := make(chan struct{})
	for _, name := range []string{"first", "second"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			h := startFindingsHarness(t, []domainGatewayStage{locatedFindingProducer()})
			if ready.Add(1) == 2 {
				close(allReady)
			}
			select {
			case <-allReady:
			case <-h.ctx.Done():
				t.Fatal("independent Server failed to become active")
			}
			project := createProjectResource(t, h.client, h.baseURL)
			source := uploadProjectScopeArtifact(t, h.client, h.baseURL, project.ProjectID, "sources", "concurrent-source", "application/zip", auditProgramZip(t, map[string][]byte{"app.py": []byte(findingsSource)}))
			body, err := json.Marshal(map[string]any{"workflow": "source-findings-review@3", "artifacts": map[string]artifactRef{"source": source}})
			if err != nil {
				t.Fatal(err)
			}
			runID := postProjectRun(t, h.client, h.baseURL, project.ProjectID, "concurrent-finding", body, false)
			waitForFindingsRun(t, h, runID, "succeeded")
			receipts := findingsReceipts(t, h, runID)
			if len(receipts) != 1 {
				t.Fatalf("Finding receipts=%d", len(receipts))
			}
			if err := checkLocatedProposal(receipts[0].Document); err != nil {
				t.Fatal(err)
			}
			if h.gateway.CompletedStages() != 1 || len(h.gateway.Failures()) != 0 {
				t.Fatalf("gateway did not complete: %v", h.gateway.Failures())
			}
		})
	}
}
