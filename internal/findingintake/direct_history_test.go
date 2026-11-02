package findingintake

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/jackc/pgx/v5/pgxpool"
)

type directHistoryFactForTest struct {
	Schema    string `json:"schema"`
	ReceiptID string `json:"receiptId"`
	Kind      string `json:"kind"`
	Original  struct {
		AuditID  string `json:"auditId"`
		RecordID string `json:"recordId"`
		Semantic string `json:"semanticAssessment"`
	} `json:"original"`
	Materials struct {
		Result   ExactArtifact `json:"result"`
		Contract ExactArtifact `json:"contract"`
	} `json:"materials"`
}

// Read the production source-history API and exact ordinary Artifact bytes;
// there is no table of source assessments or special historical reader.
func readDirectHistoryForTest(t *testing.T, ctx context.Context, pool *pgxpool.Pool, owner, project, receipt, audit string) []directHistoryFactForTest {
	t.Helper()
	intake, err := New(pool)
	if err != nil {
		t.Fatal(err)
	}
	store, err := artifacts.NewService(artifactpostgres.NewRepository(pool)).Project(project)
	if err != nil {
		t.Fatal(err)
	}
	var records []directHistoryFactForTest
	after := ""
	var revision *uint64
	for pass := 0; ; pass++ {
		if pass > 4 {
			t.Fatal("unexpected history cardinality in functional fixture")
		}
		page, err := intake.ListProjectSourceHistory(ctx, owner, project, receipt, after, 64, revision)
		if err != nil {
			t.Fatal(err)
		}
		revision = &page.ProjectRevision
		for _, metadata := range page.Items {
			if !strings.Contains(metadata.Ref.Name, ".direct-verification.") {
				continue
			}
			read, err := store.Read(ctx, metadata.Ref)
			if err != nil {
				t.Fatal(err)
			}
			var fact directHistoryFactForTest
			if err := json.Unmarshal(read.Payload.Data, &fact); err != nil || fact.Schema != auditdomain.FindingSourceHistorySchema || fact.ReceiptID != receipt || fact.Kind != "direct-verification" {
				t.Fatalf("invalid current history: %+v %v", fact, err)
			}
			if audit == "" || fact.Original.AuditID == audit {
				records = append(records, fact)
			}
		}
		if len(page.Items) < 64 {
			break
		}
		after = page.Items[len(page.Items)-1].Ref.Name
	}
	return records
}
