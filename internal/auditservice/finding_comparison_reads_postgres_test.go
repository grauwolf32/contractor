package auditservice

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A source read may precede the Project/Audit metadata lookup. Capture its
// revisions in the same source statement so deletion in that gap is stale.
func TestFindingCandidateSourceDeletionBetweenSourceAndFenceIsStale(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project, audit = "source-fence-owner", "source-fence-project", "source-fence-audit"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "source-fence-first", auditFindingDocument("source-fence-first"), correlationFingerprint(t))
	trace := &candidateSourceInterleavingTrace{after: func() {
		if _, err := pool.Exec(ctx, "DELETE FROM audits WHERE audit_id=$1", audit); err != nil {
			t.Fatal(err)
		}
	}}
	config := pool.Config()
	config.MaxConns, config.MinConns = 1, 0
	config.ConnConfig.Tracer = trace
	reader, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	service := &Service{pool: reader, now: time.Now}
	if _, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: audit, FindingID: first, Scope: "project", Limit: 20}); !errors.Is(err, auditstore.ErrCorrelationStale) {
		t.Fatalf("deleted source escaped fence: %v", err)
	}
	if !trace.fired {
		t.Fatal("source interleaving did not run")
	}
}

type candidateSourceInterleavingTrace struct {
	after func()
	fired bool
}
type candidateSourceQueryKey struct{}

func (q *candidateSourceInterleavingTrace) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	return context.WithValue(ctx, candidateSourceQueryKey{}, strings.Contains(data.SQL, "coalesce($4::text,finding.first_receipt_id)"))
}
func (q *candidateSourceInterleavingTrace) TraceQueryEnd(ctx context.Context, _ *pgx.Conn, _ pgx.TraceQueryEndData) {
	match, _ := ctx.Value(candidateSourceQueryKey{}).(bool)
	if match && !q.fired {
		q.fired = true
		q.after()
	}
}

func TestSharedFindingCandidateStructuralSitesRemainPossibleAcrossSnapshots(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	var fixture struct {
		Cases []struct {
			Name, File, ZipBase64 string
			Line                  int64
			Derived               json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-anchor-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	const owner, project = "structural-search-owner", "structural-search-project"
	sourceID := ""
	targetID := ""
	for i, audit := range []string{"structural-search-a", "structural-search-b"} {
		createCorrelationAudit(t, ctx, pool, owner, project, audit)
		selected := false
		for _, c := range fixture.Cases {
			if c.Name != "python-site" {
				continue
			}
			selected = true
			basis, err := auditdomain.DecodeFindingAnchorDerivation(c.Derived)
			if err != nil {
				t.Fatal(err)
			}
			data, err := base64.StdEncoding.DecodeString(c.ZipBase64)
			if err != nil {
				t.Fatal(err)
			}
			source := contracts.ExactContent(*basis.SourceRef, "application/zip", data)
			proposal := auditFindingDocument(audit)
			proposal.Subject = nil
			proposal.Locations = []auditdomain.FindingLocation{{File: c.File, Line: &c.Line}}
			commit := strings.Repeat("a", 40)
			if i == 1 {
				commit = strings.Repeat("b", 40)
			}
			anchors, err := auditdomain.ResolveFindingAnchors(basis, "https://example.test:443/repo.git", commit, source, proposal)
			if err != nil {
				t.Fatal(err)
			}
			id := seedAuditFindingProjected(t, ctx, pool, project, owner, audit, audit, proposal, auditdomain.UnavailableFindingFingerprint("not_provided"), anchors)
			if i == 0 {
				sourceID = id
			} else {
				targetID = id
			}
		}
		if !selected {
			t.Fatal("no structural fixture")
		}
	}
	service := &Service{pool: pool, now: time.Now}
	page, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: "structural-search-a", FindingID: sourceID, Scope: "project", Limit: 20})
	if err != nil || len(page.Items) != 1 || page.Items[0].Tier != 3 || page.Items[0].AutomaticallyCorrelated || page.Items[0].Finding.FindingID != targetID || len(page.Items[0].Reasons[0].Limitations) == 0 {
		t.Fatalf("structural sites became exact issues: %+v %v", page, err)
	}
	comparison, err := service.CompareFindings(ctx, FindingComparisonParams{OwnerID: owner, ProjectID: project, LeftAuditID: "structural-search-a", RightAuditID: "structural-search-b", Limit: 20})
	if err != nil || len(comparison.Items) != 2 || comparison.Items[0].Kind != "unmatched" || comparison.Items[1].Kind != "unmatched" {
		t.Fatalf("weak sites became an intersection: %+v %v", comparison, err)
	}
}

func TestFindingCandidateByteBoundKeepsEveryOmittedRowReachable(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project, audit = "byte-page-owner", "byte-page-project", "byte-page-audit"
	createCorrelationAudit(t, ctx, pool, owner, project, audit)
	document := auditFindingDocument("byte-source")
	document.Title = strings.Repeat("<", 300)
	document.Subject = &auditdomain.FindingSubject{Kind: "component", Key: "same-byte-subject"}
	for n := range 8 {
		document.StandardRefs = append(document.StandardRefs, auditdomain.StandardReference{Scheme: strings.Repeat("s", 160), Version: strings.Repeat("\x01", 160), RequirementID: strings.Repeat("r", 159) + fmt.Sprint(n)})
	}
	source := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, "byte-source", document)
	expected := map[string]bool{}
	for n := range 60 {
		suffix := fmt.Sprintf("byte-target-%03d-%s", n, strings.Repeat("x", 100))
		document.ClientKey = "candidate-" + suffix
		id := seedAuditFindingDocument(t, ctx, pool, project, owner, audit, suffix, document)
		expected[id] = true
	}
	service := &Service{pool: pool, now: time.Now}
	p := FindingCandidatesParams{OwnerID: owner, AuditID: audit, FindingID: source, Scope: "project", Limit: 100}
	seen := map[string]bool{}
	pages := 0
	for {
		page, err := service.FindingCandidates(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(page)
		if err != nil || len(raw)+FindingCorrelationMaximumCursorBytes+128 > auditstore.FindingComparisonMaximumBytes {
			t.Fatalf("page exceeds byte budget: %d %v", len(raw), err)
		}
		if pages == 0 && (!page.Page.HasMore || len(page.Items) >= 60) {
			t.Fatalf("fixture failed to exercise a byte boundary: %+v", page.Page)
		}
		for _, candidate := range page.Items {
			if seen[candidate.Finding.FindingID] || !expected[candidate.Finding.FindingID] {
				t.Fatal("byte traversal repeated/invented observation")
			}
			seen[candidate.Finding.FindingID] = true
			if !candidate.Finding.TitleTruncated || candidate.AutomaticallyCorrelated {
				t.Fatalf("preview/truth changed: %+v", candidate)
			}
		}
		pages++
		if !page.Page.HasMore {
			break
		}
		if pages > 10 || page.Next == nil {
			t.Fatal("byte cursor made no progress")
		}
		p.Cursor = page.Next
	}
	if len(seen) != len(expected) {
		t.Fatalf("byte trimming hid observations: got %d want %d", len(seen), len(expected))
	}
}
