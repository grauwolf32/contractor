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
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestFindingComparisonTraversalsPreserveEveryObservationAndScopedFences(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project = "comparison-owner", "comparison-project"
	for _, audit := range []string{"comparison-a", "comparison-b", "comparison-c"} {
		createCorrelationAudit(t, ctx, pool, owner, project, audit)
	}
	intake, err := findingintake.New(pool)
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, findings: intake, now: time.Now}
	fp := correlationFingerprint(t)
	firstDocument := auditFindingDocument("comparison-first")
	firstDocument.Subject = nil
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, "comparison-a", "comparison-first", firstDocument, fp)
	if _, err := pool.Exec(ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence)
 SELECT receipt_id,'comparison-b',project_id,proposal_ref,evidence FROM finding_proposal_audit_holds
 WHERE audit_id='comparison-a' AND receipt_id='receipt-comparison-first'`); err != nil {
		t.Fatal(err)
	}
	third := seedAuditFindingDocument(t, ctx, pool, project, owner, "comparison-c", "comparison-third", auditFindingDocument("comparison-third"), fp)
	for _, v := range []struct{ audit, suffix, value string }{{"comparison-a", "comparison-left", "b"}, {"comparison-b", "comparison-right", "c"}} {
		other := fp
		other.Value = strings.Repeat(v.value, 64)
		seedAuditFindingDocument(t, ctx, pool, project, owner, v.audit, v.suffix, auditFindingDocument(v.suffix), other)
	}
	weakIDs := []string{}
	for _, v := range []struct{ audit, suffix string }{{"comparison-a", "comparison-weak-a"}, {"comparison-b", "comparison-weak-b"}} {
		document := auditFindingDocument(v.suffix)
		document.Subject = &auditdomain.FindingSubject{Kind: "component", Key: "same-component"}
		document.StandardRefs = []auditdomain.StandardReference{{Scheme: "CWE", Version: "4.20", RequirementID: "CWE-89"}}
		weakIDs = append(weakIDs, seedAuditFindingDocument(t, ctx, pool, project, owner, v.audit, v.suffix, document))
	}
	params := FindingComparisonParams{OwnerID: owner, ProjectID: project, LeftAuditID: "comparison-a", RightAuditID: "comparison-b", Limit: 1}
	initial, err := service.CompareFindings(ctx, params)
	if err != nil || len(initial.Items) != 1 || !initial.Page.HasMore || initial.Next == nil {
		t.Fatalf("first comparison=%+v %v", initial, err)
	}
	var total, shared, unmatched int
	seen := map[string]bool{}
	page := initial
	for pages := 0; ; pages++ {
		if pages > 10 {
			t.Fatal("comparison cursor did not terminate")
		}
		for _, row := range page.Items {
			if seen[row.RowKey] {
				t.Fatalf("repeated comparison row %s", row.RowKey)
			}
			seen[row.RowKey] = true
			total++
			if row.Kind == "both" {
				shared++
				if !row.PreviewsShareReceipt || row.Left.Origin.ReceiptID != row.Right.Origin.ReceiptID || row.Representative.AuditID == "comparison-c" {
					t.Fatalf("shared receipt/pair representative lost: %+v", row)
				}
			}
			if row.Kind == "unmatched" {
				unmatched++
			}
		}
		if !page.Page.HasMore {
			break
		}
		params.Cursor = page.Next
		page, err = service.CompareFindings(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
	}
	if total != 5 || shared != 1 || unmatched != 2 {
		t.Fatalf("comparison accounting: total=%d shared=%d unmatched=%d", total, shared, unmatched)
	}
	source, err := service.GetFinding(ctx, owner, "comparison-a", first)
	if err != nil {
		t.Fatal(err)
	}
	group := source.Correlation.GroupKey
	members := FindingMembersParams{OwnerID: owner, ProjectID: project, GroupKey: group, Limit: 1}
	projectPage, err := service.FindingGroupMembers(ctx, members)
	if err != nil || projectPage.Next == nil {
		t.Fatalf("member page=%+v %v", projectPage, err)
	}
	pairMembers := members
	pairMembers.AuditIDs = []string{"comparison-a", "comparison-b"}
	pairPage, err := service.FindingGroupMembers(ctx, pairMembers)
	if err != nil || pairPage.Next == nil {
		t.Fatalf("pair members=%+v %v", pairPage, err)
	}
	candidates := FindingCandidatesParams{OwnerID: owner, AuditID: "comparison-a", FindingID: first, Scope: "project", Limit: 1}
	candidatePage, err := service.FindingCandidates(ctx, candidates)
	if err != nil || len(candidatePage.Items) != 1 || candidatePage.Items[0].Tier != 1 || !candidatePage.Items[0].AutomaticallyCorrelated || candidatePage.Next == nil {
		t.Fatalf("automatic candidates=%+v %v", candidatePage, err)
	}
	family, err := service.FindingCandidates(ctx, FindingCandidatesParams{OwnerID: owner, AuditID: "comparison-a", FindingID: weakIDs[0], Scope: "project", Limit: 100})
	if err != nil || len(family.Items) != 1 || family.Items[0].Tier != 4 || family.Items[0].AutomaticallyCorrelated || family.Items[0].Finding.FindingID != weakIDs[1] {
		t.Fatalf("weak family=%+v %v", family, err)
	}
	// A third Audit's changed rating invalidates Project traversal, but neither
	// the selected pair's comparison nor its pair-scoped member traversal.
	decideFindingForTest(t, ctx, service, owner, "comparison-c", third, 1, "comparison-third-fp", auditstore.VerdictFalsePositive, nil, nil)
	members.Cursor = projectPage.Next
	if _, err := service.FindingGroupMembers(ctx, members); !errors.Is(err, auditstore.ErrCorrelationStale) {
		t.Fatalf("Project fence ignored third Audit: %v", err)
	}
	pairMembers.Cursor = pairPage.Next
	if _, err := service.FindingGroupMembers(ctx, pairMembers); err != nil {
		t.Fatalf("unrelated Audit invalidated pair members: %v", err)
	}
	params.Cursor = initial.Next
	if _, err := service.CompareFindings(ctx, params); err != nil {
		t.Fatalf("unrelated Audit invalidated comparison: %v", err)
	}
	candidates.Cursor = candidatePage.Next
	if _, err := service.FindingCandidates(ctx, candidates); !errors.Is(err, auditstore.ErrCorrelationStale) {
		t.Fatalf("candidate fence ignored rating: %v", err)
	}
	seedAuditFinding(t, ctx, pool, project, owner, "comparison-a", "comparison-later")
	if _, err := service.CompareFindings(ctx, params); !errors.Is(err, auditstore.ErrCorrelationStale) {
		t.Fatalf("selected Audit did not stale comparison: %v", err)
	}
	if _, err := service.CompareFindings(ctx, FindingComparisonParams{OwnerID: "foreign-owner", ProjectID: project, LeftAuditID: "comparison-a", RightAuditID: "comparison-b", Limit: 20}); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign comparison=%v", err)
	}
	candidates.Cursor = nil
	candidates.SourceReceiptID = "receipt-comparison-third"
	if _, err := service.FindingCandidates(ctx, candidates); !errors.Is(err, auditstore.ErrNotFound) {
		t.Fatalf("foreign contribution selected=%v", err)
	}
}

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
		if _, err := pool.Exec(ctx, "DELETE FROM audit_findings WHERE finding_id=$1", first); err != nil {
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

func TestFindingCandidateSelectedContributionDoesNotBridgePrimaryGroups(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, url)
	const owner, project = "contribution-search-owner", "contribution-search-project"
	for _, audit := range []string{"contribution-search-a", "contribution-search-b"} {
		createCorrelationAudit(t, ctx, pool, owner, project, audit)
	}
	fp := correlationFingerprint(t)
	other := fp
	other.Value = strings.Repeat("b", 64)
	first := seedAuditFindingDocument(t, ctx, pool, project, owner, "contribution-search-a", "contribution-primary", auditFindingDocument("contribution-primary"), fp)
	supporting := seedAuditFindingDocument(t, ctx, pool, project, owner, "contribution-search-a", "contribution-supporting", auditFindingDocument("contribution-supporting"), other)
	target := seedAuditFindingDocument(t, ctx, pool, project, owner, "contribution-search-b", "contribution-target", auditFindingDocument("contribution-target"), other)
	if _, err := pool.Exec(ctx, "DELETE FROM audit_findings WHERE finding_id=$1", supporting); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `INSERT INTO audit_finding_contributions(finding_id,audit_id,receipt_id,relation,proposal_ref)
 SELECT $1,'contribution-search-a',receipt_id,'contributing',proposal_ref FROM finding_proposal_audit_holds
 WHERE receipt_id='receipt-contribution-supporting' AND audit_id='contribution-search-a'`, first); err != nil {
		t.Fatal(err)
	}
	service := &Service{pool: pool, now: time.Now}
	params := FindingCandidatesParams{OwnerID: owner, AuditID: "contribution-search-a", FindingID: first, Scope: "project", SourceReceiptID: "receipt-contribution-supporting", Limit: 20}
	found := false
	for pages := 0; pages < 5; pages++ {
		page, err := service.FindingCandidates(ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		for _, candidate := range page.Items {
			if candidate.Finding.FindingID == target {
				if candidate.Tier != 2 || candidate.AutomaticallyCorrelated || candidate.Reasons[0].SourceReceiptID != params.SourceReceiptID {
					t.Fatalf("contribution silently regrouped: %+v", candidate)
				}
				found = true
			}
		}
		if !page.Page.HasMore {
			break
		}
		params.Cursor = page.Next
	}
	if !found {
		t.Fatal("selected supporting receipt was not searchable")
	}
	var sourceKey, targetKey string
	if err := pool.QueryRow(ctx, "SELECT correlation_group_key FROM audit_findings WHERE finding_id=$1", first).Scan(&sourceKey); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT correlation_group_key FROM audit_findings WHERE finding_id=$1", target).Scan(&targetKey); err != nil {
		t.Fatal(err)
	}
	if sourceKey == targetKey {
		t.Fatal("secondary contribution bridged primary groups")
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
