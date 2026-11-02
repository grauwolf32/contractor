package evalservice

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// pageStatementTracer counts the statements a traced pool sends and, among
// them, the comparison page statements.
type pageStatementTracer struct {
	mu                 sync.Mutex
	statements, cohort int
}

func (p *pageStatementTracer) TraceQueryStart(ctx context.Context, _ *pgx.Conn, data pgx.TraceQueryStartData) context.Context {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.statements++
	if strings.Contains(data.SQL, "WITH filtered AS MATERIALIZED") {
		p.cohort++
	}
	return ctx
}
func (*pageStatementTracer) TraceQueryEnd(context.Context, *pgx.Conn, pgx.TraceQueryEndData) {}

func (p *pageStatementTracer) take() (statements, cohort int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	statements, cohort = p.statements, p.cohort
	p.statements, p.cohort = 0, 0
	return statements, cohort
}

// At the admitted native member ceiling, every page of the published view's
// 500 pairs and 1,000 members, the empty page after the last cursor included,
// reads its exact filtered count and its items in one comparison statement
// (EVM-R245), and the member pages return exactly the complete member views
// that the pair pages carry.
func TestSharedPostgresSelectedPagesAtNativeCeilingReadOneStatementPerPage(t *testing.T) {
	h := newHarness(t)
	ctx := t.Context()
	e := h.preparedAtNativeCeiling(t, 1)
	claims, err := evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).Claim(ctx, "collector", time.Minute, 10)
	if err != nil {
		t.Fatal(err)
	}
	var claim evalstore.Claim
	for _, candidate := range claims {
		if candidate.ExperimentID == e.ID {
			claim = candidate
		}
	}
	if claim.ExperimentID == "" {
		t.Fatal("the prepared experiment was not claimable for collection")
	}
	var view *evalstore.View
	for range evaldomain.MaxNativeMembers/evaldomain.CollectionBatchSize + 1 {
		if err = h.service.CollectView(ctx, e, claim); err != nil {
			t.Fatal(err)
		}
		if view, err = evalstore.NewPostgresStore(h.pool, artifactpgtest.NewStorage()).LatestView(ctx, e.OwnerID, e.ID); err != nil {
			t.Fatal(err)
		}
		if view != nil {
			break
		}
	}
	if view == nil {
		t.Fatal("collection published no view")
	}

	trace := &pageStatementTracer{}
	config := h.pool.Config()
	config.ConnConfig.Tracer = trace
	traced, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	defer traced.Close()
	original := h.service.pool
	h.service.pool = traced
	defer func() { h.service.pool = original }()
	onePage := func(name string, statements, cohort int) {
		t.Helper()
		if cohort != 1 {
			t.Fatalf("%s sent %d comparison statements among %d, want 1", name, cohort, statements)
		}
	}

	sides := map[string][]byte{}
	pageParams := MemberPageParams{OwnerID: e.OwnerID, ExperimentID: e.ID, AfterOrdinal: -1, Limit: evaldomain.MaxPageSize}
	pairPages, statementsPerPage := 0, 0
	for p := (PairPageParams{MemberPageParams: pageParams}); ; pairPages++ {
		page, err := h.service.Pairs(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		statements, cohort := trace.take()
		onePage("pair page", statements, cohort)
		statementsPerPage = statements
		if page.FilteredCount != evaldomain.MaxNativeMembers/2 || page.Snapshot != view.Snapshot {
			t.Fatalf("pair page counted %d in %s", page.FilteredCount, page.Snapshot)
		}
		for _, pair := range page.Items {
			for _, side := range []evaldomain.MemberView{pair.A, pair.B} {
				if _, seen := sides[side.Member.ID]; seen {
					t.Fatalf("member %s repeated", side.Member.ID)
				}
				encoded, err := json.Marshal(side)
				if err != nil {
					t.Fatal(err)
				}
				sides[side.Member.ID] = encoded
			}
		}
		if len(page.Items) == 0 {
			if page.HasMore || page.LastOrdinal != -1 {
				t.Fatalf("empty pair page %+v", page.SelectedPage)
			}
			break
		}
		p.AfterOrdinal = page.LastOrdinal
	}
	if pairPages != 5 || len(sides) != evaldomain.MaxNativeMembers {
		t.Fatalf("%d pair pages carried %d members", pairPages, len(sides))
	}
	memberPages := 0
	for _, filter := range []string{"", "unresolved"} {
		p := pageParams
		p.Filter = filter
		for seen := 0; ; memberPages++ {
			page, err := h.service.Members(ctx, p)
			if err != nil {
				t.Fatal(err)
			}
			statements, cohort := trace.take()
			onePage("member page", statements, cohort)
			if page.FilteredCount != evaldomain.MaxNativeMembers {
				t.Fatalf("member page filter %q counted %d", filter, page.FilteredCount)
			}
			for _, member := range page.Items {
				encoded, err := json.Marshal(member)
				if err != nil {
					t.Fatal(err)
				}
				if string(encoded) != string(sides[member.Member.ID]) {
					t.Fatalf("member page item %s differs from its pair side", member.Member.ID)
				}
			}
			seen += len(page.Items)
			if len(page.Items) == 0 {
				if seen != evaldomain.MaxNativeMembers || page.HasMore {
					t.Fatalf("member pages with filter %q returned %d members", filter, seen)
				}
				break
			}
			p.AfterOrdinal = page.LastOrdinal
		}
	}
	t.Logf("%d native members: %d pair and %d member pages, each one comparison statement among %d statements of its read",
		evaldomain.MaxNativeMembers, pairPages+1, memberPages+2, statementsPerPage)
}
