package auditpriority_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	ap "github.com/grauwolf32/contractor/internal/auditpriority"
)

func selectionDigestFixture(character string) string {
	return "sha256:" + strings.Repeat(character, 64)
}

func selectionItemsFixture(count int) []ap.ItemIdentity {
	items := make([]ap.ItemIdentity, count)
	for i := range items {
		items[i] = ap.ItemIdentity{Key: fmt.Sprintf("CHECK-%04d", i), Version: "1"}
	}
	return items
}

func selectionFixture(t *testing.T, count, topN int) (ap.Pool, ap.CycleBinding, []ap.BoundVerdict) {
	t.Helper()
	pool, err := ap.NewPool(selectionDigestFixture("a"), selectionItemsFixture(count))
	if err != nil {
		t.Fatal(err)
	}
	digest, err := ap.PoolDigest(pool)
	if err != nil {
		t.Fatal(err)
	}
	cycle := ap.CycleBinding{
		CycleID: "cycle-1", InventoryDigest: pool.InventoryDigest, PoolDigest: digest,
		ContextDigest: selectionDigestFixture("b"), PolicyDigest: selectionDigestFixture("c"),
		PromptDigest: selectionDigestFixture("d"), ModelConfigDigest: selectionDigestFixture("e"), TopN: topN,
	}
	verdicts := make([]ap.BoundVerdict, len(pool.Candidates))
	for i, candidate := range pool.Candidates {
		verdict := ap.Verdict{
			ItemKey: candidate.ItemKey, Priority: ap.PriorityMedium, Confidence: ap.ConfidenceLow,
			Rationale:   "General checklist coverage; no stronger contextual signal is available.",
			EvidenceIDs: []string{}, MissingContext: []string{"Service description is absent."},
		}
		verdicts[i], err = ap.BindVerdict(cycle, candidate, verdict, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	return pool, cycle, verdicts
}

func requirePriorityError(t *testing.T, err error, code string) {
	t.Helper()
	if err == nil {
		t.Fatal("accepted invalid priority input")
	}
	var diagnostic *ap.Error
	if !errors.As(err, &diagnostic) {
		t.Fatalf("untyped diagnostic: %T: %v", err, err)
	}
	if code != "" && diagnostic.Code != code {
		t.Fatalf("diagnostic code = %q, want %q", diagnostic.Code, code)
	}
}

func TestResolveTopNPolicyAndOverride(t *testing.T) {
	integer := func(value int) *int { return &value }
	tests := []struct {
		name     string
		policy   ap.Policy
		override *int
		want     int
	}{
		{name: "omitted defaults", want: 10},
		{name: "explicit default", policy: ap.Policy{DefaultTopN: integer(12)}, want: 12},
		{name: "profile ceiling", policy: ap.Policy{MaxTopN: integer(12)}, want: 10},
		{name: "explicit override", override: integer(12), want: 12},
		{name: "lower override", policy: ap.Policy{DefaultTopN: integer(12)}, override: integer(10), want: 10},
		{name: "maximum", override: integer(1000), want: 1000},
		{name: "override at profile ceiling", policy: ap.Policy{MaxTopN: integer(12)}, override: integer(12), want: 12},
		{name: "zero default", policy: ap.Policy{DefaultTopN: integer(0)}},
		{name: "small default", policy: ap.Policy{DefaultTopN: integer(9)}},
		{name: "large default", policy: ap.Policy{DefaultTopN: integer(1001)}},
		{name: "zero ceiling", policy: ap.Policy{MaxTopN: integer(0)}},
		{name: "small ceiling", policy: ap.Policy{MaxTopN: integer(9)}},
		{name: "large ceiling", policy: ap.Policy{MaxTopN: integer(1001)}},
		{name: "default exceeds ceiling", policy: ap.Policy{DefaultTopN: integer(12), MaxTopN: integer(10)}},
		{name: "override cannot repair invalid profile", policy: ap.Policy{DefaultTopN: integer(12), MaxTopN: integer(10)}, override: integer(10)},
		{name: "negative override", override: integer(-1)},
		{name: "zero override", override: integer(0)},
		{name: "small override", override: integer(9)},
		{name: "large override", override: integer(1001)},
		{name: "override exceeds profile ceiling", policy: ap.Policy{MaxTopN: integer(12)}, override: integer(13)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ap.ResolveTopN(test.policy, test.override)
			if test.want == 0 {
				requirePriorityError(t, err, ap.CodeInvalidPolicy)
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("ResolveTopN = (%d, %v), want (%d, nil)", got, err, test.want)
			}
		})
	}
}

func TestSelectionExactTopNBoundaries(t *testing.T) {
	for _, count := range []int{0, 1, 3, 9, 10, 11, 100, 1000} {
		for _, topN := range []int{10, 12} {
			t.Run(fmt.Sprintf("remaining=%d/topN=%d", count, topN), func(t *testing.T) {
				pool, cycle, verdicts := selectionFixture(t, count, topN)
				selection, err := ap.Select(pool, cycle, verdicts, nil)
				if err != nil {
					t.Fatal(err)
				}
				wantSelected := min(count, topN)
				if selection.Schema != ap.SelectionSchema || selection.Cycle != cycle || selection.Rows == nil || len(selection.Rows) != count || selection.SelectedCount != wantSelected || selection.DeferredCount != count-wantSelected {
					t.Fatalf("unexpected selection shape/count: %+v", selection)
				}
				wantCode := ""
				if count == 0 {
					wantCode = ap.CodeNoRemainingCandidates
				} else if count < topN {
					wantCode = ap.CodeFewerCandidates
				}
				if selection.Code != wantCode {
					t.Fatalf("selection code = %q, want %q", selection.Code, wantCode)
				}
				for i, row := range selection.Rows {
					if row.Candidate != pool.Candidates[i] || row.Rank != i+1 || row.Selected != (i < wantSelected) {
						t.Fatalf("row %d lost identity, rank or disposition: %+v", i, row)
					}
					wantCode := ap.CodeDeferredTopN
					if row.Selected {
						wantCode = ""
					}
					if row.Code != wantCode || row.Verdict.Priority != ap.PriorityMedium {
						t.Fatalf("row %d changed priority instead of recording disposition: %+v", i, row)
					}
				}
				if err := selection.Validate(); err != nil {
					t.Fatalf("selector emitted invalid selection: %v", err)
				}
			})
		}
	}
	_, err := ap.NewPool(selectionDigestFixture("a"), selectionItemsFixture(1001))
	requirePriorityError(t, err, ap.CodeInvalidPool)
	pool, cycle, verdicts := selectionFixture(t, 1000, 1000)
	selection, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil || selection.SelectedCount != 1000 || selection.DeferredCount != 0 || selection.Code != "" {
		t.Fatalf("maximum topN did not select the full permitted inventory: %v", err)
	}
	if err := selection.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCandidateIdentityAndPoolCanonicalOrdering(t *testing.T) {
	items := selectionItemsFixture(11)
	pool, err := ap.NewPool(selectionDigestFixture("a"), items)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := ap.MarshalPool(pool)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := ap.PoolDigest(pool)
	if err != nil {
		t.Fatal(err)
	}
	for rotation := 0; rotation < len(items); rotation++ {
		permuted := append(slices.Clone(items[rotation:]), items[:rotation]...)
		slices.Reverse(permuted)
		other, err := ap.NewPool(selectionDigestFixture("a"), permuted)
		if err != nil {
			t.Fatal(err)
		}
		otherBytes, err := ap.MarshalPool(other)
		if err != nil {
			t.Fatal(err)
		}
		otherDigest, err := ap.PoolDigest(other)
		if err != nil || otherDigest != digest || !bytes.Equal(otherBytes, canonical) {
			t.Fatalf("input ordering changed canonical pool: %v", err)
		}
	}
	for i, candidate := range pool.Candidates {
		id, err := ap.CandidateID(pool.InventoryDigest, candidate.ItemKey, candidate.ItemVersion)
		if err != nil || id != candidate.ID {
			t.Fatalf("candidate identity cannot be reproduced: %v", err)
		}
		if i > 0 && pool.Candidates[i-1].ID >= candidate.ID {
			t.Fatal("candidate IDs are not in strict canonical order")
		}
	}
	baseID, err := ap.CandidateID(selectionDigestFixture("a"), "AUTHZ-07", "1")
	if err != nil {
		t.Fatal(err)
	}
	for _, identity := range [][3]string{
		{selectionDigestFixture("b"), "AUTHZ-07", "1"},
		{selectionDigestFixture("a"), "AUTHZ-08", "1"},
		{selectionDigestFixture("a"), "AUTHZ-07", "2"},
	} {
		id, err := ap.CandidateID(identity[0], identity[1], identity[2])
		if err != nil || id == baseID {
			t.Fatalf("identity failed to bind inventory, item key and version: %v", err)
		}
	}
	nilPool, err := ap.NewPool(selectionDigestFixture("a"), nil)
	if err != nil || nilPool.Candidates == nil {
		t.Fatalf("empty input was not normalized to a non-null collection: %v", err)
	}
	items[0].Key = "CHANGED"
	afterMutation, err := ap.MarshalPool(pool)
	if err != nil || !bytes.Equal(afterMutation, canonical) {
		t.Fatalf("pool retained mutable input storage: %v", err)
	}
}

func TestCandidateIdentityV1Golden(t *testing.T) {
	// Independently computed SHA-256 vectors over the spec's canonical ASCII
	// JSON protect the published v1 domain separation and field names.
	const wantID = "priority-candidate-067d273ba1f740aaa3075cb304aa6a250b3d7af865378b6e6ff148b72a4f80fb"
	id, err := ap.CandidateID(selectionDigestFixture("a"), "AUTHZ-07", "1")
	if err != nil || id != wantID {
		t.Fatalf("v1 candidate identity changed: %q, %v", id, err)
	}
	pool, err := ap.NewPool(selectionDigestFixture("a"), []ap.ItemIdentity{{Key: "AUTHZ-07", Version: "1"}})
	if err != nil {
		t.Fatal(err)
	}
	digest, err := ap.PoolDigest(pool)
	const wantDigest = "sha256:183b6fdcf4ad55a8ba41946e7678d47140e221aacaaadcf269e168432fa17790"
	if err != nil || digest != wantDigest {
		t.Fatalf("v1 pool contract changed: %q, %v", digest, err)
	}
}

func TestDeferredHighPriorityKeepsItsVerdict(t *testing.T) {
	pool, cycle, verdicts := selectionFixture(t, 11, 10)
	for i := range verdicts {
		verdicts[i].Verdict.Priority = ap.PriorityHigh
		verdicts[i].Verdict.Confidence = ap.ConfidenceLow
	}
	// Higher confidence must not displace the first ten canonical IDs or
	// change this high-priority candidate's verdict to justify deferral.
	want := verdicts[10].Verdict
	want.Confidence = ap.ConfidenceHigh
	verdicts[10].Verdict = want
	slices.Reverse(verdicts)
	selection, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	row := selection.Rows[10]
	if selection.SelectedCount != 10 || selection.DeferredCount != 1 || row.Rank != 11 || row.Selected ||
		row.Code != ap.CodeDeferredTopN || row.Candidate.ID != pool.Candidates[10].ID || !reflect.DeepEqual(row.Verdict, want) {
		t.Fatalf("high-priority deferral lost its actual verdict or cutoff: %+v", row)
	}
}

func TestPoolRejectsInvalidAndForgedMembership(t *testing.T) {
	for name, items := range map[string][]ap.ItemIdentity{
		"same identity":               {{Key: "AUTHZ-07", Version: "1"}, {Key: "AUTHZ-07", Version: "1"}},
		"same key different versions": {{Key: "AUTHZ-07", Version: "1"}, {Key: "AUTHZ-07", Version: "2"}},
		"empty key":                   {{Key: "", Version: "1"}},
		"empty version":               {{Key: "AUTHZ-07", Version: ""}},
		"overlong key":                {{Key: strings.Repeat("A", ap.MaxIdentifierBytes+1), Version: "1"}},
		"overlong version":            {{Key: "AUTHZ-07", Version: strings.Repeat("A", ap.MaxItemVersionBytes+1)}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := ap.NewPool(selectionDigestFixture("a"), items)
			requirePriorityError(t, err, ap.CodeInvalidPool)
		})
	}
	for _, invalidDigest := range []string{"", strings.Repeat("a", 64), "sha256:" + strings.Repeat("A", 64), selectionDigestFixture("g")} {
		_, err := ap.NewPool(invalidDigest, nil)
		requirePriorityError(t, err, ap.CodeInvalidPool)
	}
	mutations := map[string]func(*ap.Pool){
		"schema":                  func(p *ap.Pool) { p.Schema = "other-schema" },
		"null candidates":         func(p *ap.Pool) { p.Candidates = nil },
		"forged ID":               func(p *ap.Pool) { p.Candidates[0].ID = p.Candidates[1].ID },
		"changed item key":        func(p *ap.Pool) { p.Candidates[0].ItemKey = "FOREIGN" },
		"changed version":         func(p *ap.Pool) { p.Candidates[0].ItemVersion = "2" },
		"changed inventory":       func(p *ap.Pool) { p.InventoryDigest = selectionDigestFixture("f") },
		"noncanonical membership": func(p *ap.Pool) { slices.Reverse(p.Candidates) },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			pool, cycle, verdicts := selectionFixture(t, 11, 10)
			mutate(&pool)
			_, err := ap.MarshalPool(pool)
			requirePriorityError(t, err, ap.CodeInvalidPool)
			_, err = ap.PoolDigest(pool)
			requirePriorityError(t, err, ap.CodeInvalidPool)
			_, err = ap.Select(pool, cycle, verdicts, nil)
			requirePriorityError(t, err, "")
		})
	}
}

func TestSelectOrdersPriorityThenIDWithoutConfidenceBias(t *testing.T) {
	pool, cycle, verdicts := selectionFixture(t, 12, 10)
	priorities := []ap.Priority{ap.PriorityLow, ap.PriorityHigh, ap.PriorityMedium, ap.PriorityCritical}
	confidences := []ap.Confidence{ap.ConfidenceLow, ap.ConfidenceMedium, ap.ConfidenceHigh}
	for i := range verdicts {
		verdicts[i].Verdict.Priority = priorities[i%len(priorities)]
		verdicts[i].Verdict.Confidence = confidences[i%len(confidences)]
	}
	selection, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantIndices := []int{3, 7, 11, 1, 5, 9, 2, 6, 10, 0, 4, 8}
	for i, sourceIndex := range wantIndices {
		if selection.Rows[i].Candidate.ID != pool.Candidates[sourceIndex].ID || !reflect.DeepEqual(selection.Rows[i].Verdict, verdicts[sourceIndex].Verdict) {
			t.Fatalf("rank %d does not follow priority then ID or lost the original verdict", i+1)
		}
	}
	canonical, err := ap.MarshalSelection(selection)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := ap.SelectionDigest(selection)
	if err != nil {
		t.Fatal(err)
	}
	for rotation := 0; rotation < len(verdicts); rotation++ {
		permuted := append(slices.Clone(verdicts[rotation:]), verdicts[:rotation]...)
		slices.Reverse(permuted)
		other, err := ap.Select(pool, cycle, permuted, nil)
		if err != nil {
			t.Fatal(err)
		}
		otherBytes, err := ap.MarshalSelection(other)
		if err != nil {
			t.Fatal(err)
		}
		otherDigest, err := ap.SelectionDigest(other)
		if err != nil || otherDigest != digest || !bytes.Equal(canonical, otherBytes) {
			t.Fatalf("verdict arrival ordering changed selection: %v", err)
		}
	}
	for i := range verdicts {
		verdicts[i].Verdict.Confidence = ap.ConfidenceHigh
	}
	other, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range other.Rows {
		if other.Rows[i].Candidate.ID != selection.Rows[i].Candidate.ID || other.Rows[i].Selected != selection.Rows[i].Selected {
			t.Fatal("confidence changed selection membership or rank")
		}
	}
	var roundTrip ap.Selection
	if err := json.Unmarshal(canonical, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if err := roundTrip.Validate(); err != nil {
		t.Fatalf("canonical selection cannot be decoded and validated: %v", err)
	}
}

func TestSelectionRequiresCompleteSameCycleVerdicts(t *testing.T) {
	mutations := map[string]func(*ap.Pool, *ap.CycleBinding, *[]ap.BoundVerdict){
		"missing verdict":   func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { *v = (*v)[:10] },
		"no verdicts":       func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { *v = nil },
		"duplicate verdict": func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { (*v)[10] = (*v)[0] },
		"extra verdict":     func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { *v = append(*v, (*v)[0]) },
		"unknown candidate": func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) {
			(*v)[0].CandidateID = "priority-candidate-" + strings.Repeat("0", 64)
		},
		"wrong item key": func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { (*v)[0].Verdict.ItemKey = "FOREIGN" },
		"invented evidence": func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) {
			(*v)[0].Verdict.EvidenceIDs = []string{"unavailable-finding"}
		},
		"null evidence":        func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { (*v)[0].Verdict.EvidenceIDs = nil },
		"null missing context": func(_ *ap.Pool, _ *ap.CycleBinding, v *[]ap.BoundVerdict) { (*v)[0].Verdict.MissingContext = nil },
		"cycle pool mismatch":  func(_ *ap.Pool, c *ap.CycleBinding, _ *[]ap.BoundVerdict) { c.PoolDigest = selectionDigestFixture("f") },
		"cycle inventory mismatch": func(_ *ap.Pool, c *ap.CycleBinding, _ *[]ap.BoundVerdict) {
			c.InventoryDigest = selectionDigestFixture("f")
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			pool, cycle, verdicts := selectionFixture(t, 11, 10)
			mutate(&pool, &cycle, &verdicts)
			selection, err := ap.Select(pool, cycle, verdicts, nil)
			requirePriorityError(t, err, "")
			if selection.SelectedCount != 0 || len(selection.Rows) != 0 {
				t.Fatal("failed ranking returned a partial executable selection")
			}
		})
	}
	for name, mutate := range cycleBindingMutations() {
		t.Run("foreign verdict binding/"+name, func(t *testing.T) {
			pool, cycle, verdicts := selectionFixture(t, 11, 10)
			mutate(&verdicts[0].Cycle)
			_, err := ap.Select(pool, cycle, verdicts, nil)
			requirePriorityError(t, err, "")
		})
	}
}

func cycleBindingMutations() map[string]func(*ap.CycleBinding) {
	return map[string]func(*ap.CycleBinding){
		"cycle ID":    func(c *ap.CycleBinding) { c.CycleID = "cycle-2" },
		"inventory":   func(c *ap.CycleBinding) { c.InventoryDigest = selectionDigestFixture("f") },
		"pool":        func(c *ap.CycleBinding) { c.PoolDigest = selectionDigestFixture("f") },
		"context":     func(c *ap.CycleBinding) { c.ContextDigest = selectionDigestFixture("f") },
		"policy":      func(c *ap.CycleBinding) { c.PolicyDigest = selectionDigestFixture("f") },
		"prompt":      func(c *ap.CycleBinding) { c.PromptDigest = selectionDigestFixture("f") },
		"model route": func(c *ap.CycleBinding) { c.ModelConfigDigest = selectionDigestFixture("f") },
		"top N":       func(c *ap.CycleBinding) { c.TopN = 12 },
	}
}

func TestCycleBindingRejectsMissingAndMalformedPins(t *testing.T) {
	_, base, _ := selectionFixture(t, 1, 10)
	for _, topN := range []int{0, 9, 1001} {
		cycle := base
		cycle.TopN = topN
		requirePriorityError(t, cycle.Validate(), ap.CodeInvalidBinding)
	}
	cycle := base
	cycle.CycleID = ""
	requirePriorityError(t, cycle.Validate(), ap.CodeInvalidBinding)
	for _, field := range []string{"InventoryDigest", "PoolDigest", "ContextDigest", "PolicyDigest", "PromptDigest", "ModelConfigDigest"} {
		for _, bad := range []string{"", "sha256:bad", "sha256:" + strings.Repeat("A", 64)} {
			t.Run(field+"/"+bad, func(t *testing.T) {
				cycle := base
				reflect.ValueOf(&cycle).Elem().FieldByName(field).SetString(bad)
				requirePriorityError(t, cycle.Validate(), ap.CodeInvalidBinding)
			})
		}
	}
}

func TestBindVerdictChecksIdentityAndEvidenceScope(t *testing.T) {
	pool, cycle, verdicts := selectionFixture(t, 2, 10)
	verdict := verdicts[0].Verdict
	verdict.EvidenceIDs = []string{"finding-3"}
	bound, err := ap.BindVerdict(cycle, pool.Candidates[0], verdict, []string{"finding-3"})
	if err != nil || bound.Cycle != cycle || bound.CandidateID != pool.Candidates[0].ID {
		t.Fatalf("exact identity/evidence binding failed: %v", err)
	}
	_, err = ap.BindVerdict(cycle, pool.Candidates[0], verdict, []string{"finding-4"})
	requirePriorityError(t, err, "")
	_, err = ap.BindVerdict(cycle, pool.Candidates[1], verdict, []string{"finding-3"})
	requirePriorityError(t, err, "")
	forged := pool.Candidates[0]
	forged.ItemVersion = "2"
	_, err = ap.BindVerdict(cycle, forged, verdict, []string{"finding-3"})
	requirePriorityError(t, err, "")
	verdict.EvidenceIDs[0] = "MUTATED"
	verdict.MissingContext[0] = "MUTATED"
	if bound.Verdict.EvidenceIDs[0] != "finding-3" || bound.Verdict.MissingContext[0] == "MUTATED" {
		t.Fatal("bound verdict retained mutable model input slices")
	}
}

func TestSelectionValidationRejectsForgedDispositionAndIdentity(t *testing.T) {
	mutations := map[string]func(*ap.Selection){
		"schema":                func(s *ap.Selection) { s.Schema = "other-schema" },
		"null rows":             func(s *ap.Selection) { s.Rows = nil },
		"missing row":           func(s *ap.Selection) { s.Rows = s.Rows[:10] },
		"duplicate row":         func(s *ap.Selection) { s.Rows[10] = s.Rows[0] },
		"rank":                  func(s *ap.Selection) { s.Rows[0].Rank = 2 },
		"missing selected item": func(s *ap.Selection) { s.Rows[0].Selected = false },
		"extra selected item":   func(s *ap.Selection) { s.Rows[10].Selected = true },
		"selected count":        func(s *ap.Selection) { s.SelectedCount = 9 },
		"deferred count":        func(s *ap.Selection) { s.DeferredCount = 0 },
		"selected code":         func(s *ap.Selection) { s.Rows[0].Code = ap.CodeDeferredTopN },
		"deferred code":         func(s *ap.Selection) { s.Rows[10].Code = "" },
		"overall code":          func(s *ap.Selection) { s.Code = ap.CodeFewerCandidates },
		"candidate ID":          func(s *ap.Selection) { s.Rows[0].Candidate.ID = s.Rows[1].Candidate.ID },
		"candidate version":     func(s *ap.Selection) { s.Rows[0].Candidate.ItemVersion = "2" },
		"verdict item key":      func(s *ap.Selection) { s.Rows[0].Verdict.ItemKey = "FOREIGN" },
		"pool digest":           func(s *ap.Selection) { s.Cycle.PoolDigest = selectionDigestFixture("f") },
		"inventory digest":      func(s *ap.Selection) { s.Cycle.InventoryDigest = selectionDigestFixture("f") },
		"priority order":        func(s *ap.Selection) { s.Rows[10].Verdict.Priority = ap.PriorityCritical },
		"same priority ID order": func(s *ap.Selection) {
			s.Rows[0], s.Rows[1] = s.Rows[1], s.Rows[0]
			s.Rows[0].Rank, s.Rows[1].Rank = 1, 2
		},
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			pool, cycle, verdicts := selectionFixture(t, 11, 10)
			selection, err := ap.Select(pool, cycle, verdicts, nil)
			if err != nil {
				t.Fatal(err)
			}
			mutate(&selection)
			requirePriorityError(t, selection.Validate(), "")
			_, err = ap.MarshalSelection(selection)
			requirePriorityError(t, err, "")
			_, err = ap.SelectionDigest(selection)
			requirePriorityError(t, err, "")
		})
	}
}

func TestSelectionDigestCoversVerdictsAndSemanticCycle(t *testing.T) {
	pool, cycle, verdicts := selectionFixture(t, 11, 10)
	base, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	baseDigest, err := ap.SelectionDigest(base)
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ap.Selection){
		"selected rationale": func(s *ap.Selection) { s.Rows[0].Verdict.Rationale = "New rationale." },
		"deferred rationale": func(s *ap.Selection) { s.Rows[10].Verdict.Rationale = "Deferred rationale remains recorded." },
		"confidence":         func(s *ap.Selection) { s.Rows[10].Verdict.Confidence = ap.ConfidenceHigh },
		"missing context":    func(s *ap.Selection) { s.Rows[10].Verdict.MissingContext = []string{"No architecture supplied."} },
		"evidence reference": func(s *ap.Selection) { s.Rows[10].Verdict.EvidenceIDs = []string{"finding-3"} },
		"cycle ID":           func(s *ap.Selection) { s.Cycle.CycleID = "cycle-2" },
		"context":            func(s *ap.Selection) { s.Cycle.ContextDigest = selectionDigestFixture("f") },
		"policy":             func(s *ap.Selection) { s.Cycle.PolicyDigest = selectionDigestFixture("f") },
		"prompt":             func(s *ap.Selection) { s.Cycle.PromptDigest = selectionDigestFixture("f") },
		"model route":        func(s *ap.Selection) { s.Cycle.ModelConfigDigest = selectionDigestFixture("f") },
	} {
		t.Run(name, func(t *testing.T) {
			selection, err := ap.Select(pool, cycle, verdicts, nil)
			if err != nil {
				t.Fatal(err)
			}
			mutate(&selection)
			digest, err := ap.SelectionDigest(selection)
			if err != nil || digest == baseDigest {
				t.Fatalf("semantic selection change omitted from digest: %v", err)
			}
		})
	}
	cycle.TopN = 12
	for i := range verdicts {
		verdicts[i].Cycle = cycle
	}
	other, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	otherDigest, err := ap.SelectionDigest(other)
	if err != nil || otherDigest == baseDigest {
		t.Fatalf("effective topN omitted from selection digest: %v", err)
	}
}

func TestSelectionDetachesInputsAndAllowsConcurrentReaders(t *testing.T) {
	pool, cycle, verdicts := selectionFixture(t, 11, 10)
	for i := range verdicts {
		verdicts[i].Verdict.EvidenceIDs = []string{"finding-3"}
	}
	selection, err := ap.Select(pool, cycle, verdicts, []string{"finding-3"})
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := ap.MarshalSelection(selection)
	if err != nil {
		t.Fatal(err)
	}
	verdicts[0].Verdict.EvidenceIDs[0] = "MUTATED"
	verdicts[0].Verdict.MissingContext[0] = "MUTATED"
	pool.Candidates[0].ItemKey = "MUTATED"
	afterMutation, err := ap.MarshalSelection(selection)
	if err != nil || !bytes.Equal(afterMutation, canonical) {
		t.Fatalf("selection shares mutable input storage: %v", err)
	}
	selection.Rows[1].Verdict.EvidenceIDs[0] = "finding-4"
	selection.Rows[1].Verdict.MissingContext[0] = "Changed only in selection."
	if verdicts[1].Verdict.EvidenceIDs[0] != "finding-3" || verdicts[1].Verdict.MissingContext[0] == "Changed only in selection." {
		t.Fatal("output mutation escaped into original verdict")
	}
	pool, cycle, verdicts = selectionFixture(t, 100, 10)
	want, err := ap.Select(pool, cycle, verdicts, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantDigest, err := ap.SelectionDigest(want)
	if err != nil {
		t.Fatal(err)
	}
	var readers sync.WaitGroup
	for i := 0; i < 8; i++ {
		readers.Go(func() {
			for j := 0; j < 4; j++ {
				got, err := ap.Select(pool, cycle, verdicts, nil)
				if err != nil {
					t.Error(err)
					return
				}
				digest, err := ap.SelectionDigest(got)
				if err != nil || digest != wantDigest {
					t.Errorf("concurrent selection changed: digest=%s err=%v", digest, err)
					return
				}
				if err := want.Validate(); err != nil {
					t.Error(err)
					return
				}
			}
		})
	}
	readers.Wait()
}
