package auditdomain

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFindingChecksProjectionIndexesExactBytesAndPreservesUnknownMethods(t *testing.T) {
	doc := testProjectedFindingProposal()
	doc.ProposedChecks = []ProposedCheck{{Objective: "\u041a\u0438\u0440\u0438\u043b\u043b\u0438\u0446\u0430 <>& \\\"\n\u2028\u2029", Method: "verify"}, {Objective: "Ordinary valid method", Method: "operator-method"}, {Objective: strings.Repeat("a", 64<<10), Method: "replay"}}
	// Pretty, noncanonical input proves offsets belong to the exact source.
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	projection, err := BuildFindingChecksProjection(data, doc)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeFindingChecksProjection(projection)
	if err != nil {
		t.Fatal(err)
	}
	var decoded FindingChecksProjection
	err = json.Unmarshal(encoded, &decoded)
	if err != nil {
		t.Fatal(err)
	}
	if projection.ProposalDigest != DigestBytes(data) || len(decoded.Checks) != 3 || decoded.Checks[0].Status != "registered" || decoded.Checks[1].Status != "unsupported" || decoded.Checks[2].Status != "plan-too-large" {
		t.Fatalf("projection = %+v", decoded)
	}
	for i, entry := range decoded.Checks {
		fragment := data[entry.Fragment.Offset : entry.Fragment.Offset+entry.Fragment.Length]
		var check ProposedCheck
		if err := json.Unmarshal(fragment, &check); err != nil || check != doc.ProposedChecks[i] || DigestBytes(fragment) != entry.Fragment.Digest {
			t.Fatalf("indexed exact check %d differs: %+v %v", i, check, err)
		}
		if entry.Status != "unsupported" {
			plan, err := BuildFindingCheckPlan(doc, i)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := EncodeCheckPlan(plan)
			if err != nil || DigestBytes(encoded) != entry.PlanDigest {
				t.Fatalf("plan digest %d differs", i)
			}
		}
	}
	bad := decoded
	bad.Checks = append([]FindingCheckProjection{}, decoded.Checks...)
	bad.Checks[0].Phase = "replay"
	if _, err := EncodeFindingChecksProjection(bad); err == nil {
		t.Fatal("guessed phase accepted")
	}
	for _, edit := range []func(*FindingChecksProjection){func(p *FindingChecksProjection) { p.Subject = nil }, func(p *FindingChecksProjection) { p.Checks[0].Fragment.Length = p.SizeBytes + 1 }, func(p *FindingChecksProjection) { p.Discarded = true }} {
		_ = json.Unmarshal(encoded, &bad)
		edit(&bad)
		if _, err := EncodeFindingChecksProjection(bad); err == nil {
			t.Fatal("invalid projection accepted")
		}
	}
}

func TestFindingChecksProjectionBounds512HeadersAndDiscard(t *testing.T) {
	doc := testProjectedFindingProposal()
	doc.ProposedChecks = make([]ProposedCheck, MaximumCoverageValues)
	for i := range doc.ProposedChecks {
		doc.ProposedChecks[i] = ProposedCheck{Objective: "check", Method: strings.Repeat("x", 160)}
	}
	data, err := EncodeFindingProposal(doc)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := BuildFindingChecksProjection(data, doc)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeFindingChecksProjection(projection)
	if err != nil || len(encoded) > MaximumFindingChecksProjectionBytes {
		t.Fatalf("maximum projection = %d,%v", len(encoded), err)
	}
	projection = FindingChecksProjection{Schema: FindingChecksProjectionSchema, ProposalDigest: DigestBytes(data), SizeBytes: int64(len(data)), Discarded: true, Checks: []FindingCheckProjection{}}
	encoded, err = EncodeFindingChecksProjection(projection)
	if err != nil {
		t.Fatal(err)
	}
	var decoded FindingChecksProjection
	err = json.Unmarshal(encoded, &decoded)
	if err != nil || projection.Subject != nil || projection.Limitations != nil || len(projection.Checks) != 0 {
		t.Fatal("discarded projection invalid", err)
	}
}

func TestFindingChecksProjectionMaximumRegisteredHeadersAndStrictPresence(t *testing.T) {
	doc := testProjectedFindingProposal()
	doc.ProposedChecks = make([]ProposedCheck, MaximumCoverageValues)
	for i := range doc.ProposedChecks {
		doc.ProposedChecks[i] = ProposedCheck{Objective: strings.Repeat("x", 16000), Method: "applicability"}
	}
	base, err := EncodeCheckPlan(CheckPlan{Schema: "contractor.audit.finding-check-plan.v1", Method: "applicability", Phase: "applicability", Objective: "x"})
	if err != nil {
		t.Fatal(err)
	}
	doc.ProposedChecks[0].Objective = strings.Repeat("x", MaximumAcceptedPlanBytes-len(base)+1)
	data, err := EncodeFindingProposal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 7<<20 {
		t.Fatal("fixture did not reach a large exact offset")
	}
	projection, err := BuildFindingChecksProjection(data, doc)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeFindingChecksProjection(projection)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > MaximumFindingChecksProjectionBytes || projection.Checks[0].PlanBytes != 65536 || projection.Checks[0].Status != "registered" || projection.Checks[511].Fragment.Offset < 7<<20 {
		t.Fatalf("registered bound = %d bytes, first %+v", len(encoded), projection.Checks[0])
	}
	bad := projection
	overlapping := projection.Checks[0].Fragment
	bad.Subject = &overlapping
	if _, err = EncodeFindingChecksProjection(bad); err == nil {
		t.Fatal("overlapping context accepted")
	}
}
