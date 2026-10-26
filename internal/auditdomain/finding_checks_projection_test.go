package auditdomain

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestFindingChecksProjectionIndexesExactBytesAndPreservesUnknownMethods(t *testing.T) {
	doc := testFindingInventoryDocument(t).Proposals[0].Document
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
	decoded, err := DecodeFindingChecksProjection(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if projection.ProposalDigest != DigestBytes(data) || len(decoded.Checks) != 3 || decoded.Checks[0].Status != "registered" || decoded.Checks[1].Status != "unsupported" || decoded.Checks[2].Status != "plan-too-large" {
		t.Fatalf("projection = %+v", decoded)
	}
	context, err := ExtractFindingChecksContext(data, *decoded.Subject, *decoded.Limitations)
	if err != nil || !sameCanonicalValue(context.Subject, doc.Subject) || !sameCanonicalValue(context.Limitations, doc.Limitations) {
		t.Fatalf("context = %+v, %v", context, err)
	}
	for i, entry := range decoded.Checks {
		check, plan, err := ExtractFindingCheck(data, entry)
		if err != nil || check != doc.ProposedChecks[i] {
			t.Fatalf("check%d = %+v, %v", i, check, err)
		}
		if entry.Status != "unsupported" {
			planBytes, err := EncodeCheckPlan(plan)
			if err != nil || DigestBytes(planBytes) != entry.PlanDigest {
				t.Fatalf("plan%d digest mismatch", i)
			}
		}
	}
	corrupt := bytes.Clone(data)
	corrupt[decoded.Checks[0].Fragment.Offset] = 'x'
	if _, _, err := ExtractFindingCheck(corrupt, decoded.Checks[0]); err == nil {
		t.Fatal("corrupted fragment accepted")
	}
	bad := decoded
	bad.Checks = append([]FindingCheckProjection{}, decoded.Checks...)
	bad.Checks[0].Phase = "replay"
	if _, err := EncodeFindingChecksProjection(bad); err == nil {
		t.Fatal("guessed phase accepted")
	}
	for _, edit := range []func(*FindingChecksProjection){func(p *FindingChecksProjection) { p.Subject = nil }, func(p *FindingChecksProjection) { p.Checks[0].Fragment.Length = p.SizeBytes + 1 }, func(p *FindingChecksProjection) { p.Discarded = true }} {
		bad, _ = DecodeFindingChecksProjection(encoded)
		edit(&bad)
		if _, err := EncodeFindingChecksProjection(bad); err == nil {
			t.Fatal("invalid projection accepted")
		}
	}
}

func TestFindingChecksProjectionBounds512HeadersAndDiscard(t *testing.T) {
	doc := testFindingInventoryDocument(t).Proposals[0].Document
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
	projection, err = DiscardedFindingChecksProjection(DigestBytes(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = EncodeFindingChecksProjection(projection)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = DecodeFindingChecksProjection(encoded); err != nil || projection.Subject != nil || projection.Limitations != nil || len(projection.Checks) != 0 {
		t.Fatal("discarded projection invalid", err)
	}
}
