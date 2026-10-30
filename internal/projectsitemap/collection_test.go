package projectsitemap

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

func TestCollectionIngressValidatesOneSourceIndependentContract(t *testing.T) {
	for _, kind := range []string{"http", "caido", "ffuf", "katana", "new-observer.v7"} {
		valid := CollectionIngress{Kind: kind, Observations: []Observation{observation("/users")}}
		if err := valid.Validate(); err != nil {
			t.Fatalf("valid source %s rejected: %v", kind, err)
		}
		valid.Observations = []Observation{}
		if err := valid.Validate(); err != nil {
			t.Fatalf("empty batch rejected: %v", err)
		}
		valid.Gaps = []CollectionGap{{Reason: "projection", Count: 2}}
		if err := valid.Validate(); err != nil {
			t.Fatalf("gap-only batch rejected: %v", err)
		}
	}
	cases := []struct {
		name   string
		mutate func(*CollectionIngress)
	}{
		{"reserved manual", func(v *CollectionIngress) { v.Kind = "manual" }},
		{"reserved openapi", func(v *CollectionIngress) { v.Kind = "openapi" }},
		{"invalid source name", func(v *CollectionIngress) { v.Kind = "source/name" }},
		{"long source name", func(v *CollectionIngress) { v.Kind = strings.Repeat("x", contracts.MaxNameLength+1) }},
		{"missing observations", func(v *CollectionIngress) { v.Observations = nil }},
		{"unknown gap", func(v *CollectionIngress) { v.Gaps = []CollectionGap{{Reason: "provider-error", Count: 1}} }},
		{"zero gap", func(v *CollectionIngress) { v.Gaps = []CollectionGap{{Reason: "projection", Count: 0}} }},
		{"unbounded gap", func(v *CollectionIngress) {
			v.Gaps = []CollectionGap{{Reason: "projection", Count: MaxOmissionCount + 1}}
		}},
		{"duplicate gap reason", func(v *CollectionIngress) {
			v.Gaps = []CollectionGap{{Reason: "projection", Count: 1}, {Reason: "projection", Count: 2}}
		}},
		{"versionless evidence", func(v *CollectionIngress) { v.Artifact = &contracts.ArtifactRef{Namespace: "research", Name: "report"} }},
		{"non UTC timestamp", func(v *CollectionIngress) { v.Observations[0].At = "2026-10-29T15:00:00+03:00" }},
		{"malformed explicit parameter", func(v *CollectionIngress) { v.Observations[0].Parameters = []string{"invalid\nname"} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input := CollectionIngress{Kind: "new-observer.v7", Observations: []Observation{observation("/users")}}
			tc.mutate(&input)
			if err := input.Validate(); err == nil {
				t.Fatal("invalid normalized batch accepted")
			}
		})
	}
}

func TestCollectionIngressRejectsExtraAuthorityAndUntypedNulls(t *testing.T) {
	for _, raw := range []string{
		`{"kind":"new-observer","observations":null}`,
		`{"kind":"new-observer"}`,
		`{"kind":"new-observer","observations":[],"gaps":null}`,
		`{"kind":"new-observer","observations":[],"artifact":null}`,
		`{"kind":"new-observer","observations":[],"sourceId":"forged"}`,
		`{"kind":"new-observer","observations":[],"runId":"other"}`,
		`{"kind":"new-observer","observations":[],"format":"custom@1"}`,
		`{"kind":"new-observer","observations":[],"gaps":[{"sourceId":"forged","reason":"projection","count":1}]}`,
	} {
		var input CollectionIngress
		if err := strictjson.DecodeNonNull([]byte(raw), &input); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	var editor EditorMutation
	if err := json.Unmarshal([]byte(`{"operation":"declare","expectedRevision":"absent","url":"https://example.test/","method":"GET","artifact":{"namespace":"research","name":"report","revision":"r1"}}`), &editor); err == nil {
		t.Fatal("manual editor gained observation evidence authority")
	}
}
