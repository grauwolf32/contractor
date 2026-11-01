package pgstore

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func admissionProgressFixture() auditstore.AdmissionProgress {
	at := time.Date(2040, 1, 1, 0, 0, 0, 0, time.UTC)
	return auditstore.AdmissionProgress{Schema: auditstore.AdmissionSchema, Generation: "generation", Revision: 1, AuditID: "audit", PreviousRoundID: "round-one", RoundID: "round-two", RoundOrdinal: 2, ProfileDigest: auditdomain.DigestBytes([]byte("profile")), BaselineDigest: auditdomain.DigestBytes([]byte("baseline")), SourceCutoffAt: at, DeadlineAt: at.Add(time.Minute), Budget: json.RawMessage(`{}`), PagesDigest: auditdomain.DigestBytes(nil)}
}
func progressRoot() *contracts.ExactArtifact {
	revision := "root-revision"
	return &contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace("audit"), Name: "root", Revision: &revision}, Digest: auditdomain.DigestBytes([]byte("root")), MediaType: auditdomain.PreparedRoundRootMediaType, SizeBytes: 100}
}
func TestAdmissionProgressSeparatesPublishedAndStagedMembership(t *testing.T) {
	initial := admissionProgressFixture()
	published := initial
	published.SelectedCount, published.PublishedPages, published.PagesDigest, published.Exhausted = 3, 2, auditdomain.DigestBytes([]byte("tip")), true
	rooted := published
	rooted.Root = progressRoot()
	partial := rooted
	partial.StagedCount, partial.StagedPages = 2, 1
	complete := rooted
	complete.StagedCount, complete.StagedPages = 3, 2
	for _, state := range []auditstore.AdmissionProgress{initial, published, rooted, partial, complete} {
		data, err := encodeAdmissionProgress(state)
		if err != nil {
			t.Fatal(err)
		}
		var decoded auditstore.AdmissionProgress
		if strictjson.Decode(data, &decoded) != nil {
			t.Fatal("canonical progress did not decode")
		}
	}
	for _, tc := range []struct {
		name   string
		mutate func(*auditstore.AdmissionProgress)
	}{
		{"published beyond selected", func(s *auditstore.AdmissionProgress) { s.PublishedPages = 4 }},
		{"missing page", func(s *auditstore.AdmissionProgress) { s.PublishedPages = 0 }},
		{"staged beyond selected", func(s *auditstore.AdmissionProgress) { s.StagedCount = 4 }},
		{"staged pages beyond published", func(s *auditstore.AdmissionProgress) { s.StagedPages = 3; s.StagedCount = 3 }},
		{"pages without items", func(s *auditstore.AdmissionProgress) { s.StagedPages = 1; s.StagedCount = 0 }},
		{"items without pages", func(s *auditstore.AdmissionProgress) { s.StagedPages = 0; s.StagedCount = 1 }},
		{"partial final page", func(s *auditstore.AdmissionProgress) { s.StagedPages = 2; s.StagedCount = 2 }},
		{"partial final count", func(s *auditstore.AdmissionProgress) { s.StagedPages = 1; s.StagedCount = 3 }},
		{"invalid tip", func(s *auditstore.AdmissionProgress) { s.PagesDigest = "bad" }},
		{"stage before root", func(s *auditstore.AdmissionProgress) { s.Root = nil; s.StagedCount = 1; s.StagedPages = 1 }},
		{"foreign root", func(s *auditstore.AdmissionProgress) { s.Root.Ref.Namespace = "foreign" }},
		{"missing root revision", func(s *auditstore.AdmissionProgress) { s.Root.Ref.Revision = nil }},
		{"root media", func(s *auditstore.AdmissionProgress) { s.Root.MediaType = "application/json" }},
		{"root missing bytes", func(s *auditstore.AdmissionProgress) { s.Root.SizeBytes = 0 }},
		{"root byte overflow", func(s *auditstore.AdmissionProgress) {
			s.Root.SizeBytes = auditdomain.MaximumPreparedRoundRootBytes + 1
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := rooted
			state.Root = progressRoot()
			tc.mutate(&state)
			if _, err := encodeAdmissionProgress(state); err == nil {
				t.Fatal("invalid progress accepted")
			}
		})
	}
	initial.PagesDigest = auditdomain.DigestBytes([]byte("nonempty"))
	if _, err := encodeAdmissionProgress(initial); err == nil {
		t.Fatal("nonempty initial tip accepted")
	}
	data, _ := encodeAdmissionProgress(rooted)
	for _, old := range []string{`"selection_pages":0,`, `"prepared_pages":0,`, `"selection_root":null,`, `"prepared_root":null,`, `"selection_pages_digest":"",`} {
		var decoded auditstore.AdmissionProgress
		if strictjson.Decode(bytes.Replace(data, []byte(`"schema":`), []byte(old+`"schema":`), 1), &decoded) == nil {
			t.Fatalf("old checkpoint field accepted: %s", old)
		}
	}
}
