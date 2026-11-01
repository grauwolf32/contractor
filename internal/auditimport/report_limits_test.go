package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// A valid near-8-MiB projection does not imply that its report entry fits:
// the entry also carries exact attribution, state and review metadata.
func TestImporterRejectsConditionBytesBeforeRecordingOversizedReport(t *testing.T) {
	row, projectionBytes := conditionHeavyReportFinding(t, "finding-1")
	entry, err := json.Marshal(newReportFinding(row))
	if err != nil {
		t.Fatal(err)
	}
	if len(entry) <= projectionBytes {
		t.Fatal("fixture has no report metadata overhead")
	}
	store := &fakeImportStore{findings: []auditstore.ReportFinding{row}}
	access := &fakeImportArtifacts{project: map[string][]byte{}, writes: map[string][]byte{}}
	importer, err := New(store, store, &fakeImportRuns{}, access)
	if err != nil {
		t.Fatal(err)
	}
	audit := auditstore.Audit{AuditID: "condition-limit", Revision: 1}
	claim := auditstore.ControllerClaim{AuditID: audit.AuditID, HolderID: "holder", Epoch: 1}
	// The projection alone would fit exactly. The actual serialized finding
	// crosses the existing complete-report bound and must fail before mutation.
	build := reportBuild{projectRevision: 1, positions: []reportPosition{{section: auditstore.ReportSectionProposed}}, contentBytes: artifacts.MaxPayloadSize - projectionBytes}
	before := build.contentBytes
	err = importer.recordNextPage(t.Context(), claim, audit, &build)
	if !errors.Is(err, ErrPermanent) || !strings.Contains(err.Error(), "machine report exceeds its bound") || len(store.pages) != 0 || len(access.writes) != 0 || build.contentBytes != before {
		t.Fatalf("oversized conditions: err=%v pages=%d writes=%d bytes=%d", err, len(store.pages), len(access.writes), build.contentBytes)
	}
	// At the exact entry budget the conditions are retained in full; no field
	// is trimmed to make the page fit. Publication still has its envelope cap.
	build.contentBytes = artifacts.MaxPayloadSize - len(entry)
	if err = importer.recordNextPage(t.Context(), claim, audit, &build); err != nil {
		t.Fatal(err)
	}
	if len(store.pages) != 1 || string(store.pages[0].content) != string(entry) || build.contentBytes != artifacts.MaxPayloadSize {
		t.Fatal("exact-budget entry was changed or dropped")
	}
}

func TestImporterRejectsConditionReportEnvelopeBeforePublishing(t *testing.T) {
	store := &fakeImportStore{}
	build := reportBuild{projectRevision: 1, findings: 8}
	for index := range 8 {
		row, _ := conditionHeavyReportFinding(t, "finding-"+string(rune('1'+index)))
		entry, err := json.Marshal(newReportFinding(row))
		if err != nil {
			t.Fatal(err)
		}
		store.pages = append(store.pages, fakeReportPage{projectRevision: 1, revision: 1, page: auditstore.ReportPage{Ordinal: index, Section: auditstore.ReportSectionProposed, EntryCount: 1, Exhausted: index == 7}, content: entry})
		build.contentBytes += len(entry)
		if index != 0 {
			build.contentBytes++
		}
	}
	if build.contentBytes >= artifacts.MaxPayloadSize {
		t.Fatal("fixture entries already exceed the bound before the report envelope")
	}
	access := &fakeImportArtifacts{project: map[string][]byte{}, writes: map[string][]byte{}}
	importer, err := New(store, store, &fakeImportRuns{}, access)
	if err != nil {
		t.Fatal(err)
	}
	audit := auditstore.Audit{AuditID: "condition-limit", ProjectID: "condition-project", Revision: 1, UpdatedAt: time.Unix(100, 0), BaselineSnapshot: json.RawMessage(`{}`)}
	snapshot := auditstore.ReconcileSnapshot{Audit: audit, Round: &auditstore.Round{RoundID: "condition-round", Ordinal: 1}}
	worked, err := importer.publishReport(context.Background(), auditstore.ControllerClaim{AuditID: audit.AuditID, HolderID: "holder", Epoch: 1}, snapshot, loadResultProfile(t), auditbaseline.ReportProjection{}, build)
	if worked || !errors.Is(err, ErrPermanent) || !strings.Contains(err.Error(), "machine report exceeds its bound") || len(access.writes) != 0 || store.committed.Machine.Artifact.Ref.Revision != nil || store.proposed.Machine.Artifact.Ref.Revision != nil {
		t.Fatalf("oversized report envelope: worked=%t error=%v writes=%d", worked, err, len(access.writes))
	}
}

// Eight individually valid projections fit the entry budget, while their
// exact metadata plus the final report envelope can exceed it. Use the real
// existing bounds, without adding a reduced test-only production limit.
func conditionHeavyReportFinding(t *testing.T, id string) (auditstore.ReportFinding, int) {
	t.Helper()
	conditions := make([]string, 128)
	for index := range conditions {
		conditions[index] = strings.Repeat("x", auditdomain.MaximumStringBytes)
	}
	conditions[len(conditions)-1] = "x"
	revision := "r1"
	row := auditstore.ReportFinding{FindingID: id, State: reportState(auditstore.FindingProposed), Revision: 1, CreatedAt: time.Unix(1, 0),
		FirstProposal: auditstore.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "retained", Name: "original", Revision: &revision}, Digest: auditdomain.DigestBytes([]byte("exact")), MediaType: "application/json", SizeBytes: 8 << 20},
		Projection:    auditdomain.FindingReportProjection{Title: "Condition-heavy candidate", Description: "The conditions remain separate from this description.", Subject: nil, Preconditions: conditions, StandardRefs: []auditdomain.StandardReference{}, Limitations: []string{}},
	}
	initial, err := json.Marshal(newReportFinding(row))
	if err != nil {
		t.Fatal(err)
	}
	remaining := auditdomain.MaximumDocumentBytes - 16 - len(initial)
	if remaining < 0 || remaining >= auditdomain.MaximumStringBytes {
		t.Fatalf("condition fixture growth=%d", remaining)
	}
	conditions[len(conditions)-1] = strings.Repeat("x", remaining+1)
	projection, err := auditdomain.EncodeFindingReportProjection(row.Projection)
	if err != nil {
		t.Fatal(err)
	}
	entry, err := json.Marshal(newReportFinding(row))
	if err != nil || len(entry) != auditdomain.MaximumDocumentBytes-16 {
		t.Fatalf("condition entry bytes=%d error=%v", len(entry), err)
	}
	return row, len(projection)
}
