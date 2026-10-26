//go:build integration

package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func admissionFixture(t *testing.T, name string) (collectingAuditFixture, *ClaimedAdmission, auditstore.AdmissionStepParams) {
	t.Helper()
	f := newCollectingAuditFixture(t, name, 32<<20)
	if _, err := f.pool.Exec(f.ctx, `UPDATE audit_rounds SET state='closed' WHERE audit_id=$1;`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(f.ctx, `UPDATE audits SET max_rounds=4,max_items_per_round=4096,max_items_total=10000 WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	return f, NewClaimedAdmission(f.pool), auditstore.AdmissionStepParams{Claim: f.claim, PreviousRoundID: *f.audit.CurrentRoundID}
}
func admissionProposal(t *testing.T, f collectingAuditFixture) (auditstore.ExactArtifact, []byte, auditdomain.FindingChecksProjection) {
	t.Helper()
	doc := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate", Title: "Possible issue", Description: "Exact source", Subject: &auditdomain.FindingSubject{Kind: "openapi-operation", Key: "get-widget"}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{{Objective: "Trace Unicode Кириллица <>&", Method: "static-trace"}, {Objective: "An ordinary unknown method", Method: "custom-method"}}, Limitations: []string{"partial-trace"}, SeveritySuggestion: "medium"}
	data, err := auditdomain.EncodeFindingProposal(doc)
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	scope, _ := artifacts.ProjectScope(f.audit.ProjectID)
	written, err := service.CreateFrozen(f.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "admission-test-proposal"}, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		t.Fatal(err)
	}
	exact := contracts.ExactContent(written.Ref, written.MediaType, data)
	projection, err := auditdomain.BuildFindingChecksProjection(data, doc)
	if err != nil {
		t.Fatal(err)
	}
	return exact, data, projection
}
func indexAdmissionReceipt(t *testing.T, f collectingAuditFixture, exact auditstore.ExactArtifact, projection auditdomain.FindingChecksProjection, id string) {
	t.Helper()
	insertAuditProposalHoldForRoundTest(t, f.ctx, f.pool, f.audit, exact, id, *f.execution.RunID)
	encoded, err := auditdomain.EncodeFindingChecksProjection(projection)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE finding_proposal_receipts SET checks_projection=$2::text::json WHERE receipt_id=$1`, id, string(encoded)); err != nil {
		t.Fatal(err)
	}
}
func TestPostgresAdmissionAnchorsDBCutoffAndDoesNotResetDeadline(t *testing.T) {
	f, store, p := admissionFixture(t, "admission-clock")
	exact, _, projection := admissionProposal(t, f)
	indexAdmissionReceipt(t, f, exact, projection, "receipt-admission-clock")
	if _, err := f.pool.Exec(f.ctx, `DELETE FROM finding_proposal_audit_holds WHERE audit_id=$1`, f.audit.AuditID); err != nil {
		t.Fatal(err)
	}
	p.Now = time.Date(2040, 1, 1, 0, 0, 0, 0, time.UTC)
	called := false
	initial, err := store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		called = true
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil || !initial.Changed || called || !initial.DeadlineAt.Equal(p.Now.Add(10*time.Minute)) || !initial.SourceCutoffAt.Before(p.Now) {
		t.Fatalf("initial=%+v called=%v err=%v", initial, called, err)
	}
	descriptor, _ := json.Marshal(exact)
	if _, err := f.pool.Exec(f.ctx, `INSERT INTO finding_proposal_audit_holds(receipt_id,audit_id,project_id,proposal_ref,evidence) VALUES($1,$2,$3,$4::jsonb,'[]')`, "receipt-admission-clock", f.audit.AuditID, f.audit.ProjectID, descriptor); err != nil {
		t.Fatal(err)
	}
	// Cursor upper is empty; a late old receipt is found by hold acceptance time.
	next, err := store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		page, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 128)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if !page.Exhausted || !page.LateSources || len(page.Candidates) != 0 {
			t.Fatalf("late source page=%+v", page)
		}
		state := s.State
		state.Exhausted = true
		return auditstore.AdmissionMutation{Next: state}, nil
	})
	if err != nil || !next.Changed || !next.DeadlineAt.Equal(initial.DeadlineAt) {
		t.Fatalf("next=%+v err=%v", next, err)
	}
	restarted, err := store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		return auditstore.AdmissionMutation{Restart: true}, nil
	})
	if err != nil || restarted.Generation == initial.Generation || !restarted.DeadlineAt.Equal(initial.DeadlineAt) {
		t.Fatalf("restart=%+v err=%v", restarted, err)
	}
	p.Now = initial.DeadlineAt
	if _, err = store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		t.Fatal("expired callback")
		return auditstore.AdmissionMutation{}, nil
	}); !errors.Is(err, auditstore.ErrAdmissionDeadline) {
		t.Fatalf("deadline=%v", err)
	}
}
func TestPostgresAdmissionExactSourceSavepointAndDecisionReplay(t *testing.T) {
	f, store, p := admissionFixture(t, "admission-source")
	exact, _, projection := admissionProposal(t, f)
	id := "receipt-admission-source"
	indexAdmissionReceipt(t, f, exact, projection, id)
	initial, err := store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		t.Fatal("first callback")
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := auditstore.AdmissionDecision{ReceiptID: id, ProposedCheckOrdinal: 1, Proposal: exact, SourceCheckDigest: projection.Checks[1].Fragment.Digest, Disposition: "rejected", Code: "unsupported_method"}
	next, err := store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		page, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 128)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if len(page.Candidates) != 2 {
			t.Fatalf("candidates=%+v", page)
		}
		got, e := s.Reads.ReadFindingCheck(ctx, id, 0)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if got.Check.Method != "static-trace" || got.Context.Subject.Key != "get-widget" {
			t.Fatalf("source=%+v", got)
		}
		payload := s.reader.work.payload
		if _, e = s.Reads.ReadFindingCheck(ctx, id, 1); e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		if s.reader.work.payload != payload {
			t.Fatal("source was read twice")
		}
		kept, e := s.WithCandidate(ctx, func(ctx context.Context, a *artifacts.Service, _ auditstore.AdmissionReader) (bool, error) {
			scope, _ := artifacts.ProjectScope(f.audit.ProjectID)
			_, e := a.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "rolled-back-candidate"}, artifacts.Payload{MediaType: "application/json", Data: []byte(`{}`)})
			return false, e
		})
		if e != nil || kept {
			return auditstore.AdmissionMutation{}, e
		}
		state := s.State
		state.Cursor = page.Candidates[1].Cursor
		state.Exhausted = page.Exhausted
		return auditstore.AdmissionMutation{Next: state, Decisions: []auditstore.AdmissionDecision{decision}}, nil
	})
	if err != nil || !next.Changed {
		t.Fatalf("commit=%+v err=%v", next, err)
	}
	var count int
	if err = f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_id=$1 AND name='rolled-back-candidate'`, f.audit.ProjectID).Scan(&count); err != nil || count != 0 {
		t.Fatalf("speculative retained writes=%d,%v", count, err)
	}
	var decisionJSON []byte
	if err = f.pool.QueryRow(f.ctx, `SELECT source_provenance FROM audit_artifact_links WHERE audit_id=$1 AND source_provenance->>'schema'=$2`, f.audit.AuditID, auditstore.AdmissionDecisionSchema).Scan(&decisionJSON); err != nil {
		t.Fatal(err)
	}
	var provenance map[string]any
	if json.Unmarshal(decisionJSON, &provenance) != nil || provenance["disposition"] != "rejected" {
		t.Fatalf("provenance=%s", decisionJSON)
	}
	replay, err := store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil || replay.Changed || replay.Revision != next.Revision || replay.Generation != initial.Generation {
		t.Fatalf("replay=%+v %v", replay, err)
	}
}
