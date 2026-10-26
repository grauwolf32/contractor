//go:build integration

package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configtest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func stagedAdmissionProfile(t *testing.T, f collectingAuditFixture) *auditbaseline.CheckRoutes {
	t.Helper()
	fixtures := "../../../config/testdata"
	root := configtest.CopyWithPolicies(t, filepath.Join(fixtures, "valid"))
	if err := os.CopyFS(root, os.DirFS(filepath.Join(fixtures, "audit-profile-catalog"))); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(fixtures, "audit-profile-template.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Replace(string(data), "PROFILE_NAME", "staged-finding", 1)
	text = strings.Replace(text, "  workflows:", "  methods:\n    trace:\n      planSchema: contractor.audit.finding-check-plan.v1\n      method: static-trace\n      phase: verify\n      workflowRole: check\n      requiredCapabilities: []\n  workflows:", 1)
	if err = os.MkdirAll(filepath.Join(root, "audit-profiles"), 0755); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, "audit-profiles", "staged.yaml"), []byte(text), 0644); err != nil {
		t.Fatal(err)
	}
	catalog, err := configload.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("staged-finding@1")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, _ := json.Marshal(profile)
	baseline := auditbaseline.BaselineSnapshot{Schema: auditbaseline.Schema, ActiveChecks: contracts.ActiveChecksProhibited, Scope: auditbaseline.Scope{Target: "fixture", AuthorizationScope: "fixture only"}}
	refs, err := resolved.WorkflowSkillRefs(profile.Workflows["check"].Workflow)
	if err != nil {
		t.Fatal(err)
	}
	for _, ref := range refs {
		baseline.Skills = append(baseline.Skills, agentskills.RunSkillSnapshot{Name: ref.Name})
	}
	baselineBytes, _ := json.Marshal(baseline)
	if _, err = f.pool.Exec(f.ctx, `UPDATE audits SET profile_name=$2,profile_version=$3,profile_digest=$4,profile_snapshot=$5::jsonb,baseline_snapshot=$6::jsonb WHERE audit_id=$1`, f.audit.AuditID, profile.Ref.Name, profile.Ref.Version, profile.Ref.Digest, snapshot, baselineBytes); err != nil {
		t.Fatal(err)
	}
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	// Use the exact persisted JSONB baseline digest, like both production paths.
	routes, err := auditbaseline.NewCheckRoutes(audit, profile, baseline)
	if err != nil {
		t.Fatal(err)
	}
	return routes
}
func stagedProposal(t *testing.T, f collectingAuditFixture, count int, ids ...string) {
	t.Helper()
	receiptID := "receipt-staged-source"
	if len(ids) > 0 {
		receiptID = ids[0]
	}
	doc := auditdomain.FindingProposal{Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate", Title: "Potential issue", Description: "One exact immutable source", Subject: &auditdomain.FindingSubject{Kind: "openapi-operation", Key: "get-widget"}, Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{}, EvidenceIDs: []string{}, Limitations: []string{}, SeveritySuggestion: "medium", ProposedChecks: []auditdomain.ProposedCheck{}}
	for i := range count {
		doc.ProposedChecks = append(doc.ProposedChecks, auditdomain.ProposedCheck{Method: "static-trace", Objective: fmt.Sprintf("Trace exact source %d", i)})
	}
	data, err := auditdomain.EncodeFindingProposal(doc)
	if err != nil {
		t.Fatal(err)
	}
	service := artifacts.NewService(artifacts.NewPostgresRepository(f.pool))
	scope, _ := artifacts.ProjectScope(f.audit.ProjectID)
	written, err := service.CreateFrozen(f.ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(f.audit.AuditID), Name: "staged-source-" + receiptID}, artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: data})
	if err != nil {
		t.Fatal(err)
	}
	projection, err := auditdomain.BuildFindingChecksProjection(data, doc)
	if err != nil {
		t.Fatal(err)
	}
	indexAdmissionReceipt(t, f, contracts.ExactContent(written.Ref, written.MediaType, data), projection, receiptID)
}
func writeStagedDocument(t *testing.T, ctx context.Context, s *AdmissionSession, name, media string, data []byte) auditstore.ExactArtifact {
	t.Helper()
	scope, _ := artifacts.ProjectScope(s.reader.project)
	written, err := s.Artifacts.CreateFrozen(ctx, scope, contracts.ArtifactRef{Namespace: auditdomain.ArtifactNamespace(s.State.AuditID), Name: name}, artifacts.Payload{MediaType: media, Data: data})
	if err != nil {
		t.Fatal(err)
	}
	return contracts.ExactContent(written.Ref, written.MediaType, data)
}
func preparedTestItem(t *testing.T, ctx context.Context, s *AdmissionSession, routes *auditbaseline.CheckRoutes, c auditstore.AdmissionCandidate, local int) auditdomain.PreparedRoundItem {
	t.Helper()
	source, err := s.Reads.ReadFindingCheck(ctx, c.ReceiptID, c.Ordinal)
	if err != nil {
		t.Fatal(err)
	}
	ordinal := c.Ordinal
	accepted, err := routes.Build(auditdomain.AcceptedPlanSource{Kind: "finding-receipt", ReceiptID: c.ReceiptID, ProposedCheckOrdinal: &ordinal, RunID: source.Source.OriginRunID, Proposal: source.Source.Original, RetainedProposal: source.Source.Retained}, source.Plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	sourceBytes, err := auditdomain.EncodeFindingCheckSourceDocument(auditdomain.FindingCheckSourceDocument{Schema: auditdomain.FindingCheckSourceSchema, Source: accepted.Source, Check: source.Check, Subject: source.Context.Subject, Limitations: source.Context.Limitations})
	if err != nil {
		t.Fatal(err)
	}
	sourceExact := writeStagedDocument(t, ctx, s, fmt.Sprintf("candidate-source-%03d", ordinal), auditdomain.JSONMediaType, sourceBytes)
	inventory, err := auditdomain.BuildFindingCheckInventory(sourceBytes, auditdomain.FindingCheckInventoryOptions{Common: auditdomain.InventoryOptions{Round: s.State.RoundOrdinal, ProfileMode: "custom-checklist", WorkflowRole: "check", SourceInputName: "proposal_inventory", SourceRef: sourceExact.Ref, ApprovalRequirement: auditdomain.ApprovalNone}, Check: auditdomain.FindingCheckOptions{WorkflowRole: "check", ApprovalRequirement: auditdomain.ApprovalNone, AcceptedPlan: &accepted}})
	if err != nil {
		t.Fatal(err)
	}
	task := writeStagedDocument(t, ctx, s, fmt.Sprintf("candidate-task-%03d", ordinal), auditdomain.PackageMediaType, inventory.Tasks[0].Package)
	execution := inventory.ExecutionManifest.Items[0]
	execution.Ordinal = local
	execution.TaskRef = &task.Ref
	return auditdomain.PreparedRoundItem{ItemID: fmt.Sprintf("typed-item-%03d", ordinal), ItemKey: accepted.ItemKey, Ordinal: s.State.SelectedCount + local, Kind: "finding-verification", SubjectKey: execution.SubjectKey, WorkflowRole: "check", Task: task, InitialState: "ready", ApprovalKind: "none", Origin: auditdomain.PreparedItemOrigin{Schema: auditdomain.ItemOriginSchema, SourceRef: sourceExact.Ref, SourceContentDigest: inventory.SourceContentDigest, SourceMediaType: auditdomain.JSONMediaType, CanonicalInventoryDigest: inventory.CanonicalInventoryDigest, EntryKey: accepted.ItemKey}, Coverage: auditdomain.RoundResultCoverage{Status: auditdomain.CoverageNotTested, Requested: inventory.Coverage.Rows[0].Requested, Completed: []string{}, Gaps: []string{}}, ProposalSources: []auditdomain.PreparedProposalSource{{ReceiptID: c.ReceiptID, ProposedCheckOrdinal: ordinal, Proposal: source.Source.Retained}}, Execution: execution}
}
func preparedAdmissionFixture(t *testing.T, count int) (collectingAuditFixture, *ClaimedAdmission, auditstore.AdmissionStepParams, auditstore.AdmissionProgress) {
	t.Helper()
	f, store, p := admissionFixture(t, fmt.Sprintf("admission-staging-%d", count))
	routes := stagedAdmissionProfile(t, f)
	stagedProposal(t, f, count)
	state, err := store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		return auditstore.AdmissionMutation{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for state.SelectedCount < count {
		state, err = store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
			candidates, e := s.Reads.ReadCandidates(ctx, s.State.Cursor, 16)
			if e != nil {
				return auditstore.AdmissionMutation{}, e
			}
			page := auditdomain.PreparedRoundPage{Schema: auditdomain.RoundSelectionPageSchema, AuditID: s.State.AuditID, RoundID: s.State.RoundID, Generation: s.State.Generation, PageOrdinal: s.State.SelectionPages, FirstItemOrdinal: s.State.SelectedCount, PreviousPageDigest: s.State.SelectionPagesDigest, Items: []auditdomain.PreparedRoundItem{}}
			for i, c := range candidates.Candidates {
				page.Items = append(page.Items, preparedTestItem(t, ctx, s, routes, c, i))
			}
			data, e := auditdomain.EncodeRoundSelectionPage(page)
			if e != nil {
				return auditstore.AdmissionMutation{}, e
			}
			exact := writeStagedDocument(t, ctx, s, fmt.Sprintf("selection-%03d", page.PageOrdinal), auditdomain.RoundSelectionPageMediaType, data)
			next := s.State
			next.Cursor = candidates.Candidates[len(candidates.Candidates)-1].Cursor
			next.Exhausted = candidates.Exhausted
			return auditstore.AdmissionMutation{Next: next, SelectionPage: &exact, PageContent: data}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	state, err = store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		root := auditdomain.RoundSelectionRoot{Schema: auditdomain.RoundSelectionRootSchema, AuditID: s.State.AuditID, RoundID: s.State.RoundID, Generation: s.State.Generation, ProfileDigest: s.State.ProfileDigest, BaselineDigest: s.State.BaselineDigest, PageCount: s.State.SelectionPages, ItemCount: s.State.SelectedCount, PagesDigest: s.State.SelectionPagesDigest}
		data, e := auditdomain.EncodeRoundSelectionRoot(root)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		exact := writeStagedDocument(t, ctx, s, "selection-root", auditdomain.RoundSelectionRootMediaType, data)
		return auditstore.AdmissionMutation{SelectionRoot: &exact}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for state.StagedCount < count {
		state, err = store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
			_, data, e := s.Reads.ReadSelectionPage(ctx, s.State.PreparedPages)
			if e != nil {
				return auditstore.AdmissionMutation{}, e
			}
			page, e := auditdomain.DecodeRoundSelectionPage(data)
			if e != nil {
				return auditstore.AdmissionMutation{}, e
			}
			page.Schema = auditdomain.PreparedRoundPageSchema
			page.PreviousPageDigest = s.State.PagesDigest
			data, e = auditdomain.EncodePreparedRoundPage(page)
			if e != nil {
				return auditstore.AdmissionMutation{}, e
			}
			exact := writeStagedDocument(t, ctx, s, fmt.Sprintf("prepared-%03d", page.PageOrdinal), auditdomain.PreparedRoundPageMediaType, data)
			return auditstore.AdmissionMutation{PreparedPage: &exact, PageContent: data}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if count, e := f.store.CountItems(f.ctx, f.audit.AuditID); e != nil || count != 1 {
		t.Fatalf("private membership leaked: %d,%v", count, e)
	}
	state, err = store.WithStep(f.ctx, p, func(ctx context.Context, s *AdmissionSession) (auditstore.AdmissionMutation, error) {
		root := auditdomain.PreparedRoundRoot{Schema: auditdomain.PreparedRoundRootSchema, AuditID: s.State.AuditID, RoundID: s.State.RoundID, Generation: s.State.Generation, ProfileDigest: s.State.ProfileDigest, BaselineDigest: s.State.BaselineDigest, Source: *s.State.SelectionRoot, PageCount: s.State.PreparedPages, ItemCount: s.State.StagedCount, PagesDigest: s.State.PagesDigest}
		data, e := auditdomain.EncodePreparedRoundRoot(root)
		if e != nil {
			return auditstore.AdmissionMutation{}, e
		}
		exact := writeStagedDocument(t, ctx, s, "prepared-root", auditdomain.PreparedRoundRootMediaType, data)
		return auditstore.AdmissionMutation{PreparedRoot: &exact}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return f, store, p, state
}
func TestPostgresAdmissionStagesMoreThan128AndAcceptsConstantParentSeal(t *testing.T) {
	f, store, p, state := preparedAdmissionFixture(t, 130)

	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	params := auditstore.AcceptRoundParams{Claim: f.claim, ExpectedAuditRevision: audit.Revision, PreviousRoundID: p.PreviousRoundID, RoundID: state.RoundID, RoundOrdinal: 2, Manifest: *state.PreparedRoot, PreparedRoot: state.PreparedRoot, Generation: state.Generation}
	accepted, changed, err := f.store.AcceptNextRound(f.ctx, params)
	if err != nil || !changed || accepted.ExpectedItemCount != 130 {
		t.Fatalf("accepted=%+v,%v,%v", accepted, changed, err)
	}
	if _, changed, err = f.store.AcceptNextRound(f.ctx, params); err != nil || changed {
		t.Fatalf("lost reply replay changed=%v,%v", changed, err)
	}
	selected, err := f.store.ReadSelectedPreparedManifest(f.ctx, f.audit.AuditID, state.RoundID, []string{"typed-item-000", "typed-item-129"}, 1<<20, 512)
	if err != nil || len(selected) != 2 || selected[1].Ordinal != 129 {
		t.Fatalf("sparse selected=%+v,%v", selected, err)
	}
	one, _ := json.Marshal(selected[0])
	selected, err = f.store.ReadSelectedPreparedManifest(f.ctx, f.audit.AuditID, state.RoundID, []string{"typed-item-000", "typed-item-129"}, len(one)+32, 512)
	if err != nil || len(selected) != 1 {
		t.Fatalf("bounded prefix=%d,%v", len(selected), err)
	}
	if _, err = f.pool.Exec(f.ctx, `INSERT INTO audit_items(item_id,audit_id,round_id,item_key,ordinal,kind,subject_key,task_ref,task_digest,origin,workflow_role)
 SELECT 'untyped-rogue',audit_id,$2,'untyped-rogue',130,kind,subject_key,task_ref,task_digest,origin,workflow_role FROM audit_items WHERE item_id=$1`, f.itemID, state.RoundID); err == nil {
		t.Fatal("untyped row inserted into sealed typed membership")
	}
	if _, err = f.pool.Exec(f.ctx, `UPDATE audit_items SET prepared_execution='{}' WHERE item_id='typed-item-000'`); err == nil {
		t.Fatal("private execution projection mutable")
	}
	if _, err = f.pool.Exec(f.ctx, `DELETE FROM audit_items WHERE item_id='typed-item-000'`); err == nil {
		t.Fatal("accepted membership deletable")
	}
	if count, e := f.store.CountItems(f.ctx, f.audit.AuditID); e != nil || count != 131 {
		t.Fatalf("accepted count=%d,%v", count, e)
	}
	if _, err = store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		return auditstore.AdmissionMutation{}, nil
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("old Round still admits: %v", err)
	}
}

func TestPostgresAdmissionCancellationRemovesPrivateRowsInBoundedPasses(t *testing.T) {
	f, store, p, state := preparedAdmissionFixture(t, 17)
	audit, err := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = f.store.RequestDelete(f.ctx, auditstore.DeleteParams{OwnerID: audit.OwnerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision, IdempotencyKey: "cancel-private", RequestDigest: auditdomain.DigestBytes([]byte("cancel"))})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.WithStep(f.ctx, p, func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error) {
		t.Fatal("closed dispatch ran callback")
		return auditstore.AdmissionMutation{}, nil
	}); !errors.Is(err, auditstore.ErrPrecondition) {
		t.Fatalf("closed admission=%v", err)
	}
	previous := 17
	for step := 0; step < 8; step++ {
		changed, e := f.store.SettleUndispatched(f.ctx, f.claim, 8)
		if e != nil {
			t.Fatal(e)
		}
		if changed < 1 || changed > 8 {
			t.Fatalf("cleanup work=%d", changed)
		}
		var physical, counter int
		if e = f.pool.QueryRow(f.ctx, `SELECT (SELECT count(*) FROM audit_items WHERE audit_id=$1 AND round_id=$2),prepared_staged_count FROM audit_rounds WHERE audit_id=$1 AND round_id=$2`, audit.AuditID, state.RoundID).Scan(&physical, &counter); e != nil {
			t.Fatal(e)
		}
		if counter != physical || previous-physical > 8 {
			t.Fatalf("cleanup previous=%d physical=%d counter=%d", previous, physical, counter)
		}
		previous = physical
		if physical == 0 {
			break
		}
	}
	if previous != 0 {
		t.Fatal("private membership did not drain")
	}
	if visible, e := f.store.CountItems(f.ctx, audit.AuditID); e != nil || visible != 1 {
		t.Fatalf("private drain changed accepted count=%d,%v", visible, e)
	}
}

func TestPostgresAdmissionPreparedAndAcceptedMembershipPurgeWithAudit(t *testing.T) {
	for _, publish := range []bool{false, true} {
		t.Run(fmt.Sprint(publish), func(t *testing.T) {
			f, _, p, state := preparedAdmissionFixture(t, 9)
			if publish {
				audit, e := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
				if e != nil {
					t.Fatal(e)
				}
				_, changed, e := f.store.AcceptNextRound(f.ctx, auditstore.AcceptRoundParams{Claim: f.claim, ExpectedAuditRevision: audit.Revision, PreviousRoundID: p.PreviousRoundID, RoundID: state.RoundID, RoundOrdinal: state.RoundOrdinal, Manifest: *state.PreparedRoot, PreparedRoot: state.PreparedRoot, Generation: state.Generation})
				if e != nil || !changed {
					t.Fatalf("publish=%v,%v", changed, e)
				}
			}
			code := "fixture_invalid_result"
			source := testExact("outputs", "invalid-result", "result-r1")
			if _, inserted, e := f.store.Collect(f.ctx, auditstore.CollectParams{Claim: f.claim, ReceiptID: "typed-purge-collection", ExecutionID: f.execution.ExecutionID, Disposition: auditstore.CollectionInvalidResult, SourceOutput: &source, ErrorCode: &code, RequestDigest: auditdomain.DigestBytes([]byte(code)), Items: []auditstore.CollectionItem{{ExecutionItemID: f.memberID, Disposition: auditstore.CollectionInvalidResult, FinalDisposition: auditstore.FinalInvalidResult, Coverage: auditstore.Coverage{Status: auditstore.CoverageInconclusive, Requested: []string{}, Completed: []string{}, Gaps: []string{code}}}}}); e != nil || !inserted {
				t.Fatalf("collect=%v,%v", inserted, e)
			}
			claim := drainDeletedAuditForPurge(t, f)
			// This shared purge fixture removes the native Run row directly. Seed the
			// exact receipt boundary that the Run deletion hook records; its complete
			// production path is covered by finding intake's deletion integration tests.
			tag, e := f.pool.Exec(f.ctx, `UPDATE finding_proposal_receipts SET source_run_deleted_at=clock_timestamp(),retention_state='audit-held' WHERE run_id=$1 AND NOT EXISTS(SELECT 1 FROM workflow_runs WHERE run_id=$1)`, *f.execution.RunID)
			if e != nil || tag.RowsAffected() != 1 {
				t.Fatalf("deleted producer boundary=%d,%v", tag.RowsAffected(), e)
			}
			for range 8 {
				n, e := f.store.SettleUndispatched(f.ctx, claim, 8)
				if e != nil {
					t.Fatal(e)
				}
				if n == 0 {
					break
				}
			}
			namespace := auditdomain.ArtifactNamespace(f.audit.AuditID)
			for attempt := range 16 {
				if e := f.store.PurgeClaimed(f.ctx, claim, namespace); e != nil {
					t.Fatal(e)
				}
				_, e := f.store.Get(f.ctx, f.audit.OwnerID, f.audit.AuditID)
				if errors.Is(e, auditstore.ErrNotFound) {
					break
				}
				if e != nil {
					t.Fatal(e)
				}
				if attempt == 15 {
					t.Fatal("typed Audit purge did not converge")
				}
			}
			for _, table := range []string{"audit_items", "audit_rounds", "audit_artifact_links", "finding_proposal_audit_holds", "audit_review_requests"} {
				assertPurgedTable(t, f, table)
			}
			var bindings int
			if e := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$1 AND namespace=$2`, f.audit.ProjectID, namespace).Scan(&bindings); e != nil || bindings != 0 {
				t.Fatalf("typed frozen namespace remains=%d,%v", bindings, e)
			}
			var projectionBytes []byte
			if e := f.pool.QueryRow(f.ctx, `SELECT checks_projection FROM finding_proposal_receipts WHERE receipt_id='receipt-staged-source'`).Scan(&projectionBytes); e != nil {
				t.Fatal(e)
			}
			projection, e := auditdomain.DecodeFindingChecksProjection(projectionBytes)
			if e != nil || !projection.Discarded || len(projection.Checks) != 0 {
				t.Fatalf("last hold source discarded=%v checks=%d error=%v", projection.Discarded, len(projection.Checks), e)
			}
		})
	}
}
