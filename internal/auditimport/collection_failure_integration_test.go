//go:build integration

package auditimport

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/artifactstest"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/hookstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// An earlier attempt retained every proposal, so the next attempt reads the
// pinned standards first while it resolves the result's proposal selections.
// A storage failure of that read leaves the execution collecting without a
// receipt or a consumed item attempt, and a later read collects it.
func TestPostgresStandardReadFailureAfterRetentionKeepsExecutionCollecting(t *testing.T) {
	var pinned auditstandards.PinnedPackage
	f := newCompletionFixtureWithOptions(t, completionPool(t), completionOptions{
		acceptance: "automatic",
		standards: func(t *testing.T, f *completionFixture) []auditstandards.PinnedPackage {
			pinned = f.retainStandard(t, top10ImportPackage(t))
			return []auditstandards.PinnedPackage{pinned}
		},
	})
	receiptID := f.proposal(t, "candidate", auditdomain.FindingProposal{
		Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate",
		Title: "Candidate", Description: "Check the access control of the handler",
		Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
		Preconditions: []string{},
		StandardRefs: []auditdomain.StandardReference{{
			Scheme: pinned.Reference.Scheme, Version: pinned.Reference.Version, RequirementID: "A01:2025",
		}},
		EvidenceIDs:    []string{},
		ProposedChecks: []auditdomain.ProposedCheck{},
		Limitations:    []string{},
	})
	f.writeResult(t, map[string][]auditdomain.ProposalSelection{
		"first": {{InvocationID: f.proposalInvocation(), ClientKey: "candidate"}},
	})
	execution, snapshot := f.terminal(t, runstore.RunSucceeded)
	access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
	mustCompletion(t, err)
	intake, err := findingintake.New(f.pool)
	mustCompletion(t, err)
	faults := &collectionFaults{standardRef: pinned.Retained.Artifact}
	importer, err := New(f.audits, f.audits, f.runs, collectionFaultAccess{ArtifactAccess: access, faults: faults}, intake)
	mustCompletion(t, err)

	// The first attempt retains the proposal and then loses its result read.
	faults.bindingErr = errors.New("result binding storage unavailable")
	changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
	if changed || !errors.Is(err, faults.bindingErr) || faults.standardReads != 1 {
		t.Fatalf("attempt with lost result read = (%t, %v), standard reads %d", changed, err, faults.standardReads)
	}
	var retention string
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT retention_state FROM finding_proposal_receipts WHERE receipt_id = $1`, receiptID).Scan(&retention))
	if retention != string(findingintake.RetentionProjectHeld) {
		t.Fatalf("first attempt did not retain the proposal: %q", retention)
	}
	f.requireCollecting(t, execution.ExecutionID)

	// Retention now skips the proposal; resolution reads the standard first.
	faults.bindingErr = nil
	faults.standardErr = errors.New("standard storage unavailable")
	changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
	if changed || !errors.Is(err, faults.standardErr) || errors.Is(err, ErrPermanent) || faults.standardReads != 2 {
		t.Fatalf("attempt with lost standard read = (%t, %v), standard reads %d", changed, err, faults.standardReads)
	}
	f.requireCollecting(t, execution.ExecutionID)

	faults.standardErr = nil
	changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
	if err != nil || !changed {
		t.Fatalf("attempt after the storage recovered = (%t, %v)", changed, err)
	}
	f.requireAcceptedAssessment(t, execution.ExecutionID, receiptID)
}

// Pinned standard data that cannot be validated never validates on a retry, so
// the production collection path settles it through one contract-invalid
// receipt instead of judging the Worker's proposal or retrying. Retention reads
// the standards for a proposal it has not retained yet. A proposal the owner
// held before the Run finished needs no retention read, so result resolution
// is the first to read them.
func TestPostgresInvalidPinnedStandardSettlesCollectionContractInvalid(t *testing.T) {
	pool := completionPool(t)
	for _, corruption := range []struct {
		name    string
		corrupt func(*auditstandards.PinnedPackage)
		// reads is the number of retained standard reads of the first
		// attempt: an invalid pin is rejected before its package is read.
		reads int
	}{
		{name: "invalid baseline pin", corrupt: func(pinned *auditstandards.PinnedPackage) {
			pinned.Catalog.Artifact.Namespace = "standards"
		}},
		{name: "retained package differs from its pin", reads: 1, corrupt: func(pinned *auditstandards.PinnedPackage) {
			pinned.Title = "A different standard title"
		}},
	} {
		for _, entry := range []struct {
			name      string
			ownerHold bool
		}{
			{name: "at retention"},
			{name: "at resolution after an owner hold", ownerHold: true},
		} {
			t.Run(corruption.name+"/"+entry.name, func(t *testing.T) {
				var pinned auditstandards.PinnedPackage
				f := newCompletionFixtureWithOptions(t, pool, completionOptions{
					acceptance: "automatic",
					standards: func(t *testing.T, f *completionFixture) []auditstandards.PinnedPackage {
						pinned = f.retainStandard(t, top10ImportPackage(t))
						corruption.corrupt(&pinned)
						return []auditstandards.PinnedPackage{pinned}
					},
				})
				receiptID := f.proposal(t, "candidate", auditdomain.FindingProposal{
					Schema: auditdomain.FindingProposalSchema, ClientKey: "candidate",
					Title: "Candidate", Description: "Check the access control of the handler",
					Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
					Preconditions: []string{},
					StandardRefs: []auditdomain.StandardReference{{
						Scheme: pinned.Reference.Scheme, Version: pinned.Reference.Version, RequirementID: "A01:2025",
					}},
					EvidenceIDs:    []string{},
					ProposedChecks: []auditdomain.ProposedCheck{},
					Limitations:    []string{},
				})
				f.writeResult(t, map[string][]auditdomain.ProposalSelection{
					"first": {{InvocationID: f.proposalInvocation(), ClientKey: "candidate"}},
				})
				intake, err := findingintake.New(f.pool)
				mustCompletion(t, err)
				if entry.ownerHold {
					var proposalRef contracts.ArtifactRef
					mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT proposal_ref FROM finding_proposal_receipts WHERE receipt_id = $1`, receiptID).Scan(&proposalRef))
					_, _, err = intake.ImportIntoAudit(f.ctx, findingintake.ImportRequest{
						OwnerID: "owner", AuditID: f.id, RunID: f.run.RunID, Proposal: proposalRef,
					})
					mustCompletion(t, err)
				}
				execution, snapshot := f.terminal(t, runstore.RunSucceeded)
				access, err := NewArtifactAccess(f.artifacts, hookstore.NewClaimedArtifacts(f.pool))
				mustCompletion(t, err)
				faults := &collectionFaults{standardRef: pinned.Retained.Artifact}
				importer, err := New(f.audits, f.audits, f.runs, collectionFaultAccess{ArtifactAccess: access, faults: faults}, intake)
				mustCompletion(t, err)

				changed, err := importer.Collect(f.ctx, f.claim, snapshot, execution)
				if err != nil || !changed {
					t.Fatalf("collection with invalid pinned standard = (%t, %v)", changed, err)
				}
				f.requireContractInvalid(t, execution.ExecutionID)
				if faults.standardReads != corruption.reads {
					t.Fatalf("retained standard reads = %d, want %d", faults.standardReads, corruption.reads)
				}
				// A retry with the stale observation replays the same receipt.
				changed, err = importer.Collect(f.ctx, f.claim, snapshot, execution)
				if err != nil || !changed {
					t.Fatalf("replayed collection with invalid pinned standard = (%t, %v)", changed, err)
				}
				f.requireContractInvalid(t, execution.ExecutionID)
				var assessments int
				mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT count(*) FROM artifact_bindings WHERE scope_kind='project' AND scope_id=$2 AND namespace='finding-history' AND name LIKE $1`, auditdomain.FindingSourceHistoryPrefix(receiptID)+"direct-verification.%", f.id).Scan(&assessments))
				if assessments != 0 {
					t.Fatalf("contract-invalid collection assessed the proposal %d times", assessments)
				}
			})
		}
	}
}

// collectionFaults holds the read failures a test injects into one importer.
type collectionFaults struct {
	standardRef   contracts.ArtifactRef
	standardReads int
	// standardErr fails every read of the standard while it is set.
	standardErr error
	// bindingErr fails every Run binding read while it is set.
	bindingErr error
}

type collectionFaultAccess struct {
	ArtifactAccess
	faults *collectionFaults
}

func (a collectionFaultAccess) WithClaim(claim auditstore.ControllerClaim) ArtifactAccess {
	a.ArtifactAccess = a.ArtifactAccess.WithClaim(claim)
	return a
}

func (a collectionFaultAccess) ReadProjectExact(
	ctx context.Context, projectID string, expected auditstore.ExactArtifact,
) ([]byte, error) {
	if expected.Ref.SameExact(a.faults.standardRef) {
		a.faults.standardReads++
		if a.faults.standardErr != nil {
			return nil, a.faults.standardErr
		}
	}
	return a.ArtifactAccess.ReadProjectExact(ctx, projectID, expected)
}

func (a collectionFaultAccess) ReadRunBinding(
	ctx context.Context, runID string, ref contracts.ArtifactRef,
) (auditstore.ExactArtifact, []byte, bool, error) {
	if a.faults.bindingErr != nil {
		return auditstore.ExactArtifact{}, nil, false, a.faults.bindingErr
	}
	return a.ArtifactAccess.ReadRunBinding(ctx, runID, ref)
}

// retainStandard freezes pkg in an Audit-managed Namespace of the fixture
// Project and returns its pin.
func (f *completionFixture) retainStandard(t *testing.T, pkg auditstandards.Package) auditstandards.PinnedPackage {
	t.Helper()
	scope, err := artifacts.ProjectScope(f.id)
	mustCompletion(t, err)
	written, err := f.artifacts.CreateFrozen(f.ctx, scope,
		contracts.ArtifactRef{Namespace: "audit-fixture-standards", Name: auditstandards.ArtifactName(pkg.Reference())},
		artifacts.Payload{MediaType: auditstandards.MediaType, Data: pkg.Payload()})
	mustCompletion(t, err)
	pinned := auditstandards.PinnedPackage{
		Reference: pkg.Reference(), Title: pkg.Document.Standard.Title,
		Source: pkg.Document.Standard.Source, License: pkg.Document.Standard.License,
		Retained: auditstandards.ExactPackage{
			Artifact: written.Ref, Digest: pkg.Digest, MediaType: auditstandards.MediaType,
			SizeBytes: int64(len(pkg.Payload())),
		},
	}
	pinned.Catalog = pinned.Retained
	pinned.Catalog.Artifact.Namespace = artifactpolicy.AuditStandardCatalogNamespace
	mustCompletion(t, auditstandards.ValidatePinnedPackage(pinned))
	return pinned
}

func (f *completionFixture) proposalInvocation() string { return f.id + "-submission" }

// proposal stores one finding proposal of the fixture Run with its durable
// intake receipt, as Runtime submission does, and returns the receipt ID.
func (f *completionFixture) proposal(t *testing.T, clientKey string, document auditdomain.FindingProposal) string {
	t.Helper()
	proposal, err := auditdomain.EncodeFindingProposal(document)
	mustCompletion(t, err)
	decoded, err := auditdomain.DecodeFindingProposal(proposal)
	mustCompletion(t, err)
	projection, err := auditdomain.EncodeFindingReportProjection(auditdomain.ProjectFindingReport(decoded))
	mustCompletion(t, err)
	written, err := artifactstest.WriteFindingProposal(f.ctx, f.artifacts, f.run.RunID, clientKey,
		artifacts.Payload{MediaType: "application/json", Data: proposal})
	mustCompletion(t, err)
	proposalRef, err := json.Marshal(written.Ref)
	mustCompletion(t, err)
	receiptID := f.id + "-finding-receipt-" + clientKey
	_, err = f.pool.Exec(f.ctx, `
INSERT INTO finding_proposal_receipts (
    receipt_id, proposal_id, allocation_id, runtime_agent_id, runtime_instance_id,
    stage_execution_id, logical_agent_name, invocation_id, submission_id,
    client_key, request_digest, run_id, owner_id, project_id,
    audit_execution_id, audit_id, audit_role, workflow_name, workflow_version,
    workflow_schema_version, workflow_configuration_ref, workflow_closure_digest,
    proposal_ref, proposal_digest, proposal_media_type, proposal_size_bytes, evidence,
    report_projection
) VALUES (
    $1, $2, $3, 'runtime', 'instance', 'stage', 'checker', $3, $3,
    $14, $4, $5, 'owner', $6, $7, $6, 'check', $8, $9,
    'contractor/v1alpha1', '{}', $10, $11::jsonb, $4, 'application/json', $12, '[]', $13
)`, receiptID, f.id+"-proposal-"+clientKey, f.proposalInvocation(), completionDigest(proposal),
		f.run.RunID, f.id, f.execution.ExecutionID, f.run.WorkflowName,
		f.run.WorkflowVersion, completionDigest(f.run.WorkflowSnapshot),
		string(proposalRef), len(proposal), string(projection), clientKey)
	mustCompletion(t, err)
	return receiptID
}

// writeResult publishes the Worker's result package for every manifest item.
// Each item is inconclusive and selects the proposals given for its key.
func (f *completionFixture) writeResult(t *testing.T, selections map[string][]auditdomain.ProposalSelection) {
	t.Helper()
	project, err := f.artifacts.Project(f.id)
	mustCompletion(t, err)
	stored, err := project.Read(f.ctx, f.manifest.Ref)
	mustCompletion(t, err)
	manifest, err := auditdomain.DecodeExecutionManifest(stored.Payload.Data)
	mustCompletion(t, err)
	results := make([]auditdomain.CheckResult, 0, len(manifest.Items))
	for _, item := range manifest.Items {
		proposals := selections[item.ItemKey]
		if proposals == nil {
			proposals = []auditdomain.ProposalSelection{}
		}
		results = append(results, auditdomain.CheckResult{
			ItemKey: item.ItemKey, SubjectKey: item.SubjectKey, Assessment: "inconclusive",
			Summary: "The source trace was not completed.", EvidenceIDs: []string{},
			Coverage: auditdomain.ResultCoverage{
				Requested: []string{"source-trace"}, Completed: []string{}, Gaps: []string{},
			},
			Proposals: proposals,
		})
	}
	document, err := auditdomain.EncodeCheckResultSet(auditdomain.CheckResultSet{
		Schema: auditdomain.CheckResultsSchema, ExecutionManifestDigest: f.manifest.Digest, Results: results,
	})
	mustCompletion(t, err)
	payload, _, err := auditdomain.BuildPackage("result-package", auditdomain.PackageKindCheckResults, "",
		[]auditdomain.PackageInput{{
			ID: auditdomain.CheckResultsMemberID, Path: "check-results.json",
			MediaType: auditdomain.JSONMediaType, Data: document,
		}})
	mustCompletion(t, err)
	store, err := f.artifacts.Run(f.run.RunID)
	mustCompletion(t, err)
	target := f.run.AuditCompletion.Contract.ResultArtifact
	_, err = store.Write(f.ctx, contracts.ArtifactRef{Namespace: target.Namespace, Name: target.Name},
		artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload}, nil)
	mustCompletion(t, err)
}

// requireCollecting fails unless the execution has no receipt and every
// member keeps its attempt.
func (f *completionFixture) requireCollecting(t *testing.T, executionID string) {
	t.Helper()
	var state string
	var receipts, settled int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT execution.state,
       (SELECT count(*) FROM audit_executions WHERE execution_id = $1 AND collection_receipt_id IS NOT NULL),
       (SELECT count(*) FROM audit_execution_items WHERE execution_id = $1 AND collection_disposition IS NOT NULL)
  FROM audit_executions AS execution WHERE execution.execution_id = $1`, executionID).Scan(&state, &receipts, &settled))
	if state != string(auditstore.ExecutionCollecting) || receipts != 0 || settled != 0 {
		t.Fatalf("execution state=%s receipts=%d settled members=%d, want collecting", state, receipts, settled)
	}
}

// requireContractInvalid fails unless the execution has exactly one
// contract-invalid receipt and every member settled with it.
func (f *completionFixture) requireContractInvalid(t *testing.T, executionID string) {
	t.Helper()
	var disposition, errorCode string
	var receipts, members, settled int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT execution.collection_disposition, execution.collection_error_code,
       (SELECT count(*) FROM audit_executions WHERE execution_id = $1 AND collection_receipt_id IS NOT NULL),
       (SELECT count(*) FROM audit_execution_items WHERE execution_id = $1),
       (SELECT count(*) FROM audit_execution_items AS member
          JOIN audit_items AS item ON item.item_id = member.item_id
         WHERE member.execution_id = $1 AND member.state = 'settled'
           AND member.collection_disposition = $2 AND item.final_disposition = $3)
  FROM audit_executions AS execution WHERE execution.execution_id = $1`,
		executionID, string(auditstore.CollectionContractInvalid), string(auditstore.FinalInvalidResult),
	).Scan(&disposition, &errorCode, &receipts, &members, &settled))
	if disposition != string(auditstore.CollectionContractInvalid) || errorCode != "collection-contract-invalid" ||
		receipts != 1 || members == 0 || settled != members {
		t.Fatalf("execution disposition=%s error=%s receipts=%d settled members=%d/%d, want contract-invalid",
			disposition, errorCode, receipts, settled, members)
	}
}

func (f *completionFixture) requireAcceptedAssessment(t *testing.T, executionID, receiptID string) {
	t.Helper()
	var disposition string
	var assessments int
	mustCompletion(t, f.pool.QueryRow(f.ctx, `
SELECT execution.collection_disposition,
       (SELECT count(*) FROM audit_execution_items AS member
        CROSS JOIN LATERAL jsonb_to_recordset(member.finding_sources) AS source(receipt_id text)
        WHERE member.execution_id=$1 AND source.receipt_id=$2)
  FROM audit_executions AS execution WHERE execution.execution_id = $1`, executionID, receiptID).Scan(&disposition, &assessments))
	if disposition != string(auditstore.CollectionAccepted) || assessments != 1 {
		t.Fatalf("collection disposition=%s proposal assessments=%d", disposition, assessments)
	}
}
