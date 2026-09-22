//go:build integration

package auditcontroller

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// A child Run that proposed a finding may end after its Audit started to
// close. Its collection must still retain the proposal, and the Audit must
// settle with that finding visible.
func TestPostgresClosingAuditRetainsLateChildFindingProposal(t *testing.T) {
	for _, closing := range []auditstore.AuditState{auditstore.AuditCancelling, auditstore.AuditFinalizing} {
		t.Run(string(closing), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			harness := newPostgresControllerHarnessWithConfig(
				t, ctx, 0, 1, loadControllerConfigWithFindings(t, 1, 10, true),
			)
			intake, err := findingintake.New(harness.pool)
			if err != nil {
				t.Fatal(err)
			}
			controller := harness.controllerWithCollector(t, intake)
			for step := 0; step < 2; step++ {
				if worked, err := controller.RunOnce(ctx); err != nil || !worked {
					t.Fatalf("dispatch step %d = (%t, %v)", step, worked, err)
				}
			}
			executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
			if err != nil || len(executions) != 1 || executions[0].RunID == nil {
				t.Fatalf("submitted child = (%+v, %v)", executions, err)
			}
			runID := *executions[0].RunID
			submitChildFindingProposal(
				t, ctx, intake, childFindingGrant(t, ctx, harness, runID), "late-candidate", nil,
			)

			current, err := harness.audits.Get(ctx, harness.started.Audit.OwnerID, harness.started.Audit.AuditID)
			if err != nil {
				t.Fatal(err)
			}
			target := auditstore.AuditCancelled
			if closing == auditstore.AuditCancelling {
				service, err := auditservice.New(auditservice.Options{
					Pool: harness.pool, Profiles: harness.snapshot,
					TransactionLLMCredentials: controllerTransactionCredentialLookup(),
					CredentialGuard:           controllerCredentialGuard{},
				})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := service.Cancel(ctx, auditservice.MutationParams{
					OwnerID: current.OwnerID, AuditID: current.AuditID, ExpectedRevision: current.Revision,
					IdempotencyKey: "cancel-closing-audit", RequestDigest: postgresDigest("cancel-closing-audit"),
				}); err != nil {
					t.Fatal(err)
				}
				if worked, err := controller.RunOnce(ctx); err != nil || !worked {
					t.Fatalf("cancel child = (%t, %v)", worked, err)
				}
				if _, err := harness.runs.TransitionRun(
					ctx, runID, runstore.RunCancelling, runstore.RunCancelled,
					runstore.Reason{Code: "test_cancelled"},
				); err != nil {
					t.Fatal(err)
				}
			} else {
				target = auditstore.AuditCompleted
				claims, err := harness.audits.Claim(ctx, auditstore.ClaimParams{
					HolderID: "closing-test", Lease: 5 * time.Second, Limit: 1,
				})
				if err != nil || len(claims) != 1 {
					t.Fatalf("claim Audit = (%+v, %v)", claims, err)
				}
				if _, err := harness.audits.TransitionClaimed(ctx, auditstore.ClaimedTransitionParams{
					Claim: claims[0], ExpectedRevision: current.Revision,
					ExpectedState: auditstore.AuditActive, TargetState: auditstore.AuditFinalizing,
					Reason: &auditstore.StopReason{
						Code: "submission_budget_exhausted", Message: "The test closed dispatch.",
					},
				}); err != nil {
					t.Fatal(err)
				}
				if err := harness.audits.ReleaseClaim(ctx, claims[0]); err != nil {
					t.Fatal(err)
				}
				if _, err := harness.runs.TransitionRun(
					ctx, runID, runstore.RunRunning, runstore.RunFailed,
					runstore.Reason{Code: "test_failed"},
				); err != nil {
					t.Fatal(err)
				}
			}

			for operation := 0; operation < 16; operation++ {
				audit, err := harness.audits.Get(ctx, current.OwnerID, current.AuditID)
				if err != nil {
					t.Fatal(err)
				}
				if audit.State == target {
					break
				}
				worked, err := controller.RunOnce(ctx)
				if err != nil || !worked {
					t.Fatalf("settle operation %d = (%t, %v), Audit state %s", operation, worked, err, audit.State)
				}
			}
			settled, err := harness.audits.Get(ctx, current.OwnerID, current.AuditID)
			if err != nil || settled.State != target {
				t.Fatalf("settled Audit = (%+v, %v)", settled, err)
			}
			var holds, findings int
			if err := harness.pool.QueryRow(ctx, `
SELECT (SELECT count(*) FROM finding_proposal_audit_holds WHERE audit_id = $1),
       (SELECT count(*) FROM audit_findings WHERE audit_id = $1 AND state = 'proposed')`,
				settled.AuditID).Scan(&holds, &findings); err != nil {
				t.Fatal(err)
			}
			if holds != 1 || findings != 1 {
				t.Fatalf("retained holds = %d, proposed findings = %d", holds, findings)
			}
			if target == auditstore.AuditCompleted {
				service, err := auditservice.New(auditservice.Options{
					Pool: harness.pool, Profiles: harness.snapshot,
					TransactionLLMCredentials: controllerTransactionCredentialLookup(),
					CredentialGuard:           controllerCredentialGuard{},
				})
				if err != nil {
					t.Fatal(err)
				}
				report, err := service.GetReport(ctx, settled.OwnerID, settled.AuditID)
				if err != nil || report.Status != auditservice.ReportReady {
					t.Fatalf("Audit report = (%+v, %v)", report, err)
				}
				var machine struct {
					Findings struct {
						Proposed []struct {
							Title string `json:"title"`
						} `json:"proposed"`
					} `json:"findings"`
				}
				if err := json.Unmarshal(report.Machine, &machine); err != nil ||
					len(machine.Findings.Proposed) != 1 || machine.Findings.Proposed[0].Title != "Late candidate" {
					t.Fatalf("report proposed findings = (%+v, %v)", machine.Findings, err)
				}
			}
		})
	}
}

// One model-authored proposal with an invalid standard reference must not
// poison its collection: only that proposal is rejected, its siblings stay
// retained, and the next-round inbox lists only exactly held receipts.
func TestPostgresCollectionRejectsOnlyInvalidChildFindingProposal(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newPostgresControllerHarnessWithConfig(t, ctx, 1, loadControllerConfigWithFindings(t, 1, true))
	intake, err := findingintake.New(harness.pool)
	if err != nil {
		t.Fatal(err)
	}
	controller := harness.controllerWithCollector(t, intake)
	for step := 0; step < 2; step++ {
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("dispatch step %d = (%t, %v)", step, worked, err)
		}
	}
	executions, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
	if err != nil || len(executions) != 1 || executions[0].RunID == nil {
		t.Fatalf("submitted child = (%+v, %v)", executions, err)
	}
	runID := *executions[0].RunID
	grant := childFindingGrant(t, ctx, harness, runID)
	invalid := submitChildFindingProposal(t, ctx, intake, grant, "invalid-candidate",
		[]auditdomain.StandardReference{{Scheme: "unpinned", Version: "1", RequirementID: "invented"}})
	valid := submitChildFindingProposal(t, ctx, intake, grant, "valid-candidate", nil)
	if _, err := harness.runs.TransitionRun(
		ctx, runID, runstore.RunRunning, runstore.RunFailed, runstore.Reason{Code: "test_failed"},
	); err != nil {
		t.Fatal(err)
	}
	for operation := 0; operation < 4; operation++ {
		collected, err := harness.audits.ListExecutions(ctx, harness.started.Audit.AuditID)
		if err != nil {
			t.Fatal(err)
		}
		if collected[0].State == auditstore.ExecutionCollected {
			break
		}
		if worked, err := controller.RunOnce(ctx); err != nil || !worked {
			t.Fatalf("collection operation %d = (%t, %v)", operation, worked, err)
		}
	}
	var disposition string
	if err := harness.pool.QueryRow(ctx, `
SELECT disposition FROM audit_collection_receipts WHERE execution_id = $1`,
		executions[0].ExecutionID).Scan(&disposition); err != nil || disposition != "execution-failed" {
		t.Fatalf("collection disposition = (%q, %v)", disposition, err)
	}
	if err := intake.RejectAuditCollection(ctx, findingintake.ImportRequest{
		OwnerID: harness.started.Audit.OwnerID, AuditID: harness.started.Audit.AuditID,
		RunID: runID, Proposal: invalid.Proposal.Ref,
	}, "finding-proposal-standard-invalid"); err != nil {
		t.Fatalf("replayed rejection: %v", err)
	}
	var heldReceipt, rejectedReceipt, reason string
	if err := harness.pool.QueryRow(ctx, `
SELECT (SELECT string_agg(receipt_id, ',') FROM finding_proposal_audit_holds WHERE audit_id = $1),
       (SELECT string_agg(entity_id, ',') FROM audit_events
         WHERE audit_id = $1 AND kind = 'finding.proposal_rejected'),
       (SELECT string_agg(summary->>'reason', ',') FROM audit_events
         WHERE audit_id = $1 AND kind = 'finding.proposal_rejected')`,
		harness.started.Audit.AuditID).Scan(&heldReceipt, &rejectedReceipt, &reason); err != nil {
		t.Fatal(err)
	}
	if heldReceipt != valid.ReceiptID || rejectedReceipt != invalid.ReceiptID ||
		reason != "finding-proposal-standard-invalid" {
		t.Fatalf("held=%q rejected=%q reason=%q", heldReceipt, rejectedReceipt, reason)
	}
	query := findingintake.ListQuery{Limit: 10}
	owner, auditID := harness.started.Audit.OwnerID, harness.started.Audit.AuditID
	inbox, err := intake.ListAuditInbox(ctx, owner, auditID, query)
	if err != nil || len(inbox) != 2 {
		t.Fatalf("owner inbox = (%d, %v)", len(inbox), err)
	}
	held, err := intake.ListAuditHeldInbox(ctx, owner, auditID, query)
	if err != nil || len(held) != 1 || held[0].ReceiptID != valid.ReceiptID {
		t.Fatalf("held inbox = (%+v, %v)", held, err)
	}
}

func childFindingGrant(
	t *testing.T, ctx context.Context, harness *postgresControllerHarness, runID string,
) controlplane.AllocationGrant {
	t.Helper()
	run, err := harness.runs.GetRun(ctx, runID)
	if err != nil {
		t.Fatal(err)
	}
	if run.State == runstore.RunPending {
		if _, err := harness.runs.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning,
			runstore.Reason{Code: "test_started"},
		); err != nil {
			t.Fatal(err)
		}
	}
	workflow, err := config.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	stageID, allocationID := runID+"-stage", runID+"-allocation"
	if _, err := harness.runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	binding := stage.Agents["worker"]
	runtimeAgentID := strings.Repeat("a", 64)
	gateway := contracts.LLMGatewayConfigRef{
		GatewayID: "local-litellm", Version: "1", Digest: "sha256:" + strings.Repeat("b", 64),
	}
	if err := harness.runs.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageID,
		LogicalAgentName: "worker", Namespace: binding.Namespace,
		AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: "closing-instance",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration: &runstore.AllocationRuntimeConfiguration{
			ModelPolicy: contracts.ModelPolicyRef{
				PolicyID: "worker", Version: "1", Digest: "sha256:" + strings.Repeat("c", 64),
			},
			Origins: runtimeconfig.ResolvedRuntimeConfigOrigins{
				LLMGateway: &runtimeconfig.RuntimeFieldOrigin{Layer: runtimeconfig.LayerWorkflow},
			},
			Provenance: contracts.ResolvedRuntimeConfigProvenance{
				Default: contracts.RuntimeLabelBindingProvenance{
					Label: "default", BindingRevision: 1,
					Config: contracts.RuntimeConfigRef{
						Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
						Digest: runtimeconfig.BuiltInDigest,
					},
				},
				RunLabels:       []contracts.RuntimeLabelBindingProvenance{},
				AgentLabels:     []contracts.RuntimeLabelBindingProvenance{},
				RuntimeAdapters: []contracts.RuntimeAdapterRef{}, LLMGatewayConfig: &gateway,
				RuntimeCredentialRefs: []contracts.RuntimeCredentialRef{},
			},
		},
		PerformanceCollectionPolicy: contracts.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	return controlplane.AllocationGrant{
		AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
		RuntimeInstanceID: "closing-instance", RunID: runID, StageExecutionID: stageID,
		LogicalAgentName: "worker", Namespace: binding.Namespace,
	}
}

func submitChildFindingProposal(
	t *testing.T, ctx context.Context, intake *findingintake.Service,
	grant controlplane.AllocationGrant, clientKey string,
	standardRefs []auditdomain.StandardReference,
) findingintake.Receipt {
	t.Helper()
	if standardRefs == nil {
		standardRefs = []auditdomain.StandardReference{}
	}
	invocationID := clientKey + "-invocation"
	receipt, _, err := intake.Submit(ctx, grant, findingintake.Submission{
		APIVersion: findingintake.APIVersion, InvocationID: invocationID,
		SubmissionID: findingintake.StableSubmissionID(invocationID, clientKey),
		Proposal: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: clientKey,
			Title: "Late candidate", Description: "A candidate found before the Audit closed.",
			Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: standardRefs,
			EvidenceIDs: []string{}, ProposedChecks: []auditdomain.ProposedCheck{},
			Limitations: []string{},
		},
		EvidenceRefs: []contracts.ArtifactRef{},
	})
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}
