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
			submitChildFindingProposal(t, ctx, harness, intake, runID, "late-candidate")

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

func submitChildFindingProposal(
	t *testing.T, ctx context.Context, harness *postgresControllerHarness,
	intake *findingintake.Service, runID, clientKey string,
) findingintake.Receipt {
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
	invocationID := clientKey + "-invocation"
	receipt, _, err := intake.Submit(ctx, controlplane.AllocationGrant{
		AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
		RuntimeInstanceID: "closing-instance", RunID: runID, StageExecutionID: stageID,
		LogicalAgentName: "worker", Namespace: binding.Namespace,
	}, findingintake.Submission{
		APIVersion: findingintake.APIVersion, InvocationID: invocationID,
		SubmissionID: findingintake.StableSubmissionID(invocationID, clientKey),
		Proposal: auditdomain.FindingProposal{
			Schema: auditdomain.FindingProposalSchema, ClientKey: clientKey,
			Title: "Late candidate", Description: "A candidate found before the Audit closed.",
			Subject:       &auditdomain.FindingSubject{Kind: "code", Key: "handler"},
			Preconditions: []string{}, StandardRefs: []auditdomain.StandardReference{},
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
