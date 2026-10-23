//go:build integration

package findingintake

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/auditstoretest"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/controlplane"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// A storage failure while reading the submitting Run, its StageExecution or
// the Audit provenance of an Audit-managed Run is not a denied grant or a
// deselected finding tool: the submission keeps the database cause, commits
// nothing and succeeds once the store answers again. Each case breaks exactly
// one read; the reads before it still succeed.
func TestPostgresFindingSubmissionKeepsProvenanceLookupFailure(t *testing.T) {
	for _, test := range []struct {
		name, breakRead, restoreRead, sqlState string
		auditManaged                           bool
	}{
		{
			name:        "Run",
			breakRead:   `ALTER TABLE workflow_runs RENAME TO lookup_unavailable_runs`,
			restoreRead: `ALTER TABLE lookup_unavailable_runs RENAME TO workflow_runs`,
			sqlState:    persistencepostgres.SQLStateUndefinedTable,
		},
		{
			// The allocation check joins stage_executions on its keys only;
			// the StageExecution read is the first to select the snapshot.
			name:        "StageExecution",
			breakRead:   `ALTER TABLE stage_executions RENAME COLUMN stage_spec_snapshot TO lookup_unavailable_snapshot`,
			restoreRead: `ALTER TABLE stage_executions RENAME COLUMN lookup_unavailable_snapshot TO stage_spec_snapshot`,
			sqlState:    persistencepostgres.SQLStateUndefinedColumn,
		},
		{
			// Only an Audit-managed Run reads its Audit execution, last.
			name:         "Audit",
			breakRead:    `ALTER TABLE audit_executions RENAME TO lookup_unavailable_audit_executions`,
			restoreRead:  `ALTER TABLE lookup_unavailable_audit_executions RENAME TO audit_executions`,
			sqlState:     persistencepostgres.SQLStateUndefinedTable,
			auditManaged: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			pool := isolatedFindingPool(t, ctx)
			const ownerID, projectID, runID = "lookup-owner", "lookup-project", "lookup-run"
			if _, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: projectID, OwnerID: ownerID, Kind: projectstore.KindProject,
				Name: "Lookup project", IdempotencyKey: "lookup-project",
				RequestDigest: auditdomain.DigestBytes([]byte("lookup-project")),
			}); err != nil {
				t.Fatal(err)
			}
			var grant controlplane.AllocationGrant
			var auditOrigin *AuditOrigin
			if test.auditManaged {
				grant, auditOrigin = auditSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
			} else {
				grant = quotaSubmissionGrant(t, ctx, pool, ownerID, projectID, runID)
			}
			service, err := New(pool)
			if err != nil {
				t.Fatal(err)
			}
			input := testSubmission("lookup-invocation", "lookup-candidate", []contracts.ArtifactRef{})

			if _, err := pool.Exec(ctx, test.breakRead); err != nil {
				t.Fatal(err)
			}
			_, _, err = service.Submit(ctx, grant, input)
			if errors.Is(err, ErrAccessDenied) || errors.Is(err, ErrToolNotSelected) ||
				persistencepostgres.SQLState(err) != test.sqlState {
				t.Fatalf("%s lookup failure = %v, want the SQLSTATE %s cause", test.name, err, test.sqlState)
			}
			var receipts int
			if err := pool.QueryRow(ctx, `SELECT count(*) FROM finding_proposal_receipts`).Scan(&receipts); err != nil || receipts != 0 {
				t.Fatalf("receipts after a failed lookup = (%d, %v), want none", receipts, err)
			}
			if _, err := pool.Exec(ctx, test.restoreRead); err != nil {
				t.Fatal(err)
			}

			receipt, replayed, err := service.Submit(ctx, grant, input)
			if err != nil || replayed || receipt.Origin.RunID != runID ||
				receipt.Origin.StageExecutionID != grant.StageExecutionID {
				t.Fatalf("submission after recovery = (%+v, %t, %v)", receipt, replayed, err)
			}
			if (auditOrigin == nil) != (receipt.Origin.Audit == nil) ||
				auditOrigin != nil && *receipt.Origin.Audit != *auditOrigin {
				t.Fatalf("Audit origin after recovery = %+v, want %+v", receipt.Origin.Audit, auditOrigin)
			}
		})
	}
}

// auditSubmissionGrant creates runID as an Audit-managed check Run bound to a
// one-item Audit execution, with a running StageExecution whose builder
// selects the finding tool, and returns the builder's allocation grant and the
// Audit origin intake must record for it.
func auditSubmissionGrant(
	t *testing.T, ctx context.Context, pool *pgxpool.Pool, ownerID, projectID, runID string,
) (controlplane.AllocationGrant, *AuditOrigin) {
	t.Helper()
	snapshot, err := configload.Load("../config/testdata/valid")
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	binding := stage.Agents["builder"]
	binding.Template.Toolsets = append(binding.Template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"},
		Tools: []string{"finding"},
	})
	stage.Agents["builder"] = binding
	workflow.Stages[workflow.EntryStage] = stage
	workflowJSON, err := json.Marshal(workflow)
	if err != nil {
		t.Fatal(err)
	}
	stageJSON, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}

	digest := func(value string) string { return auditdomain.DigestBytes([]byte(runID + "-" + value)) }
	exact := func(name string) auditstore.ExactArtifact {
		revision := name + "-r1"
		return auditstore.ExactArtifact{
			Ref:    contracts.ArtifactRef{Namespace: "audit-test", Name: name, Revision: &revision},
			Digest: digest(name),
		}
	}
	audits := auditstoretest.NewStore(pool)
	audit, _, err := audits.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: runID + "-audit", OwnerID: ownerID, ProjectID: projectID,
		Profile: auditstore.ProfileIdentity{Name: "profile", Version: "1", Digest: digest("profile")},
		ProfileSnapshot: json.RawMessage(
			`{"interaction":{"findingConfirmation":"human-required"},"workflows":{"check":{"kind":"check"}}}`,
		),
		InputSelection: json.RawMessage(`{}`),
		Limits: auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1,
			MaxSubmittedRuns: 1, MaxItemRunAttempts: 1, MaxEvidenceBytes: 1 << 20},
		IdempotencyKey: runID + "-audit", RequestDigest: digest("audit"),
	})
	if err != nil {
		t.Fatal(err)
	}
	manifest, task := exact("manifest"), exact("task")
	roundID, itemID := runID+"-round", runID+"-item"
	if _, _, err := audits.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: ownerID, AuditID: audit.AuditID, ExpectedRevision: audit.Revision,
		RoundID: roundID, RoundOrdinal: 1, Manifest: manifest,
		BaselineSnapshot: json.RawMessage(`{"baseline":"test"}`), DeadlineAt: time.Now().Add(time.Hour),
		Items: []auditstore.MaterializedItem{{
			ItemID: itemID, ItemKey: "check", Ordinal: 0, Kind: "test", SubjectKey: "subject",
			Task: task, WorkflowRole: "check", InitialState: auditstore.ItemReady,
			Origin: auditstore.ItemOrigin{
				Schema: auditstore.ItemOriginSchema, SourceRef: &task.Ref,
				SourceContentDigest: digest("source"), SourceMediaType: "application/json",
				CanonicalInventoryDigest: digest("inventory"), EntryKey: "check",
			},
			Coverage: auditstore.Coverage{
				Status: auditstore.CoverageNotTested, Requested: []string{}, Completed: []string{}, Gaps: []string{},
			},
		}},
		IdempotencyKey: runID + "-start", RequestDigest: digest("start"),
	}); err != nil {
		t.Fatal(err)
	}
	claims, err := audits.Claim(ctx, auditstore.ClaimParams{HolderID: runID + "-controller", Lease: time.Minute, Limit: 1})
	if err != nil || len(claims) != 1 {
		t.Fatalf("claim Audit = (%+v, %v)", claims, err)
	}
	claim := claims[0]
	if _, err := audits.TransitionRound(ctx, auditstore.RoundTransitionParams{
		Claim: claim, RoundID: roundID, ExpectedRevision: 1,
		ExpectedState: auditstore.RoundAccepted, TargetState: auditstore.RoundExecuting,
	}); err != nil {
		t.Fatal(err)
	}
	execution, _, err := audits.CreateExecutionIntent(ctx, auditstore.CreateExecutionIntentParams{
		Claim: claim, ExecutionID: runID + "-execution", RoundID: &roundID,
		Role: auditstore.ExecutionCheck, WorkflowRole: "check", Manifest: manifest,
		SubmissionKey: runID + "-submission", RequestDigest: digest("execution"),
		Members: []auditstore.ExecutionMemberIntent{{
			ExecutionItemID: runID + "-member", ItemID: itemID, BatchOrdinal: 0, ItemAttempt: 1,
			Task: task, Inputs: []auditstore.ExactArtifact{},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	project := projectID
	if err := persistencepostgres.InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		runs := runstore.NewPostgresStore(tx)
		if _, err := runs.CreateAuditRun(ctx, runstore.CreateAuditRunParams{
			CreateRunParams: runstore.CreateRunParams{
				LaneClass: resolved.LaneClassModel,
				RunID:     runID, OwnerID: ownerID, ProjectID: &project,
				WorkflowName: workflow.Ref.Name, WorkflowVersion: workflow.Ref.Version,
				WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: workflowJSON,
				Parameters: map[string]string{}, MetadataLabels: runstore.RunMetadataLabels{},
				RuntimeConfig: runtimeconfigtest.BuiltInRunSnapshot(),
			},
			AuditExecutionID: execution.ExecutionID, AuditSubmissionKey: execution.SubmissionKey,
		}); err != nil {
			return err
		}
		if _, err := auditstoretest.NewStore(tx).BindRun(ctx, auditstore.BindRunParams{
			Claim: claim, ExecutionID: execution.ExecutionID, RunID: runID,
		}); err != nil {
			return err
		}
		if _, err := runs.TransitionRun(
			ctx, runID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"},
		); err != nil {
			return err
		}
		_, err := runs.TransitionRun(
			ctx, runID, runstore.RunPending, runstore.RunRunning, runstore.Reason{Code: "admitted"},
		)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	runs := runstore.NewPostgresStore(pool)
	stageID, allocationID := runID+"-stage", runID+"-allocation"
	if _, err := runs.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassModel,
		StageExecutionID: stageID, RunID: runID, StageName: workflow.EntryStage, Attempt: 1,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: stageJSON,
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	runtimeAgentID := strings.Repeat("a", 64)
	if err := runs.RecordStageAllocation(ctx, runstore.StageAllocation{
		AllocationID: allocationID, StageExecutionID: stageID,
		LogicalAgentName: "builder", Namespace: binding.Namespace,
		AgentTemplateRef: binding.Template.Ref, WorkerRuntimeRef: binding.Template.Runtime,
		RuntimeAgentID: runtimeAgentID, RuntimeAgentInstanceID: "audit-instance",
		RuntimeAgentLabelRevision:         1,
		RuntimeConfigurationSchemaVersion: runstore.AllocationRuntimeConfigurationSchemaVersion,
		RuntimeConfiguration:              findingRuntimeConfiguration(),
		PerformanceCollectionPolicy:       reporting.PerformanceCollectionDisabled,
	}); err != nil {
		t.Fatal(err)
	}
	return controlplane.AllocationGrant{
			AllocationID: allocationID, RuntimeAgentID: runtimeAgentID,
			RuntimeInstanceID: "audit-instance", RunID: runID, StageExecutionID: stageID,
			LogicalAgentName: "builder", Namespace: binding.Namespace,
		}, &AuditOrigin{
			AuditID: audit.AuditID, ExecutionID: execution.ExecutionID, Role: string(auditstore.ExecutionCheck),
		}
}
