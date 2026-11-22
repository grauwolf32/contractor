package runservice

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestSharedPostgresCommonCompletionPinsExactRunContextAndMembership(t *testing.T) {
	for _, members := range []int{1, 2} {
		t.Run(map[int]string{1: "singleton", 2: "batch"}[members], func(t *testing.T) {
			f := newCommonTaskRunFixtureWithWorkflow(t, true, func(workflow *resolved.ResolvedWorkflow) {
				stage := workflow.Stages["check"]
				agent := stage.Agents["builder"]
				agent.Template.Summarizer = nil
				agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task", "submit_result", "submit_results", "evidence"}})
				stage.Agents["builder"] = agent
				workflow.Stages["check"] = stage
			}, members)
			ctx := t.Context()
			result, err := f.service.CreateAudit(ctx, f.params)
			if err != nil {
				t.Fatal(err)
			}
			run, err := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage()).GetRun(ctx, result.Run.RunID)
			if err != nil {
				t.Fatal(err)
			}
			completion := run.AuditCompletion
			if completion == nil || completion.Stage != "check" || completion.Agent != "builder" || completion.Contract.Kind != contracts.AuditCheckResultsV1 || completion.Contract.Context.Revision == nil || completion.Contract.Context.Name != "context" {
				t.Fatalf("completion=%+v", completion)
			}
			scoped, err := artifacts.NewService(artifactpgtest.NewStorage().Repository(f.pool)).Run(run.RunID)
			if err != nil {
				t.Fatal(err)
			}
			metadata, err := scoped.Metadata(ctx, completion.Contract.Context)
			if err != nil || !metadata.Frozen {
				t.Fatal("context is not exact and frozen", err)
			}
			payload, err := scoped.ReadExact(ctx, metadata.Exact())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := auditcontract.Decode(payload.Payload.Data, nil); err != nil {
				t.Fatal("invalid trusted context", err)
			}
			var document struct {
				RunID string `json:"run_id"`
				Items []struct {
					TaskID  string `json:"task_id"`
					ItemKey string `json:"item_key"`
				} `json:"items"`
			}
			if err := json.Unmarshal(payload.Payload.Data, &document); err != nil || document.RunID != run.RunID || len(document.Items) != members {
				t.Fatalf("membership=%+v error=%v", document, err)
			}
			for index, item := range document.Items {
				if item.TaskID != f.params.TaskIDs[index] || item.ItemKey != auditcontract.ExecutionItemKey(index) {
					t.Fatal("trusted membership differs")
				}
			}
			replay, err := f.service.CreateAudit(ctx, f.params)
			if err != nil || !replay.Replayed || !reflect.DeepEqual(replay.Run.AuditCompletion, completion) {
				t.Fatal("restart/replay changed completion", err)
			}
			changed := *completion
			changed.Contract.Context = changed.Contract.Context.Clone()
			changed.Contract.Context.Name = "foreign"
			if err := runstore.NewPostgresStore(f.pool, artifactpgtest.NewStorage()).SetAuditCompletion(ctx, run.RunID, changed); !errors.Is(err, runstore.ErrConflict) {
				t.Fatal("completion authority changed", err)
			}
			invalid := f.params
			invalid.TaskIDs = []string{"foreign-task"}
			if _, err := f.service.CreateAudit(ctx, invalid); !errors.Is(err, auditstore.ErrConflict) {
				t.Fatal("foreign membership accepted", err)
			}
		})
	}
}
