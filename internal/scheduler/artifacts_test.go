package scheduler

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// recordingForkRepository records the forks that reach storage; any other
// Repository method panics on the nil embedded interface.
type recordingForkRepository struct {
	artifacts.Repository
	forks []artifacts.Fork
}

func (r *recordingForkRepository) Fork(_ context.Context, fork artifacts.Fork) (artifacts.ForkResult, error) {
	r.forks = append(r.forks, fork)
	return artifacts.ForkResult{}, nil
}

// The Scheduler binds and publishes only exact Run output revisions
// (ART-R90, ART-R99): a versionless source or a publication of another slot
// never reaches the store, and the forks that do carry the output rules. Only
// the output binding replays an identical source; publication only creates.
func TestOutputForksBindAndPublishOnlyExactOutputs(t *testing.T) {
	ctx := context.Background()
	repository := &recordingForkRepository{}
	service := artifacts.NewService(repository)
	revision := "revision-1"
	if _, err := bindOutput(ctx, service, "run-1", "result",
		contracts.ArtifactRef{Namespace: "builder", Name: "result"}); !errors.Is(err, artifacts.ErrExactRevisionRequired) {
		t.Fatalf("versionless output binding = %v", err)
	}
	for _, source := range []contracts.ArtifactRef{
		{Namespace: "outputs", Name: "openapi"},
		{Namespace: "builder", Name: "openapi", Revision: &revision},
		{Namespace: "outputs", Name: "other", Revision: &revision},
	} {
		if _, err := publishRunOutput(ctx, service, "run-1", "project-1", "openapi", source); err == nil {
			t.Errorf("publication of %+v was accepted", source)
		}
	}
	if len(repository.forks) != 0 {
		t.Fatalf("invalid output forks reached the store: %+v", repository.forks)
	}

	result := contracts.ArtifactRef{Namespace: "builder", Name: "result", Revision: &revision}
	if _, err := bindOutput(ctx, service, "run-1", "result", result); err != nil {
		t.Fatal(err)
	}
	output := contracts.ArtifactRef{Namespace: "outputs", Name: "openapi", Revision: &revision}
	if _, err := publishRunOutput(ctx, service, "run-1", "project-1", "openapi", output); err != nil {
		t.Fatal(err)
	}
	run, _ := artifacts.RunScope("run-1")
	project, _ := artifacts.ProjectScope("project-1")
	if len(repository.forks) != 2 {
		t.Fatalf("output forks = %+v", repository.forks)
	}
	bound, published := repository.forks[0], repository.forks[1]
	if bound.Kind != artifacts.LineageOutputBind || bound.Source != run || bound.Target != run ||
		!bound.SourceRef.SameExact(result) || bound.TargetRef != (contracts.ArtifactRef{Namespace: "outputs", Name: "result"}) ||
		bound.Frozen || bound.SourceFrozen || !bound.Replay {
		t.Fatalf("output binding fork = %+v", bound)
	}
	if published.Kind != artifacts.LineageProjectOutputPublish || published.Source != run || published.Target != project ||
		!published.SourceRef.SameExact(output) || !published.SourceFrozen ||
		published.TargetRef != (contracts.ArtifactRef{Namespace: "outputs", Name: "openapi"}) ||
		published.Frozen || published.Replay {
		t.Fatalf("Project publication fork = %+v", published)
	}
}

// The memory double of the result commit binds each Workflow output once, as
// the store does (ART-R90): the binding of the candidate's own source is
// replayed, while one of another revision, or a candidate source revision
// that is gone when the acceptance commits, refuses the acceptance as invalid
// output state, which ends the Run without replacing the binding.
func TestMemoryResultCommitBindsEachOutputOnce(t *testing.T) {
	for _, test := range []struct {
		name          string
		existing      string
		sourceRemoved bool
		state         runstore.WorkflowRunState
		reason        string
	}{
		{"candidate source", "output-result-r1", false, runstore.RunSucceeded, "workflow_succeeded"},
		{"another revision", "output-earlier", false, runstore.RunFailed, "scheduler_state_invalid"},
		{"source gone", "", true, runstore.RunFailed, "scheduler_state_invalid"},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := newSchedulerHarness(t)
			var existing contracts.ArtifactRef
			if test.existing != "" {
				existing = exactRef("outputs", "result", test.existing)
				h.persistence.outputs["result"] = existing
			}
			if test.sourceRemoved {
				h.scheduler.persistence = &removeSourceBeforeResultCommit{
					memoryAtomicPersistence: h.persistence, artifacts: h.artifacts,
					key: "run-1/builder/copied/result-r1",
				}
			}
			if worked, err := h.scheduler.RunOnce(t.Context()); err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			if h.store.run.State != test.state || h.store.run.StateReason.Code != test.reason {
				t.Fatalf("Run = %s (%+v), want %s (%s)", h.store.run.State, h.store.run.StateReason, test.state, test.reason)
			}
			if len(h.store.stages) != 1 || h.store.stages[0].State != runstore.StageSucceeded ||
				h.store.stages[0].AcceptedResult == nil {
				t.Fatalf("StageExecutions = %+v, want the candidate accepted", h.store.stages)
			}
			want := map[string]contracts.ArtifactRef{}
			if test.existing != "" {
				want["result"] = existing
			}
			if !reflect.DeepEqual(h.persistence.outputs, want) {
				t.Fatalf("outputs = %+v, want %+v", h.persistence.outputs, want)
			}
		})
	}
}

// removeSourceBeforeResultCommit removes one artifact revision from the
// memory store after the candidate was validated, just before its acceptance
// commits.
type removeSourceBeforeResultCommit struct {
	*memoryAtomicPersistence
	artifacts *memoryArtifactResolver
	key       string
}

func (p *removeSourceBeforeResultCommit) CommitResultProgression(ctx context.Context, value ResultProgression) error {
	delete(p.artifacts.values, p.key)
	return p.memoryAtomicPersistence.CommitResultProgression(ctx, value)
}
