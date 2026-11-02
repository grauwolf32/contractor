package runstore

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// DescribeOriginalInputs returns immutable Run-creation input descriptors in
// requested name order, omitting absent inputs. Editable current bindings never
// replace the creation revisions. The shared artifact reader bounds the request
// and uses this store's DB/transaction for metadata-only reads.
func (s *PostgresStore) DescribeOriginalInputs(ctx context.Context, runID string, names []string) ([]contracts.ExactArtifact, error) {
	store, err := artifacts.NewService(artifactpostgres.NewRepository(s.db)).Run(runID)
	if err != nil {
		return nil, err
	}
	return store.DescribeForkRevisions(ctx, "inputs", names, artifacts.LineageInputFork)
}

// OriginalInputSources reads exact creation sources for a Run's pinned Workflow.
// Current editable input revisions never select or replace these references.
// The supplied RunScope retains the caller's repository or transaction.
func OriginalInputSources(ctx context.Context, run WorkflowRun, store artifacts.ScopedStore) (map[string]artifacts.LineageEdge, error) {
	workflow, err := resolved.DecodeResolvedWorkflowSnapshot(run.WorkflowSnapshot)
	if err != nil || workflow.Ref.Name != run.WorkflowName || workflow.Ref.Version != run.WorkflowVersion {
		return nil, fmt.Errorf("%w: invalid Run Workflow snapshot", blob.ErrIntegrity)
	}
	names := slices.Sorted(maps.Keys(workflow.Inputs))
	result := make(map[string]artifacts.LineageEdge, len(names))
	for start := 0; start < len(names); start += artifacts.MaxExactReadBatchSize {
		edges, err := store.DescribeForkSources(ctx, "inputs", names[start:min(start+artifacts.MaxExactReadBatchSize, len(names))], artifacts.LineageInputFork)
		if err != nil {
			return nil, err
		}
		for _, edge := range edges {
			slot := workflow.Inputs[edge.Target.Name]
			expected, scopeErr := artifacts.UserScope(run.OwnerID)
			if run.ProjectID != nil && slot.SourceScope != "user" {
				expected, scopeErr = artifacts.ProjectScope(*run.ProjectID)
			}
			if scopeErr != nil || edge.TargetScope != artifacts.ScopeRun || edge.TargetScopeID != run.RunID ||
				edge.SourceScope != expected.Kind() || edge.SourceScopeID != expected.ID() || edge.Source.ValidateExact() != nil {
				return nil, fmt.Errorf("%w: input %s has an invalid source scope", blob.ErrIntegrity, edge.Target.Name)
			}
			result[edge.Target.Name] = edge
		}
	}
	for name, slot := range workflow.Inputs {
		if _, exists := result[name]; !exists && slot.Required {
			return nil, fmt.Errorf("%w: input %s has no creation fork", blob.ErrIntegrity, name)
		}
	}
	return result, nil
}
