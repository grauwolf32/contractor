package auditservice

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func (s *Service) validateProfileDependencies(
	ctx context.Context,
	profile auditprofile.PinnedProgram,
	llmCredentials resolved.CredentialLookup,
) ([]string, []contracts.ArtifactRef, [][]string, error) {
	credentialSets := make([][]string, 0, len(profile.Workflows))
	skillsByName := make(map[string]contracts.ArtifactRef)
	sets := make([][]string, 0)
	roles := make([]string, 0, len(profile.Workflows))
	for role := range profile.Workflows {
		roles = append(roles, role)
	}
	sort.Strings(roles)
	for _, role := range roles {
		workflow := profile.Workflows[role]
		if err := resolved.ValidateResolvedWorkflowCredentials(ctx, workflow, llmCredentials); err != nil {
			// A lookup outage keeps its cause: recovery and transaction
			// conflicts stay retryable and reach the start replay check.
			if errors.Is(err, resolved.ErrCredentialLookup) {
				return nil, nil, nil, err
			}
			return nil, nil, nil, fmt.Errorf(
				"%w: Audit Workflow role %q credential selection is not runnable: %v", ErrInvalid, role, err,
			)
		}
		credentialSets = append(credentialSets, resolved.ResolvedWorkflowCredentialIDs(workflow))
		refs, err := resolved.WorkflowSkillRefs(workflow)
		if err != nil {
			return nil, nil, nil, err
		}
		for _, ref := range refs {
			skillsByName[ref.Name] = ref
		}
		sets = append(sets, resolved.WorkflowSkillSets(workflow)...)
	}
	credentialNames := mergeIDs(credentialSets...)
	names := make([]string, 0, len(skillsByName))
	for name := range skillsByName {
		names = append(names, name)
	}
	sort.Strings(names)
	refs := make([]contracts.ArtifactRef, 0, len(names))
	for _, name := range names {
		refs = append(refs, skillsByName[name])
	}
	return credentialNames, refs, sets, nil
}

func (s *Service) startedProjection(_ context.Context, audit auditstore.Audit, replayed bool) (StartedAudit, error) {
	return StartedAudit{Audit: audit, Replayed: replayed}, nil
}
func timeDurationSeconds(value int) time.Duration { return time.Duration(value) * time.Second }
