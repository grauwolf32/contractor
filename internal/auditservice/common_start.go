package auditservice

import (
	"context"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

// Start freezes the ordinary execution dependencies and activates the accepted
// program. It performs no model/domain work and creates neither a Run nor a
// synthetic first Round. The controller admits authored work after activation.
func (s *Service) Start(ctx context.Context, p StartParams) (StartedAudit, error) {
	if p.OwnerID == "" || !contracts.ValidResourceID(p.AuditID) || p.ExpectedRevision == 0 || p.IdempotencyKey == "" || !contentdigest.Valid(p.RequestDigest) {
		return StartedAudit{}, ErrInvalid
	}
	if err := validateDeadlineSeconds(p.DeadlineSeconds); err != nil {
		return StartedAudit{}, err
	}
	if replay, found, err := s.owners().LookupMutationReplay(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest); err != nil {
		return StartedAudit{}, err
	} else if found {
		return s.startedProjection(ctx, replay, true)
	}
	var result StartedAudit
	err := s.credentialGuard.WithRunCreation(ctx, func() error {
		return postgres.InTxWithRetry(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			owners, reads := ownerstore.NewTx(tx, s.storage), querystore.New(tx, s.storage)
			if err := owners.LockMutation(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey); err != nil {
				return err
			}
			if replay, found, err := owners.LookupMutationReplay(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest); err != nil {
				return err
			} else if found {
				result = StartedAudit{Audit: replay, Replayed: true}
				return nil
			}
			audit, err := reads.Get(ctx, p.OwnerID, p.AuditID)
			if err != nil {
				return err
			}
			project, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(ctx, p.OwnerID, audit.ProjectID)
			if err != nil {
				return err
			}
			audit, err = owners.LockForStartStaging(ctx, p.OwnerID, p.AuditID)
			if err != nil {
				return err
			}
			if audit.State != auditstore.AuditDraft || audit.Revision != p.ExpectedRevision {
				return auditstore.ErrPrecondition
			}
			profile, err := auditprofile.DecodePinnedProgram(audit.ProfileSnapshot)
			if err != nil || profile.Ref.Name != audit.Profile.Name || profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
				return ErrInvalid
			}
			selection, err := DecodeDraftSelection(audit.InputSelection)
			if err != nil {
				return err
			}
			policy, err := commonActiveCheckPolicy(profile, project.ActiveChecks)
			if err != nil {
				return err
			}
			art := artifacts.NewService(s.storage.Repository(tx))
			projectArtifacts, _ := art.Project(audit.ProjectID)
			refs := []contracts.ArtifactRef{}
			expected := []contracts.ExactArtifact{}
			for _, input := range selection.Inputs {
				refs = append(refs, input.Ref)
				expected = append(expected, input)
			}
			metadata := []contracts.ExactArtifact{}
			if len(refs) > 0 {
				metadata, err = projectArtifacts.DescribeExact(ctx, refs)
				if err != nil {
					return err
				}
			}
			if len(metadata) != len(expected) {
				return ErrInvalid
			}
			for i, exact := range metadata {
				if expected[i].Verify(exact) != nil {
					return ErrInvalid
				}
			}
			lookup, err := runtimeconfig.BindTransactionLLMCredentialLookup(tx, s.transactionLLMCredentials)
			if err != nil {
				return err
			}
			workflowIDs, skillRefs, skillSets, err := s.validateProfileDependencies(ctx, profile, lookup)
			if err != nil {
				return err
			}
			runtimeCredentials := credentials.NewRuntimeCredentialRepository(tx)
			modelFree := true
			for _, workflow := range profile.Workflows {
				modelFree = modelFree && resolved.WorkflowToolWorkersOnly(workflow)
			}
			runtimeSnapshot, err := runstore.NewRunCreationPostgresStore(tx, lookup, runtimeCredentials, s.storage).PinRuntimeLabels(ctx, selection.RuntimeLabels, modelFree)
			if err != nil {
				return err
			}
			if err := contentdigest.CheckPinnedSelection(p.ExpectedRuntimeSHA256, runtimeSnapshot); err != nil {
				return err
			}
			engagement, err := readEngagement(ctx, art, audit.ProjectID, profile, selection)
			if err != nil {
				return err
			}
			user := credentials.RuntimeCredentialUser{UserID: p.OwnerID, Operations: p.OperationsPrincipal}
			if err := validateEngagementCredentials(ctx, tx, user, engagement); err != nil {
				return err
			}
			if reasons := engagementCompatibility(profile, policy, engagement); len(reasons) > 0 {
				return unsupported(reasons)
			}
			target := project.HTTPTarget.Clone()
			if engagement != nil {
				target = nil
			}
			targetIDs := []string{}
			if target != nil && target.Credential != nil {
				if err := runtimeCredentials.ValidateRuntimeCredentialUse(ctx, user, target.Credential.CredentialID, string(target.Credential.Kind)); err != nil {
					return err
				}
				targetIDs = append(targetIDs, target.Credential.CredentialID)
			}
			skillsCatalog, err := agentskills.NewCatalog(art)
			if err != nil {
				return err
			}
			skills, err := skillsCatalog.SelectRunSources(ctx, p.OwnerID, skillRefs)
			if err != nil {
				return err
			}
			for _, skill := range skills {
				if skill.Source == nil {
					return &agentskills.RunSkillError{Code: agentskills.CodeArtifactNotFound, Name: skill.Name, Retryable: false}
				}
			}
			if err := agentskills.ValidateSelectedLimits(skills, skillSets); err != nil {
				return err
			}
			if err := contentdigest.CheckPinnedSelection(p.ExpectedSkillsSHA256, skills); err != nil {
				return err
			}

			baseline := BaselineSnapshot{Schema: BaselineSchema, Labels: selection.Labels.Clone(), RequiredPlacementLabels: append([]string{}, selection.RequiredPlacementLabels...), Inputs: contracts.CloneExactArtifacts(selection.Inputs), Scope: selection.Scope, ActiveChecks: policy, RuntimeLabels: append([]string{}, selection.RuntimeLabels...), RuntimeConfig: runtimeSnapshot, Skills: append([]agentskills.RunSkillSnapshot{}, skills...), LLMCredentialIDs: mergeIDs(workflowIDs, runtimeSnapshot.LLMCredentialIDs), RuntimeCredentialIDs: mergeIDs(runtimeSnapshot.RuntimeCredentialIDs, targetIDs), ProjectHTTPTarget: target, Standards: []auditstandards.PinnedPackage{}}
			if err := pinAssertionSource(ctx, art, audit, profile, &baseline); err != nil {
				return err
			}
			standardSelection := []auditstandards.ExactPackage{}
			for _, standard := range baseline.Standards {
				standardSelection = append(standardSelection, standard.Catalog)
			}
			if err := contentdigest.CheckPinnedSelection(p.ExpectedStandardsSHA256, standardSelection); err != nil {
				return err
			}
			links := []auditstore.ArtifactLink{}
			if baseline.InitialAssertions != nil {
				source := baseline.InitialAssertions.Source
				provenance, err := strictjson.Canonical(map[string]any{"source": source})
				if err != nil {
					return err
				}
				links = append(links, auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "assertion-source", Artifact: contracts.ExactArtifact{Ref: source.Ref, Digest: source.Digest, MediaType: source.MediaType, SizeBytes: source.SizeBytes}, SourceProvenance: provenance})
			}
			seconds := profile.Program.Limits.DeadlineSeconds
			if p.DeadlineSeconds != nil {
				seconds = *p.DeadlineSeconds
			}
			if engagement != nil {
				digest, err := engagement.document.Policy.Digest()
				if err != nil {
					return err
				}
				baseline.Engagement = &auditbaseline.EngagementSnapshot{Schema: auditbaseline.EngagementSchema, Artifact: engagement.artifact, Document: engagement.document, PolicyDigest: digest, AuthorizedBy: p.OwnerID, AuthorizedAt: s.now().UTC()}
				baseline.RuntimeCredentialIDs = mergeIDs(runtimeSnapshot.RuntimeCredentialIDs, engagement.credentialIDs)
			}
			encoded, err := EncodeBaseline(baseline)
			if err != nil {
				return err
			}
			if _, err := auditdomain.ExecutionProgram(profile.Program, encoded); err != nil {
				return err
			}
			var deadline time.Time
			if seconds > 0 {
				deadline = s.now().UTC().Add(timeDurationSeconds(seconds))
			}
			started, created, err := owners.Activate(ctx, auditstore.ActivateParams{OwnerID: p.OwnerID, AuditID: p.AuditID, ExpectedRevision: p.ExpectedRevision, BaselineSnapshot: encoded, DeadlineAt: deadline, InitialRetained: links, IdempotencyKey: p.IdempotencyKey, RequestDigest: p.RequestDigest})
			if err == nil {
				result = StartedAudit{Audit: started, Replayed: !created}
			}
			return err
		})
	})
	if err != nil && postgres.IsTransactionConflict(err) && ctx.Err() == nil {
		replay, found, readErr := s.owners().LookupMutationReplay(ctx, p.OwnerID, auditstore.MutationStart, p.IdempotencyKey, p.RequestDigest)
		if readErr != nil {
			return StartedAudit{}, readErr
		}
		if found {
			return StartedAudit{Audit: replay, Replayed: true}, nil
		}
	}
	return result, err
}

func commonActiveCheckPolicy(profile auditprofile.PinnedProgram, policy contracts.ActiveChecksPolicy) (contracts.ActiveChecksPolicy, error) {
	if !policy.Valid() {
		return "", ErrInvalid
	}
	if profile.Program.Policy.ActiveOperations == "prohibited" {
		policy = contracts.ActiveChecksProhibited
	}
	if policy == contracts.ActiveChecksProhibited {
		for _, workflow := range profile.Workflows {
			for _, stage := range workflow.Stages {
				for _, agent := range stage.Agents {
					for _, tools := range agent.Template.Toolsets {
						if resolved.SelectsActiveCheckTool(tools, agent.Template.SandboxProfile) {
							return "", fmt.Errorf("%w: selected Workflow requires active operations prohibited by the pinned policy", ErrInvalid)
						}
					}
				}
			}
		}
	}
	return policy, nil
}
