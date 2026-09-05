package auditservice

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/jackc/pgx/v5"
)

func (s *Service) Start(ctx context.Context, params StartParams) (StartedAudit, error) {
	if params.OwnerID == "" || params.AuditID == "" || params.ExpectedRevision == 0 ||
		params.IdempotencyKey == "" || !validDigest(params.RequestDigest) {
		return StartedAudit{}, fmt.Errorf("%w: Audit start request is invalid", ErrInvalid)
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationStart,
		params.IdempotencyKey, params.RequestDigest,
	); err != nil {
		return StartedAudit{}, err
	} else if found {
		return s.startedProjection(ctx, replay, true)
	}

	var result StartedAudit
	err := s.credentialGuard.WithRunCreation(ctx, func() error {
		return persistencepostgres.InTx(
			ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead},
			func(tx pgx.Tx) error {
				started, err := s.startInTransaction(ctx, tx, params)
				if err == nil {
					result = started
				}
				return err
			},
		)
	})
	if err == nil {
		return result, nil
	}
	// A concurrent winner may have committed after our REPEATABLE READ snapshot
	// began. Re-read idempotency only after rollback; mutable dependencies are
	// still never consulted by this recovery path.
	if errors.Is(err, auditstore.ErrConflict) || errors.Is(err, artifacts.ErrArtifactConflict) ||
		persistencepostgres.SQLState(err) == "40001" {
		if replay, found, replayErr := store.LookupMutationReplay(
			ctx, params.OwnerID, auditstore.MutationStart,
			params.IdempotencyKey, params.RequestDigest,
		); replayErr == nil && found {
			return s.startedProjection(ctx, replay, true)
		}
	}
	return StartedAudit{}, err
}

func (s *Service) startInTransaction(
	ctx context.Context, tx pgx.Tx, params StartParams,
) (StartedAudit, error) {
	store := auditstore.NewPostgresStore(tx)
	if replay, found, err := store.LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationStart,
		params.IdempotencyKey, params.RequestDigest,
	); err != nil {
		return StartedAudit{}, err
	} else if found {
		return s.startedProjectionWithStore(ctx, store, replay, true)
	}
	audit, err := store.Get(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return StartedAudit{}, err
	}
	if audit.State != auditstore.AuditDraft || audit.Revision != params.ExpectedRevision {
		return StartedAudit{}, auditstore.ErrPrecondition
	}
	profile, err := config.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name ||
		profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return StartedAudit{}, fmt.Errorf("stored AuditProfile snapshot failed validation")
	}
	compatibility := ProfileCompatibility(profile)
	if !compatibility.ServerCompatible {
		return StartedAudit{}, unsupported(compatibility.Reasons)
	}
	selection, err := DecodeDraftSelection(audit.InputSelection)
	if err != nil {
		return StartedAudit{}, err
	}
	project, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(
		ctx, params.OwnerID, audit.ProjectID,
	)
	if err != nil {
		return StartedAudit{}, err
	}

	workflowCredentialIDs, skillRefs, skillSets, err := s.validateProfileDependencies(ctx, profile)
	if err != nil {
		return StartedAudit{}, err
	}
	runs := runstore.NewPostgresStore(tx)
	runtimeSnapshot, err := runs.PinRuntimeLabels(ctx, selection.RuntimeLabels, s.llmCredentials)
	if err != nil {
		return StartedAudit{}, err
	}
	projectTarget := cloneProjectTarget(project.HTTPTarget)
	projectRuntimeCredentialIDs := []string{}
	if projectTarget != nil && projectTarget.Credential != nil {
		if err := s.runtimeCredentials.ValidateRuntimeCredential(
			ctx, projectTarget.Credential.CredentialID, string(projectTarget.Credential.Kind),
		); err != nil {
			return StartedAudit{}, err
		}
		projectRuntimeCredentialIDs = append(projectRuntimeCredentialIDs, projectTarget.Credential.CredentialID)
	}

	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	skillCatalog, err := agentskills.NewCatalog(artifactService)
	if err != nil {
		return StartedAudit{}, err
	}
	skills, err := skillCatalog.SelectRunSources(ctx, params.OwnerID, skillRefs)
	if err != nil {
		return StartedAudit{}, err
	}
	for _, skill := range skills {
		if skill.Source == nil {
			return StartedAudit{}, &agentskills.RunSkillError{
				Code: agentskills.CodeArtifactNotFound, Name: skill.Name, Retryable: false,
			}
		}
	}
	if err := agentskills.ValidateSelectedLimits(skills, skillSets); err != nil {
		return StartedAudit{}, err
	}

	inputPayloads, err := readAndVerifyInputs(ctx, artifactService, audit.ProjectID, profile, selection)
	if err != nil {
		return StartedAudit{}, err
	}
	inventory, err := buildInventory(profile, selection, inputPayloads)
	if err != nil {
		return StartedAudit{}, err
	}
	if reasons := InventoryCompatibility(inventory); len(reasons) != 0 {
		return StartedAudit{}, unsupported(reasons)
	}
	if len(inventory.Worklist.Items) > audit.Limits.MaxItemsPerRound ||
		len(inventory.Worklist.Items) > audit.Limits.MaxItemsTotal {
		return StartedAudit{}, fmt.Errorf("%w: Audit inventory exceeds profile limits", ErrInvalid)
	}

	projectArtifacts, err := artifactService.Project(audit.ProjectID)
	if err != nil {
		return StartedAudit{}, err
	}
	namespace := deterministicID("audit", audit.AuditID)
	taskArtifacts, executionManifest, err := writeTaskPackages(
		ctx, projectArtifacts, namespace, profile, selection, inventory,
	)
	if err != nil {
		return StartedAudit{}, err
	}
	worklistArtifact, err := writeRoundPackage(
		ctx, projectArtifacts, namespace, inventory, executionManifest,
	)
	if err != nil {
		return StartedAudit{}, err
	}

	baseline := BaselineSnapshot{
		Schema: BaselineSchema, Inputs: cloneExactInputs(selection.Inputs), Scope: selection.Scope,
		RuntimeLabels: append([]string{}, selection.RuntimeLabels...), RuntimeConfig: runtimeSnapshot,
		Skills:               append([]contracts.RunSkillSnapshot{}, skills...),
		LLMCredentialIDs:     mergeIDs(workflowCredentialIDs, runtimeSnapshot.LLMCredentialIDs),
		RuntimeCredentialIDs: mergeIDs(runtimeSnapshot.RuntimeCredentialIDs, projectRuntimeCredentialIDs),
		ProjectHTTPTarget:    projectTarget,
		Inventory: BaselineInventory{
			SourceContentDigest:      inventory.SourceContentDigest,
			CanonicalInventoryDigest: inventory.CanonicalInventoryDigest,
			Gaps:                     append([]string{}, inventory.Gaps...), Worklist: worklistArtifact,
			ExecutionManifest: executionManifest,
		},
	}
	baselineJSON, err := EncodeBaseline(baseline)
	if err != nil {
		return StartedAudit{}, err
	}
	items := make([]auditstore.MaterializedItem, len(inventory.Worklist.Items))
	for index, item := range inventory.Worklist.Items {
		coverage := inventory.Coverage.Rows[index]
		items[index] = auditstore.MaterializedItem{
			ItemID:  deterministicID("item", audit.AuditID, item.ItemKey),
			ItemKey: item.ItemKey, Ordinal: item.Ordinal, Kind: item.Kind,
			SubjectKey: item.SubjectKey, Task: taskArtifacts[index],
			WorkflowRole: item.WorkflowRole, InitialState: auditstore.ItemReady,
			Coverage: auditstore.Coverage{
				Status:    auditstore.CoverageStatus(coverage.Status),
				Requested: append([]string{}, coverage.Requested...),
				Completed: append([]string{}, coverage.Completed...),
				Gaps:      append([]string{}, coverage.Gaps...), Rationale: coverage.Rationale,
			},
		}
	}
	roundID := deterministicID("round", audit.AuditID, "1")
	started, created, err := store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: params.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: params.ExpectedRevision,
		RoundID:          roundID, RoundOrdinal: 1, Manifest: worklistArtifact,
		BaselineSnapshot: baselineJSON,
		DeadlineAt: s.now().UTC().Add(
			timeDurationSeconds(profile.Execution.DeadlineSeconds),
		),
		Items: items, IdempotencyKey: params.IdempotencyKey,
		RequestDigest: params.RequestDigest,
	})
	if err != nil {
		return StartedAudit{}, err
	}
	return s.startedProjectionWithStore(ctx, store, started, !created)
}

func (s *Service) validateProfileDependencies(
	ctx context.Context, profile config.ResolvedAuditProfile,
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
		workflow := profile.Workflows[role].Workflow
		if err := config.ValidateResolvedWorkflowCredentials(ctx, workflow, s.llmCredentials); err != nil {
			return nil, nil, nil, err
		}
		ids, err := config.ResolvedWorkflowCredentialIDs(workflow)
		if err != nil {
			return nil, nil, nil, err
		}
		credentialSets = append(credentialSets, ids)
		refs, err := config.WorkflowSkillRefs(workflow)
		if err != nil {
			return nil, nil, nil, err
		}
		for _, ref := range refs {
			skillsByName[ref.Name] = ref
		}
		sets = append(sets, config.WorkflowSkillSets(workflow)...)
	}
	credentials := mergeIDs(credentialSets...)
	names := make([]string, 0, len(skillsByName))
	for name := range skillsByName {
		names = append(names, name)
	}
	sort.Strings(names)
	refs := make([]contracts.ArtifactRef, 0, len(names))
	for _, name := range names {
		refs = append(refs, skillsByName[name])
	}
	return credentials, refs, sets, nil
}

func readAndVerifyInputs(
	ctx context.Context,
	service *artifacts.Service,
	projectID string,
	profile config.ResolvedAuditProfile,
	selection DraftSelection,
) (map[string]artifacts.ReadResult, error) {
	store, err := service.Project(projectID)
	if err != nil {
		return nil, err
	}
	result := make(map[string]artifacts.ReadResult, len(selection.Inputs))
	for name, selected := range selection.Inputs {
		contract, exists := profile.Inputs[name]
		if !exists || !acceptsMediaType(contract.MediaTypes, selected.MediaType) {
			return nil, fmt.Errorf("%w: stored Audit input no longer matches profile", ErrInvalid)
		}
		read, err := store.Read(ctx, selected.Ref)
		if err != nil {
			return nil, err
		}
		if read.Ref.Revision == nil || selected.Ref.Revision == nil ||
			*read.Ref.Revision != *selected.Ref.Revision || read.Payload.MediaType != selected.MediaType ||
			int64(len(read.Payload.Data)) != selected.SizeBytes || digestBytes(read.Payload.Data) != selected.Digest {
			return nil, fmt.Errorf("%w: exact Audit input failed integrity validation", ErrInvalid)
		}
		result[name] = read
	}
	for name, contract := range profile.Inputs {
		if _, exists := result[name]; contract.Required && !exists {
			return nil, fmt.Errorf("%w: required Audit input is missing", ErrInvalid)
		}
	}
	return result, nil
}

func buildInventory(
	profile config.ResolvedAuditProfile,
	selection DraftSelection,
	inputs map[string]artifacts.ReadResult,
) (auditdomain.Inventory, error) {
	source, exists := inputs[profile.Inventory.SourceInput]
	if !exists {
		return auditdomain.Inventory{}, fmt.Errorf("%w: inventory source input is missing", ErrInvalid)
	}
	options := auditdomain.InventoryOptions{
		Round: 1, WorkflowRole: profile.Inventory.ItemWorkflowRole,
		SourceInputName: profile.Inventory.SourceInput, SourceRef: source.Ref,
		ApprovalRequirement: auditdomain.ApprovalNone, Scope: selection.Scope.Values(),
	}
	switch profile.Inventory.Implementation {
	case "checklist@1":
		if source.Payload.MediaType == auditdomain.PackageMediaType {
			return auditdomain.Inventory{}, fmt.Errorf("%w: checklist packages are not supported by this Server", ErrInvalid)
		}
		return auditdomain.BuildChecklistInventory(source.Payload.Data, source.Payload.MediaType, options)
	case "openapi-operations@1":
		if source.Payload.MediaType == auditdomain.PackageMediaType {
			return auditdomain.BuildOpenAPIInventoryFromPackage(source.Payload.Data, options)
		}
		return auditdomain.BuildOpenAPIInventory(source.Payload.Data, source.Payload.MediaType, options)
	default:
		return auditdomain.Inventory{}, unsupported([]CompatibilityReason{ReasonAssessmentUnsupported})
	}
}

func writeTaskPackages(
	ctx context.Context,
	store artifacts.ScopedStore,
	namespace string,
	profile config.ResolvedAuditProfile,
	selection DraftSelection,
	inventory auditdomain.Inventory,
) ([]auditstore.ExactArtifact, auditdomain.ExecutionManifest, error) {
	manifest := inventory.ExecutionManifest
	manifest.Items = append([]auditdomain.ExecutionItem(nil), inventory.ExecutionManifest.Items...)
	result := make([]auditstore.ExactArtifact, len(inventory.Tasks))
	for index, task := range inventory.Tasks {
		write, err := store.Write(ctx, contracts.ArtifactRef{
			Namespace: namespace, Name: task.Item.TaskPackageID,
		}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: task.Package}, nil)
		if err != nil {
			return nil, auditdomain.ExecutionManifest{}, err
		}
		if task.PackageDigest != digestBytes(task.Package) || write.Size != int64(len(task.Package)) ||
			write.MediaType != auditdomain.PackageMediaType {
			return nil, auditdomain.ExecutionManifest{}, errors.New("stored Audit task package failed integrity validation")
		}
		result[index] = auditstore.ExactArtifact{
			Ref: write.Ref, Digest: task.PackageDigest,
			MediaType: write.MediaType, SizeBytes: write.Size,
		}
		manifest.Items[index].TaskRef = exactRefPointer(write.Ref)
		binding, exists := profile.Workflows[task.Item.WorkflowRole]
		if !exists {
			return nil, auditdomain.ExecutionManifest{}, errors.New("Audit item names an unknown Workflow role")
		}
		manifest.Items[index].Inputs = workflowInputs(binding, selection)
	}
	if err := auditdomain.ValidateDispatchExecutionManifest(manifest); err != nil {
		return nil, auditdomain.ExecutionManifest{}, err
	}
	return result, manifest, nil
}

func workflowInputs(
	binding config.ResolvedAuditWorkflowBinding,
	selection DraftSelection,
) []auditdomain.ExactInput {
	inputs := make([]auditdomain.ExactInput, 0)
	names := make([]string, 0, len(binding.Inputs))
	for name := range binding.Inputs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, workflowInput := range names {
		mapping := binding.Inputs[workflowInput]
		if mapping.Source != config.AuditInputFromAudit {
			continue
		}
		selected, exists := selection.Inputs[mapping.Name]
		if !exists {
			continue
		}
		inputs = append(inputs, auditdomain.ExactInput{
			Name: workflowInput, Ref: selected.Ref, Digest: selected.Digest,
		})
	}
	return inputs
}

func writeRoundPackage(
	ctx context.Context,
	store artifacts.ScopedStore,
	namespace string,
	inventory auditdomain.Inventory,
	manifest auditdomain.ExecutionManifest,
) (auditstore.ExactArtifact, error) {
	worklist, err := auditdomain.EncodeWorklist(inventory.Worklist)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	execution, err := auditdomain.EncodeExecutionManifest(manifest)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	coverage, err := auditdomain.EncodeCoverage(inventory.Coverage)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	packageID := deterministicID("worklist", digestBytes(execution), inventory.CanonicalInventoryDigest)
	payload, validated, err := auditdomain.BuildPackage(
		packageID, auditdomain.PackageKindWorklist, "", []auditdomain.PackageInput{
			{ID: "coverage", Path: "coverage.json", MediaType: "application/json", Data: coverage},
			{ID: "execution-manifest", Path: "execution.json", MediaType: "application/json", Data: execution},
			{ID: "inventory", Path: "inventory.json", MediaType: "application/json", Data: inventory.CanonicalInventory},
			{ID: "worklist", Path: "worklist.json", MediaType: "application/json", Data: worklist},
		},
	)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	write, err := store.Write(ctx, contracts.ArtifactRef{
		Namespace: namespace, Name: "round-1-worklist",
	}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload}, nil)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if validated.Digest != digestBytes(payload) || write.Size != int64(len(payload)) {
		return auditstore.ExactArtifact{}, errors.New("stored Audit worklist package failed integrity validation")
	}
	return auditstore.ExactArtifact{
		Ref: write.Ref, Digest: validated.Digest,
		MediaType: write.MediaType, SizeBytes: write.Size,
	}, nil
}

func (s *Service) startedProjection(
	ctx context.Context, audit auditstore.Audit, replayed bool,
) (StartedAudit, error) {
	return s.startedProjectionWithStore(ctx, auditstore.NewPostgresStore(s.pool), audit, replayed)
}

func (s *Service) startedProjectionWithStore(
	ctx context.Context,
	store *auditstore.PostgresStore,
	audit auditstore.Audit,
	replayed bool,
) (StartedAudit, error) {
	if audit.CurrentRoundID == nil {
		return StartedAudit{}, errors.New("started Audit has no current Round")
	}
	round, err := store.GetRound(ctx, audit.AuditID, *audit.CurrentRoundID)
	if err != nil {
		return StartedAudit{}, err
	}
	items, err := store.ListItems(ctx, audit.AuditID)
	if err != nil {
		return StartedAudit{}, err
	}
	return StartedAudit{Audit: audit, Round: round, Items: items, Replayed: replayed}, nil
}

func cloneProjectTarget(source *contracts.HTTPOriginTargetRef) *contracts.HTTPOriginTargetRef {
	if source == nil {
		return nil
	}
	result := *source
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return &result
}

func exactRefPointer(source contracts.ArtifactRef) *contracts.ArtifactRef {
	result := source
	if source.Revision != nil {
		revision := *source.Revision
		result.Revision = &revision
	}
	return &result
}

func timeDurationSeconds(value int) time.Duration {
	return time.Duration(value) * time.Second
}
