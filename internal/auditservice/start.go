package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
	"github.com/jackc/pgx/v5"
)

func (s *Service) Start(ctx context.Context, params StartParams) (StartedAudit, error) {
	if params.OwnerID == "" || params.AuditID == "" || params.ExpectedRevision == 0 ||
		params.IdempotencyKey == "" || !contentdigest.Valid(params.RequestDigest) {
		return StartedAudit{}, fmt.Errorf("%w: Audit start request is invalid", ErrInvalid)
	}
	if err := validateDeadlineSeconds(params.DeadlineSeconds); err != nil {
		return StartedAudit{}, err
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

	plan, err := s.stageStart(ctx, params)
	if err != nil {
		// A concurrent start with this key may have committed after the
		// replay lookup, leaving the draft no longer a draft.
		if errors.Is(err, auditstore.ErrPrecondition) {
			if replay, found, replayErr := store.LookupMutationReplay(
				ctx, params.OwnerID, auditstore.MutationStart,
				params.IdempotencyKey, params.RequestDigest,
			); replayErr == nil && found {
				return s.startedProjection(ctx, replay, true)
			}
		}
		return StartedAudit{}, err
	}
	var result StartedAudit
	err = s.credentialGuard.WithRunCreation(ctx, func() error {
		return persistencepostgres.InTx(
			ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead},
			func(tx pgx.Tx) error {
				started, err := s.startInTransaction(ctx, tx, params, plan)
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
		persistencepostgres.IsTransactionConflict(err) {
		if replay, found, replayErr := store.LookupMutationReplay(
			ctx, params.OwnerID, auditstore.MutationStart,
			params.IdempotencyKey, params.RequestDigest,
		); replayErr == nil && found {
			return s.startedProjection(ctx, replay, true)
		}
	}
	return StartedAudit{}, err
}

// startPlan is what an Audit start stages before its transaction: the draft
// revision it was built from, its decoded profile and selection, the pinned
// standards and, unless the profile prepares its inventory, the first Round.
// The start transaction accepts the plan only for the same draft revision,
// which proves the profile snapshot and the input selection unchanged.
type startPlan struct {
	revision  uint64
	profile   auditprofile.ResolvedAuditProfile
	selection DraftSelection
	standards []auditstandards.PinnedPackage
	round     *startRound
}

// startRound is the staged first Round: its inventory and the exact frozen
// task packages and worklist written for it.
type startRound struct {
	inventory         auditdomain.Inventory
	tasks             []auditstore.ExactArtifact
	executionManifest auditdomain.ExecutionManifest
	worklist          auditstore.ExactArtifact
}

// stageStart writes the content-named artifacts of a start before its
// transaction (AUD-R225): retained standard copies, task packages and the
// worklist, each frozen in the Audit namespace. The start transaction then
// writes only rows, and a retry of the same draft revision finds every
// artifact written and writes none again. The staging transaction holds FOR
// KEY SHARE on the draft row, so a concurrent purge either waits and removes
// what was staged or has already removed the Audit.
func (s *Service) stageStart(ctx context.Context, params StartParams) (startPlan, error) {
	var plan startPlan
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		staged, err := stageStartInTransaction(ctx, tx, params)
		if err == nil {
			plan = staged
		}
		return err
	})
	return plan, err
}

func stageStartInTransaction(ctx context.Context, tx pgx.Tx, params StartParams) (startPlan, error) {
	audit, err := auditstore.NewPostgresStore(tx).LockForStartStaging(ctx, params.OwnerID, params.AuditID)
	if err != nil {
		return startPlan{}, err
	}
	if audit.State != auditstore.AuditDraft || audit.Revision != params.ExpectedRevision {
		return startPlan{}, auditstore.ErrPrecondition
	}
	profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
	if err != nil || profile.Ref.Name != audit.Profile.Name ||
		profile.Ref.Version != audit.Profile.Version || profile.Ref.Digest != audit.Profile.Digest {
		return startPlan{}, fmt.Errorf("stored AuditProfile snapshot failed validation")
	}
	compatibility := ProfileCompatibility(profile)
	if !compatibility.ServerCompatible {
		return startPlan{}, unsupported(compatibility.Reasons)
	}
	selection, err := DecodeDraftSelection(audit.InputSelection)
	if err != nil {
		return startPlan{}, err
	}

	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	namespace := auditdomain.ArtifactNamespace(audit.AuditID)
	standardCatalog, err := auditstandards.NewCatalog(artifactService)
	if err != nil {
		return startPlan{}, err
	}
	standardRefs := make([]auditstandards.Reference, len(profile.Standards))
	for index, ref := range profile.Standards {
		standardRefs[index] = auditstandards.Reference{Scheme: ref.Scheme, Version: ref.Version}
	}
	pinnedStandards, err := standardCatalog.Pin(
		ctx, params.OwnerID, audit.ProjectID, namespace, standardRefs,
	)
	if err != nil {
		return startPlan{}, err
	}
	resolvedStandards := make([]auditstandards.ResolvedPackage, len(pinnedStandards))
	for index, pinned := range pinnedStandards {
		resolvedStandards[index], err = standardCatalog.ResolvePinned(ctx, audit.ProjectID, pinned)
		if err != nil {
			return startPlan{}, err
		}
	}
	expectedStandards := make([]auditstandards.ExactPackage, 0, len(pinnedStandards))
	for _, standard := range pinnedStandards {
		expectedStandards = append(expectedStandards, standard.Catalog)
	}
	if err := contracts.CheckPinnedSelection(params.ExpectedStandardsSHA256, expectedStandards); err != nil {
		return startPlan{}, err
	}
	inputPayloads, err := readAndVerifyInputs(ctx, artifactService, audit.ProjectID, profile, selection)
	if err != nil {
		return startPlan{}, err
	}
	plan := startPlan{
		revision: audit.Revision, profile: profile, selection: selection, standards: pinnedStandards,
	}
	if profile.HasPreparation() {
		return plan, nil
	}
	inventory, err := buildInventory(profile, selection, inputPayloads, resolvedStandards)
	if err != nil {
		return startPlan{}, err
	}
	if err := validateInventoryTaskExecution(profile, inventory); err != nil {
		return startPlan{}, err
	}
	if len(inventory.Worklist.Items) > roundItemCapacity(audit.Limits.MaxItemsPerRound, audit.Limits.MaxItemsTotal, 0) {
		return startPlan{}, fmt.Errorf("%w: Audit inventory exceeds profile limits", ErrInvalid)
	}
	writer := auditArtifactWriter{service: artifactService, projectID: audit.ProjectID}
	taskArtifacts, executionManifest, err := writeTaskPackages(
		ctx, writer, namespace, profile, selection, inventory,
	)
	if err != nil {
		return startPlan{}, err
	}
	worklistArtifact, err := writeRoundPackage(
		ctx, writer, namespace, 1, inventory, executionManifest,
	)
	if err != nil {
		return startPlan{}, err
	}
	plan.round = &startRound{
		inventory: inventory, tasks: taskArtifacts,
		executionManifest: executionManifest, worklist: worklistArtifact,
	}
	return plan, nil
}

// startInTransaction freezes the baseline and materializes the staged first
// Round. It writes rows only: every artifact it names was staged and frozen
// by stageStart for this draft revision.
func (s *Service) startInTransaction(
	ctx context.Context, tx pgx.Tx, params StartParams, plan startPlan,
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
	// The staged plan holds for this exact draft revision only.
	if audit.State != auditstore.AuditDraft || audit.Revision != params.ExpectedRevision ||
		audit.Revision != plan.revision {
		return StartedAudit{}, auditstore.ErrPrecondition
	}
	profile, selection := plan.profile, plan.selection
	project, err := projectstore.NewPostgresStore(tx).LockActiveAuditProject(
		ctx, params.OwnerID, audit.ProjectID,
	)
	if err != nil {
		return StartedAudit{}, err
	}
	txLLMCredentials, err := runtimeconfig.BindTransactionLLMCredentialLookup(
		tx, s.transactionLLMCredentials,
	)
	if err != nil {
		return StartedAudit{}, err
	}

	workflowCredentialIDs, skillRefs, skillSets, err := s.validateProfileDependencies(
		ctx, profile, txLLMCredentials,
	)
	if err != nil {
		return StartedAudit{}, err
	}
	runs := runstore.NewRunCreationPostgresStore(tx, txLLMCredentials)
	runtimeSnapshot, err := runs.PinRuntimeLabels(ctx, selection.RuntimeLabels)
	if err != nil {
		return StartedAudit{}, err
	}
	if err := contracts.CheckPinnedSelection(params.ExpectedRuntimeSHA256, runtimeSnapshot); err != nil {
		return StartedAudit{}, err
	}
	projectTarget := project.HTTPTarget.Clone()
	projectRuntimeCredentialIDs := []string{}
	if projectTarget != nil && projectTarget.Credential != nil {
		if err := credentials.NewRuntimeCredentialRepository(tx).ValidateRuntimeCredentialUse(
			ctx, credentials.RuntimeCredentialUser{UserID: params.OwnerID, Operations: params.OperationsPrincipal},
			projectTarget.Credential.CredentialID, string(projectTarget.Credential.Kind),
		); err != nil {
			return StartedAudit{}, err
		}
		projectRuntimeCredentialIDs = append(projectRuntimeCredentialIDs, projectTarget.Credential.CredentialID)
	}

	artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
	standardLinks, err := auditStandardLinks(plan.standards)
	if err != nil {
		return StartedAudit{}, err
	}
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

	if err := contracts.CheckPinnedSelection(params.ExpectedSkillsSHA256, skills); err != nil {
		return StartedAudit{}, err
	}
	baseline := BaselineSnapshot{
		Schema: BaselineSchema, Inputs: cloneExactInputs(selection.Inputs), Scope: selection.Scope,
		RuntimeLabels: append([]string{}, selection.RuntimeLabels...), RuntimeConfig: runtimeSnapshot,
		Skills:               append([]contracts.RunSkillSnapshot{}, skills...),
		LLMCredentialIDs:     mergeIDs(workflowCredentialIDs, runtimeSnapshot.LLMCredentialIDs),
		RuntimeCredentialIDs: mergeIDs(runtimeSnapshot.RuntimeCredentialIDs, projectRuntimeCredentialIDs),
		ProjectHTTPTarget:    projectTarget,
		Standards:            append([]auditstandards.PinnedPackage{}, plan.standards...),
	}
	deadlineSeconds := profile.Execution.DeadlineSeconds
	if params.DeadlineSeconds != nil {
		deadlineSeconds = *params.DeadlineSeconds
	}
	var deadline time.Time
	if deadlineSeconds > 0 {
		deadline = s.now().UTC().Add(timeDurationSeconds(deadlineSeconds))
	}
	if profile.HasPreparation() {
		baselineJSON, err := EncodeBaseline(baseline)
		if err != nil {
			return StartedAudit{}, err
		}
		started, created, err := store.StartPreparation(ctx, auditstore.StartPreparationParams{
			OwnerID: params.OwnerID, AuditID: audit.AuditID, ExpectedRevision: params.ExpectedRevision,
			BaselineSnapshot: baselineJSON, DeadlineAt: deadline, InitialRetained: standardLinks,
			IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
		})
		if err != nil {
			return StartedAudit{}, err
		}
		return s.startedProjectionWithStore(ctx, store, started, !created)
	}
	round := plan.round
	if round == nil {
		return StartedAudit{}, errors.New("staged Audit start has no first Round")
	}
	baseline.Inventory = &BaselineInventory{
		SourceContentDigest:      round.inventory.SourceContentDigest,
		CanonicalInventoryDigest: round.inventory.CanonicalInventoryDigest,
		StandardSelection:        cloneAuditStandardSelection(profile.Inventory.StandardSelection),
		Gaps:                     append([]string{}, round.inventory.Gaps...), Worklist: round.worklist,
		ExecutionManifest: round.executionManifest,
	}
	baselineJSON, err := EncodeBaseline(baseline)
	if err != nil {
		return StartedAudit{}, err
	}
	items, err := materializeInventoryItems(audit.AuditID, profile, round.inventory, round.tasks)
	if err != nil {
		return StartedAudit{}, err
	}
	roundID := auditdomain.DeterministicID("round", audit.AuditID, "1")
	started, created, err := store.MaterializeRound(ctx, auditstore.MaterializeRoundParams{
		OwnerID: params.OwnerID, AuditID: audit.AuditID,
		ExpectedRevision: params.ExpectedRevision,
		RoundID:          roundID, RoundOrdinal: 1, Manifest: round.worklist,
		BaselineSnapshot: baselineJSON,
		DeadlineAt:       deadline,
		Items:            items, InitialRetained: standardLinks, IdempotencyKey: params.IdempotencyKey,
		RequestDigest: params.RequestDigest,
	})
	if err != nil {
		return StartedAudit{}, err
	}
	return s.startedProjectionWithStore(ctx, store, started, !created)
}

// materializeInventoryItems is shared by direct start and prepared initial
// inventory. Workers never supply item identity, approval or coverage authority.
func materializeInventoryItems(auditID string, profile auditprofile.ResolvedAuditProfile, inventory auditdomain.Inventory, taskArtifacts []auditstore.ExactArtifact) ([]auditstore.MaterializedItem, error) {
	items := make([]auditstore.MaterializedItem, len(inventory.Worklist.Items))
	for index, item := range inventory.Worklist.Items {
		coverage := inventory.Coverage.Rows[index]
		task := inventory.Tasks[index].Document
		sourceRef := task.SourceRef
		origin := auditstore.ItemOrigin{
			Schema: auditstore.ItemOriginSchema, SourceRef: &sourceRef,
			SourceContentDigest: task.SourceContentDigest, SourceMediaType: task.SourceMediaType,
			CanonicalInventoryDigest: task.CanonicalInventoryDigest, EntryKey: task.ItemKey,
		}
		if task.Checklist != nil {
			origin.EntryVersion = task.Checklist.Version
		}
		if task.Standard != nil {
			origin.Standard = &auditstore.StandardItemOrigin{
				Scheme: task.Standard.Scheme, Version: task.Standard.Version,
				MappingKey: task.Standard.MappingKey,
				EntryIDs:   append([]string{}, task.Standard.EntryIDs...),
				EvidenceContract: auditstore.StandardEvidenceOrigin{
					ID:      task.Standard.EvidenceContract.ID,
					Version: task.Standard.EvidenceContract.Version,
				},
			}
		}
		itemID := auditdomain.DeterministicID("item", auditID, item.ItemKey)
		approvalKind, approvalDigest, initialState, err := materializedItemApproval(
			auditID, profile, itemID, item, taskArtifacts[index],
		)
		if err != nil {
			return nil, err
		}
		items[index] = auditstore.MaterializedItem{
			ItemID:  itemID,
			ItemKey: item.ItemKey, Ordinal: item.Ordinal, Kind: item.Kind,
			SubjectKey: item.SubjectKey, Task: taskArtifacts[index], Origin: origin,
			WorkflowRole: item.WorkflowRole, InitialState: initialState,
			ApprovalKind: approvalKind, ApprovalDigest: approvalDigest,
			Coverage: auditstore.Coverage{
				Status:    auditstore.CoverageStatus(coverage.Status),
				Requested: append([]string{}, coverage.Requested...),
				Completed: append([]string{}, coverage.Completed...),
				Gaps:      append([]string{}, coverage.Gaps...), Rationale: coverage.Rationale,
			},
		}
	}
	return items, nil
}

func auditStandardLinks(pinned []auditstandards.PinnedPackage) ([]auditstore.ArtifactLink, error) {
	result := make([]auditstore.ArtifactLink, len(pinned))
	for index, standard := range pinned {
		provenance, err := auditstandards.RetainedProvenance(standard)
		if err != nil {
			return nil, err
		}
		result[index] = auditstore.ArtifactLink{
			LogicalKey:       "standard/" + standard.Reference.Scheme + "/" + standard.Reference.Version,
			Artifact:         standard.Retained.Exact(),
			SourceProvenance: provenance,
			DisplayRef:       standard.Reference.Scheme + "@" + standard.Reference.Version,
		}
	}
	return result, nil
}

func materializedItemApproval(
	auditID string,
	profile auditprofile.ResolvedAuditProfile,
	itemID string,
	item auditdomain.WorklistItem,
	task auditstore.ExactArtifact,
) (auditstore.ItemApprovalKind, string, auditstore.ItemState, error) {
	kind := auditstore.ItemApprovalNone
	switch item.ApprovalRequirement {
	case auditdomain.ApprovalNone:
		return kind, "", auditstore.ItemReady, nil
	case auditdomain.ApprovalActiveCheck:
		kind = auditstore.ItemApprovalActiveCheck
	case auditdomain.ApprovalHumanReview:
		kind = auditstore.ItemApprovalApplicability
	default:
		return "", "", "", fmt.Errorf("%w: Audit item approval requirement is invalid", ErrInvalid)
	}
	encoded, err := json.Marshal(struct {
		Schema        string                      `json:"schema"`
		AuditID       string                      `json:"auditId"`
		ProfileDigest string                      `json:"profileDigest"`
		ItemID        string                      `json:"itemId"`
		ItemKey       string                      `json:"itemKey"`
		SubjectKey    string                      `json:"subjectKey"`
		WorkflowRole  string                      `json:"workflowRole"`
		ApprovalKind  auditstore.ItemApprovalKind `json:"approvalKind"`
		Task          auditstore.ExactArtifact    `json:"task"`
	}{
		Schema: "contractor.audit.item-approval.v1", AuditID: auditID,
		ProfileDigest: profile.Ref.Digest, ItemID: itemID, ItemKey: item.ItemKey,
		SubjectKey: item.SubjectKey, WorkflowRole: item.WorkflowRole,
		ApprovalKind: kind, Task: task,
	})
	if err != nil {
		return "", "", "", err
	}
	return kind, auditdomain.DigestBytes(encoded), auditstore.ItemAwaitingReview, nil
}

func (s *Service) validateProfileDependencies(
	ctx context.Context,
	profile auditprofile.ResolvedAuditProfile,
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
		workflow := profile.Workflows[role].Workflow
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
	profile auditprofile.ResolvedAuditProfile,
	selection DraftSelection,
) (map[string]artifacts.ReadResult, error) {
	store, err := service.Project(projectID)
	if err != nil {
		return nil, err
	}
	result := make(map[string]artifacts.ReadResult, len(selection.Inputs))
	for name, selected := range selection.Inputs {
		contract, exists := profile.Inputs[name]
		if !exists || !contracts.AcceptsMediaType(contract.MediaTypes, selected.MediaType) {
			return nil, fmt.Errorf("%w: stored Audit input no longer matches profile", ErrInvalid)
		}
		if selected.Validate() != nil || !selected.HasMetadata() {
			return nil, fmt.Errorf("%w: exact Audit input failed integrity validation", ErrInvalid)
		}
		read, err := store.ReadExact(ctx, selected)
		if errors.Is(err, contracts.ErrExactArtifactMismatch) {
			return nil, fmt.Errorf("%w: exact Audit input failed integrity validation: %v", ErrInvalid, err)
		}
		if err != nil {
			return nil, err
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
	profile auditprofile.ResolvedAuditProfile,
	selection DraftSelection,
	inputs map[string]artifacts.ReadResult,
	standards []auditstandards.ResolvedPackage,
) (auditdomain.Inventory, error) {
	approval := auditdomain.ApprovalNone
	if profile.Interaction.ActiveChecks == auditprofile.AuditActiveChecksApprovalRequired &&
		workflowRoleSelectsClassifiedTool(profile, profile.Inventory.ItemWorkflowRole, true) {
		approval = auditdomain.ApprovalActiveCheck
	}
	if profile.Inventory.Implementation == "standard-mappings@1" {
		if len(standards) != 1 || standards[0].Source.Artifact.ValidateExact() != nil {
			return auditdomain.Inventory{}, fmt.Errorf("%w: exact standard inventory source is missing", ErrInvalid)
		}
		var standardSelection *auditdomain.StandardSelection
		if profile.Inventory.StandardSelection != nil {
			standardSelection = &auditdomain.StandardSelection{
				Scope:    profile.Inventory.StandardSelection.Scope,
				Levels:   append([]string{}, profile.Inventory.StandardSelection.Levels...),
				EntryIDs: append([]string{}, profile.Inventory.StandardSelection.EntryIDs...),
			}
		}
		return auditdomain.BuildStandardMappingInventory(
			standards[0].Package,
			auditdomain.InventoryOptions{
				Round: 1, ProfileMode: string(profile.Mode), WorkflowRole: profile.Inventory.ItemWorkflowRole,
				SourceInputName: "standard", SourceRef: standards[0].Source.Artifact,
				ApprovalRequirement: approval, Scope: selection.Scope.Values(),
				StandardSelection: standardSelection,
			},
		)
	}
	source, exists := inputs[profile.Inventory.Source.Name]
	if !exists {
		return auditdomain.Inventory{}, fmt.Errorf("%w: inventory source input is missing", ErrInvalid)
	}
	var settings artifacts.ReadResult
	if profile.Inventory.Settings != nil {
		settings = inputs[profile.Inventory.Settings.Name]
	}
	return buildInventoryFromSources(profile, selection, source, settings, approval)
}

func buildInventoryFromSources(profile auditprofile.ResolvedAuditProfile, selection DraftSelection, source, settings artifacts.ReadResult, approval auditdomain.ApprovalRequirement) (auditdomain.Inventory, error) {
	options := auditdomain.InventoryOptions{
		Round: 1, ProfileMode: string(profile.Mode), WorkflowRole: profile.Inventory.ItemWorkflowRole,
		SourceInputName: profile.Inventory.Source.Name, SourceRef: source.Ref,
		ApprovalRequirement: approval, Scope: selection.Scope.Values(),
	}
	switch profile.Inventory.Implementation {
	case auditprofile.AuditInventoryOpenAPIScans:
		if settings.Ref.Revision == nil || settings.Payload.MediaType != "application/json" {
			return auditdomain.Inventory{}, fmt.Errorf("%w: scan settings input is missing or invalid", ErrInvalid)
		}
		return auditdomain.BuildOpenAPIScanInventory(source.Payload.Data, source.Payload.MediaType, settings.Payload.Data,
			auditdomain.ExactInput{Name: profile.Inventory.Settings.Name, Ref: settings.Ref, Digest: auditdomain.DigestBytes(settings.Payload.Data)}, options)
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
	writer auditArtifactWriter,
	namespace string,
	profile auditprofile.ResolvedAuditProfile,
	selection DraftSelection,
	inventory auditdomain.Inventory,
	prepared ...map[string]auditstore.ExactArtifact,
) ([]auditstore.ExactArtifact, auditdomain.ExecutionManifest, error) {
	manifest := inventory.ExecutionManifest
	manifest.Items = append([]auditdomain.ExecutionItem(nil), inventory.ExecutionManifest.Items...)
	result := make([]auditstore.ExactArtifact, len(inventory.Tasks))
	for index, task := range inventory.Tasks {
		write, err := writer.write(ctx, contracts.ArtifactRef{
			Namespace: namespace, Name: task.Item.TaskPackageID,
		}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: task.Package})
		if err != nil {
			return nil, auditdomain.ExecutionManifest{}, err
		}
		if task.PackageDigest != auditdomain.DigestBytes(task.Package) || write.SizeBytes != int64(len(task.Package)) ||
			write.MediaType != auditdomain.PackageMediaType {
			return nil, auditdomain.ExecutionManifest{}, inconsistentRound("a stored Audit task package differs from its built bytes", nil)
		}
		result[index] = auditstore.ExactArtifact{
			Ref: write.Ref, Digest: task.PackageDigest,
			MediaType: write.MediaType, SizeBytes: write.SizeBytes,
		}
		manifest.Items[index].TaskRef = exactRefPointer(write.Ref)
		binding, exists := profile.Workflows[task.Item.WorkflowRole]
		if !exists {
			return nil, auditdomain.ExecutionManifest{}, inconsistentRound("an Audit item names an unknown Workflow role", nil)
		}
		manifest.Items[index].Inputs = workflowInputs(binding, selection, prepared...)
	}
	if err := auditdomain.ValidateDispatchExecutionManifest(manifest); err != nil {
		return nil, auditdomain.ExecutionManifest{}, err
	}
	return result, manifest, nil
}

func workflowInputs(
	binding auditprofile.ResolvedAuditWorkflowBinding,
	selection DraftSelection,
	prepared ...map[string]auditstore.ExactArtifact,
) []auditdomain.ExactInput {
	inputs := make([]auditdomain.ExactInput, 0)
	names := make([]string, 0, len(binding.Inputs))
	for name := range binding.Inputs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, workflowInput := range names {
		mapping := binding.Inputs[workflowInput]
		var selected auditstore.ExactArtifact
		var exists bool
		switch mapping.Source {
		case auditprofile.AuditInputFromAudit:
			selected, exists = selection.Inputs[mapping.Name]
		case auditprofile.AuditInputFromPreparation:
			if len(prepared) == 1 {
				selected, exists = prepared[0][preparationInputKey(mapping)]
			}
		default:
			continue
		}
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
	writer auditArtifactWriter,
	namespace string,
	roundOrdinal int,
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
	packageID := auditdomain.DeterministicID("worklist", auditdomain.DigestBytes(execution), inventory.CanonicalInventoryDigest)
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
	write, err := writer.write(ctx, contracts.ArtifactRef{
		Namespace: namespace,
		Name:      auditdomain.DeterministicID("round-worklist", strconv.Itoa(roundOrdinal), validated.Digest),
	}, artifacts.Payload{MediaType: auditdomain.PackageMediaType, Data: payload})
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	if validated.Digest != auditdomain.DigestBytes(payload) || write.SizeBytes != int64(len(payload)) {
		return auditstore.ExactArtifact{}, inconsistentRound("a stored Audit worklist package differs from its built bytes", nil)
	}
	return auditstore.ExactArtifact{
		Ref: write.Ref, Digest: validated.Digest,
		MediaType: write.MediaType, SizeBytes: write.SizeBytes,
	}, nil
}

// auditArtifactWriter writes the frozen, content-named Audit artifacts of one
// Project through artifacts.Service.WriteAuditArtifact, which also recovers a
// write whose acknowledgement was lost.
type auditArtifactWriter struct {
	service   *artifacts.Service
	projectID string
}

func (w auditArtifactWriter) write(
	ctx context.Context,
	target contracts.ArtifactRef,
	payload artifacts.Payload,
) (auditstore.ExactArtifact, error) {
	written, err := w.service.WriteAuditArtifact(ctx, w.projectID, target, payload)
	if err != nil {
		return auditstore.ExactArtifact{}, err
	}
	return contracts.ExactContent(written.Ref, written.MediaType, payload.Data), nil
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
		if audit.Phase != auditdomain.AuditPhasePreparing && audit.Phase != auditdomain.AuditPhaseInventory {
			return StartedAudit{}, errors.New("started Audit has an invalid phase without a Round")
		}
		return StartedAudit{Audit: audit, Items: []auditstore.Item{}, Replayed: replayed}, nil
	}
	round, err := store.GetRound(ctx, audit.AuditID, *audit.CurrentRoundID)
	if err != nil {
		return StartedAudit{}, err
	}
	// The response carries the current Round and only its items, at most
	// LIM-AUD-26, however many items earlier Rounds hold.
	items, err := store.ListRoundItems(ctx, audit.AuditID, round.RoundID)
	if err != nil {
		return StartedAudit{}, err
	}
	return StartedAudit{Audit: audit, Round: round, Items: items, Replayed: replayed}, nil
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
