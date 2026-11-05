package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func New(options Options) (*Service, error) {
	if options.Storage == nil || options.Pool == nil || options.Profiles == nil || options.TransactionLLMCredentials == nil ||
		options.CredentialGuard == nil {
		return nil, errors.New("Audit service dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	findings, err := findingintake.New(options.Pool, options.Storage)
	if err != nil {
		return nil, err
	}
	standards, err := auditstandards.NewCatalog(
		artifacts.NewService(options.Storage.Repository(options.Pool)),
	)
	if err != nil {
		return nil, err
	}
	return &Service{storage: options.Storage,
		pool: options.Pool, profiles: options.Profiles,
		transactionLLMCredentials: options.TransactionLLMCredentials,
		credentialGuard:           options.CredentialGuard,
		findings:                  findings,
		standards:                 standards,
		now:                       options.Now,
	}, nil
}

func (s *Service) Profiles() []ProfileProjection {
	profiles := s.profiles.AuditProfiles()
	result := make([]ProfileProjection, 0, len(profiles))
	for _, profile := range profiles {
		result = append(result, ProfileProjection{
			Profile: profile,
		})
	}
	return result
}

func (s *Service) Profile(selector ProfileSelector) (ProfileProjection, error) {
	raw, err := normalizeProfileSelector(selector)
	if err != nil {
		return ProfileProjection{}, err
	}
	profile, err := s.profiles.AuditProfile(raw)
	if err != nil {
		return ProfileProjection{}, ErrProfileNotFound
	}
	return ProfileProjection{Profile: profile}, nil
}

func (s *Service) CreateDraft(
	ctx context.Context, params CreateDraftParams,
) (auditstore.Audit, bool, error) {
	if err := validateCreateParams(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	metadata, err := runlabels.NormalizeAuditLabels(params.Labels)
	if err != nil {
		return auditstore.Audit{}, false, fmt.Errorf("%w: Audit labels are invalid: %v", ErrInvalid, err)
	}
	placement, _ := metadata.PlacementRequirements()
	if replay, found, err := s.owners().LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationCreate,
		params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	selector, _ := normalizeProfileSelector(params.Profile)
	var profile auditprofile.PinnedProgram
	if params.PinnedProfile != nil {
		if !contentdigest.Valid(params.ExpectedProfileSHA256) || params.PinnedProfile.Ref.Name+"@"+params.PinnedProfile.Ref.Version != selector {
			return auditstore.Audit{}, false, ErrInvalid
		}
		profile, err = s.profiles.ResolvePinnedAuditProfile(ctx, *params.PinnedProfile)
	} else {
		profile, err = s.profiles.ResolveAuditProfile(ctx, selector)
	}
	if err != nil {
		if errors.Is(err, contentdigest.ErrPinnedSelectionChanged) {
			return auditstore.Audit{}, false, err
		}
		return auditstore.Audit{}, false, ErrProfileNotFound
	}
	selectedProfile := profile
	if params.PinnedProfile == nil && params.ExpectedProfileSHA256 != "" {
		selectedProfile, err = s.profiles.AuditProfile(selector)
		if err != nil {
			return auditstore.Audit{}, false, ErrProfileNotFound
		}
	}
	if err := contentdigest.CheckPinnedSelection(params.ExpectedProfileSHA256, selectedProfile); err != nil {
		return auditstore.Audit{}, false, err
	}
	project, err := projectstore.NewPostgresStore(s.pool).Get(ctx, params.OwnerID, params.ProjectID)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	if project.Kind != projectstore.KindProject {
		return auditstore.Audit{}, false, projectstore.ErrNotFound
	}
	if project.Lifecycle == projectstore.LifecycleDeleting {
		return auditstore.Audit{}, false, projectstore.ErrDeleting
	}
	labels, err := normalizeLabels(params.RuntimeLabels)
	if err != nil {
		return auditstore.Audit{}, false, fmt.Errorf("%w: Runtime labels are invalid", ErrInvalid)
	}
	scope, err := normalizeScope(params.Scope)
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	inputs, err := s.selectDraftInputs(ctx, profile, params.ProjectID, params.Inputs)
	if err != nil {
		return auditstore.Audit{}, false, err
	}

	// Engagement validation reads exact owner-scoped artifacts and credential
	// metadata only. A draft never retains credentials or starts target work.
	var engagement *engagementSelection
	hasEngagement := false
	for _, exact := range inputs {
		hasEngagement = hasEngagement || exact.MediaType == engagementcontract.MediaType
	}
	if hasEngagement {
		err = s.credentialGuard.WithRunCreation(ctx, func() error {
			return persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
				var readErr error
				engagement, readErr = readEngagement(ctx, artifacts.NewService(s.storage.Repository(tx)), params.ProjectID, profile, DraftSelection{Inputs: inputs})
				if readErr != nil {
					return readErr
				}
				return validateEngagementCredentials(ctx, tx, credentials.RuntimeCredentialUser{UserID: params.OwnerID, Operations: params.OperationsPrincipal}, engagement)
			})
		})
		if err != nil {
			return auditstore.Audit{}, false, err
		}
	}
	limits := auditstore.Limits{
		MaxRounds: profile.Program.Limits.MaxIterations, BatchSize: profile.Program.Limits.BatchSize,
		MaxItemsPerRound: profile.Program.Limits.MaxTasksPerIteration, MaxItemsTotal: profile.Program.Limits.MaxTasks,
		MaxSubmittedRuns: profile.Program.Limits.MaxSubmittedRuns, MaxItemRunAttempts: profile.Program.Limits.MaxAttempts,
		MaxEvidenceBytes: profile.Program.Limits.MaxEvidenceBytes,
	}
	if engagement != nil {
		limits.MaxItemRunAttempts = 1
	}
	profileSnapshot, err := strictjson.Canonical(profile)
	if err != nil {
		return auditstore.Audit{}, false, fmt.Errorf("encode AuditProfile snapshot: %w", err)
	}
	selection, err := encodeDraftSelection(DraftSelection{
		Labels:                  metadata,
		RequiredPlacementLabels: placement,
		Schema:                  DraftSelectionSchema, Inputs: inputs,
		RuntimeLabels: labels, Scope: scope,
	})
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return s.owners().CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: params.AuditID, OwnerID: params.OwnerID, ProjectID: params.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
		},
		ProfileSnapshot: profileSnapshot, InputSelection: selection,
		Limits:         limits,
		IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
	})
}

func (s *Service) Get(
	ctx context.Context, ownerID, auditID string,
) (auditstore.Audit, error) {
	return s.reads().Get(ctx, ownerID, auditID)
}

func (s *Service) GetReport(
	ctx context.Context, ownerID, auditID string,
) (ReportProjection, error) {
	audit, err := s.reads().Get(ctx, ownerID, auditID)
	if err != nil {
		return ReportProjection{}, err
	}
	// Bounded deletion can remove the two immutable report links in different
	// committed passes. Do not read a partial set once deletion owns the Audit.
	if audit.State == auditstore.AuditDeleting {
		return ReportProjection{Status: ReportUnavailable}, nil
	}
	machine, machineErr := s.reads().GetArtifactLink(ctx, auditID, auditstore.ReportMachineLogicalKey)
	summary, summaryErr := s.reads().GetArtifactLink(ctx, auditID, auditstore.ReportSummaryLogicalKey)
	status := ReportReady
	if machineErr != nil && !errors.Is(machineErr, auditstore.ErrNotFound) {
		return ReportProjection{}, machineErr
	}
	if summaryErr != nil && !errors.Is(summaryErr, auditstore.ErrNotFound) {
		return ReportProjection{}, summaryErr
	}
	if errors.Is(machineErr, auditstore.ErrNotFound) && errors.Is(summaryErr, auditstore.ErrNotFound) {
		switch audit.State {
		case auditstore.AuditCompleted, auditstore.AuditFailed, auditstore.AuditCancelled:
			return ReportProjection{Status: ReportUnavailable}, nil
		default:
			return ReportProjection{Status: ReportPending}, nil
		}
	}
	if machineErr != nil || summaryErr != nil {
		return ReportProjection{}, errors.New("stored Audit report link set is incomplete")
	}
	project, err := artifacts.NewService(s.storage.Repository(s.pool)).Project(audit.ProjectID)
	if err != nil {
		return ReportProjection{}, err
	}
	machineRead, err := project.Read(ctx, machine.Artifact.Ref)
	if err != nil {
		return ReportProjection{}, err
	}
	summaryRead, err := project.Read(ctx, summary.Artifact.Ref)
	if err != nil {
		return ReportProjection{}, err
	}
	if machine.Artifact.MediaType != "application/json" ||
		summary.Artifact.MediaType != "text/markdown" ||
		machineRead.Payload.MediaType != machine.Artifact.MediaType ||
		summaryRead.Payload.MediaType != summary.Artifact.MediaType ||
		auditdomain.DigestBytes(machineRead.Payload.Data) != machine.Artifact.Digest ||
		auditdomain.DigestBytes(summaryRead.Payload.Data) != summary.Artifact.Digest ||
		int64(len(machineRead.Payload.Data)) != machine.Artifact.SizeBytes ||
		int64(len(summaryRead.Payload.Data)) != summary.Artifact.SizeBytes ||
		!json.Valid(machineRead.Payload.Data) || len(summaryRead.Payload.Data) > auditstore.MaxSummaryBytes ||
		!utf8.Valid(summaryRead.Payload.Data) || strings.ContainsRune(string(summaryRead.Payload.Data), 0) {
		return ReportProjection{}, errors.New("stored Audit report failed integrity validation")
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(machineRead.Payload.Data, &object) != nil || object == nil {
		return ReportProjection{}, errors.New("stored Audit machine report is not an object")
	}
	if err := s.checkPageRevision(ctx, ownerID, auditID, audit.Revision); err != nil {
		return ReportProjection{}, err
	}
	machineArtifact, summaryArtifact := machine.Artifact, summary.Artifact
	return ReportProjection{
		Status: status, MachineArtifact: &machineArtifact, SummaryArtifact: &summaryArtifact,
		Machine: append(json.RawMessage(nil), machineRead.Payload.Data...), Summary: string(summaryRead.Payload.Data),
	}, nil
}

func (s *Service) selectDraftInputs(
	ctx context.Context,
	profile auditprofile.PinnedProgram,
	projectID string,
	selected map[string]contracts.ArtifactRef,
) (map[string]auditstore.ExactArtifact, error) {
	if selected == nil || len(selected) > auditprofile.MaxAuditProfileInputs {
		return nil, fmt.Errorf("%w: Audit inputs are invalid", ErrInvalid)
	}
	for name := range selected {
		if _, ok := profile.Program.Inputs[name]; !ok {
			return nil, fmt.Errorf("%w: Audit input slot is unknown", ErrInvalid)
		}
	}
	for name, contract := range profile.Program.Inputs {
		if _, ok := selected[name]; contract.Required && !ok {
			return nil, fmt.Errorf("%w: required Audit input is missing", ErrInvalid)
		}
	}
	projectArtifacts, err := artifacts.NewService(
		s.storage.Repository(s.pool),
	).Project(projectID)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(selected))
	for name := range selected {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make(map[string]auditstore.ExactArtifact, len(names))
	for _, name := range names {
		ref := selected[name]
		if err := ref.ValidateExact(); err != nil {
			return nil, fmt.Errorf("%w: Audit input must use an exact ArtifactRef", ErrInvalid)
		}
		metadata, err := projectArtifacts.Metadata(ctx, ref)
		if err != nil {
			return nil, err
		}
		if !contracts.AcceptsMediaType(profile.Program.Inputs[name].MediaTypes, metadata.MediaType) {
			return nil, fmt.Errorf("%w: Audit input media type is incompatible", ErrInvalid)
		}

		result[name] = auditstore.ExactArtifact{
			Ref: metadata.Ref, Digest: metadata.Digest,
			MediaType: metadata.MediaType, SizeBytes: metadata.Size,
		}
	}
	return result, nil
}

func validateCreateParams(params CreateDraftParams) error {
	if params.AuditID == "" || params.OwnerID == "" || params.ProjectID == "" ||
		!utf8.ValidString(params.OwnerID) || len([]byte(params.OwnerID)) > 256 ||
		params.IdempotencyKey == "" || !contentdigest.Valid(params.RequestDigest) {
		return fmt.Errorf("%w: Audit draft request is invalid", ErrInvalid)
	}
	_, err := normalizeProfileSelector(params.Profile)
	return err
}

func normalizeProfileSelector(value ProfileSelector) (string, error) {
	selector, err := resolved.ParseSelector(value.Name + "@" + value.Version)
	if err != nil || selector.ID != value.Name || selector.Version != value.Version {
		return "", fmt.Errorf("%w: AuditProfile selector is invalid", ErrInvalid)
	}
	return selector.String(), nil
}
