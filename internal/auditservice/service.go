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
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

func New(options Options) (*Service, error) {
	if options.Pool == nil || options.Profiles == nil || options.LLMCredentials == nil ||
		options.CredentialGuard == nil || options.RuntimeCredentials == nil {
		return nil, errors.New("Audit service dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Service{
		pool: options.Pool, profiles: options.Profiles,
		llmCredentials:     options.LLMCredentials,
		credentialGuard:    options.CredentialGuard,
		runtimeCredentials: options.RuntimeCredentials,
		now:                options.Now,
	}, nil
}

func (s *Service) Profiles() []ProfileProjection {
	profiles := s.profiles.AuditProfiles()
	result := make([]ProfileProjection, 0, len(profiles))
	for _, profile := range profiles {
		result = append(result, ProfileProjection{
			Profile: profile, Compatibility: ProfileCompatibility(profile),
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
	return ProfileProjection{Profile: profile, Compatibility: ProfileCompatibility(profile)}, nil
}

func (s *Service) CreateDraft(
	ctx context.Context, params CreateDraftParams,
) (auditstore.Audit, bool, error) {
	if err := validateCreateParams(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	store := auditstore.NewPostgresStore(s.pool)
	if replay, found, err := store.LookupMutationReplay(
		ctx, params.OwnerID, auditstore.MutationCreate,
		params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	selector, _ := normalizeProfileSelector(params.Profile)
	profile, err := s.profiles.AuditProfile(selector)
	if err != nil {
		return auditstore.Audit{}, false, ErrProfileNotFound
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
	profileSnapshot, err := json.Marshal(profile)
	if err != nil {
		return auditstore.Audit{}, false, fmt.Errorf("encode AuditProfile snapshot: %w", err)
	}
	selection, err := encodeDraftSelection(DraftSelection{
		Schema: DraftSelectionSchema, Inputs: inputs,
		RuntimeLabels: labels, Scope: scope,
	})
	if err != nil {
		return auditstore.Audit{}, false, err
	}
	return store.CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: params.AuditID, OwnerID: params.OwnerID, ProjectID: params.ProjectID,
		Profile: auditstore.ProfileIdentity{
			Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest,
		},
		ProfileSnapshot: profileSnapshot, InputSelection: selection,
		Limits: auditstore.Limits{
			MaxRounds: profile.Execution.MaxRounds, BatchSize: profile.Execution.BatchSize,
			MaxItemsPerRound:   profile.Execution.MaxItemsPerRound,
			MaxItemsTotal:      profile.Execution.MaxItemsTotal,
			MaxSubmittedRuns:   profile.Execution.MaxSubmittedRuns,
			MaxItemRunAttempts: profile.Execution.MaxItemRunAttempts,
			MaxEvidenceBytes:   profile.Execution.MaxEvidenceBytes,
		},
		IdempotencyKey: params.IdempotencyKey, RequestDigest: params.RequestDigest,
	})
}

func (s *Service) Get(
	ctx context.Context, ownerID, auditID string,
) (auditstore.Audit, error) {
	return auditstore.NewPostgresStore(s.pool).Get(ctx, ownerID, auditID)
}

func (s *Service) List(
	ctx context.Context, params auditstore.ListParams,
) ([]auditstore.Audit, error) {
	return auditstore.NewPostgresStore(s.pool).List(ctx, params)
}

func (s *Service) ListItems(
	ctx context.Context, params auditstore.ListItemsParams,
) ([]auditstore.Item, error) {
	return auditstore.NewPostgresStore(s.pool).ListItemsPage(ctx, params)
}

func (s *Service) GetRound(
	ctx context.Context, ownerID, auditID, roundID string,
) (auditstore.Round, error) {
	store := auditstore.NewPostgresStore(s.pool)
	if _, err := store.Get(ctx, ownerID, auditID); err != nil {
		return auditstore.Round{}, err
	}
	return store.GetRound(ctx, auditID, roundID)
}

func (s *Service) ListCoverage(
	ctx context.Context,
	ownerID, auditID, roundID string,
	afterOrdinal, limit int,
) ([]auditstore.CoverageRow, error) {
	store := auditstore.NewPostgresStore(s.pool)
	if _, err := store.Get(ctx, ownerID, auditID); err != nil {
		return nil, err
	}
	return store.ListCoverage(ctx, auditID, roundID, afterOrdinal, limit)
}

func (s *Service) GetReport(
	ctx context.Context, ownerID, auditID string,
) (ReportProjection, error) {
	store := auditstore.NewPostgresStore(s.pool)
	audit, err := store.Get(ctx, ownerID, auditID)
	if err != nil {
		return ReportProjection{}, err
	}
	machine, machineErr := store.GetArtifactLink(ctx, auditID, auditstore.ReportMachineLogicalKey)
	summary, summaryErr := store.GetArtifactLink(ctx, auditID, auditstore.ReportSummaryLogicalKey)
	if errors.Is(machineErr, auditstore.ErrNotFound) && errors.Is(summaryErr, auditstore.ErrNotFound) {
		switch audit.State {
		case auditstore.AuditFailed, auditstore.AuditCancelled:
			return ReportProjection{Status: ReportUnavailable}, nil
		case auditstore.AuditCompleted:
			return ReportProjection{}, errors.New("completed Audit has no committed report links")
		default:
			return ReportProjection{Status: ReportPending}, nil
		}
	}
	if machineErr != nil || summaryErr != nil {
		return ReportProjection{}, errors.New("stored Audit report link set is incomplete")
	}
	project, err := artifacts.NewService(artifacts.NewPostgresRepository(s.pool)).Project(audit.ProjectID)
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
	if machine.Artifact.MediaType != "application/json" || summary.Artifact.MediaType != "text/plain" ||
		digestBytes(machineRead.Payload.Data) != machine.Artifact.Digest ||
		digestBytes(summaryRead.Payload.Data) != summary.Artifact.Digest ||
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
	machineArtifact, summaryArtifact := machine.Artifact, summary.Artifact
	return ReportProjection{
		Status: ReportReady, MachineArtifact: &machineArtifact, SummaryArtifact: &summaryArtifact,
		Machine: append(json.RawMessage(nil), machineRead.Payload.Data...), Summary: string(summaryRead.Payload.Data),
	}, nil
}

func (s *Service) selectDraftInputs(
	ctx context.Context,
	profile config.ResolvedAuditProfile,
	projectID string,
	selected map[string]contracts.ArtifactRef,
) (map[string]auditstore.ExactArtifact, error) {
	if selected == nil || len(selected) > config.MaxAuditProfileInputs {
		return nil, fmt.Errorf("%w: Audit inputs are invalid", ErrInvalid)
	}
	for name := range selected {
		if _, ok := profile.Inputs[name]; !ok {
			return nil, fmt.Errorf("%w: Audit input slot is unknown", ErrInvalid)
		}
	}
	for name, contract := range profile.Inputs {
		if _, ok := selected[name]; contract.Required && !ok {
			return nil, fmt.Errorf("%w: required Audit input is missing", ErrInvalid)
		}
	}
	projectArtifacts, err := artifacts.NewService(
		artifacts.NewPostgresRepository(s.pool),
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
		if !acceptsMediaType(profile.Inputs[name].MediaTypes, metadata.MediaType) {
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
		params.IdempotencyKey == "" || !validDigest(params.RequestDigest) {
		return fmt.Errorf("%w: Audit draft request is invalid", ErrInvalid)
	}
	_, err := normalizeProfileSelector(params.Profile)
	return err
}

func normalizeProfileSelector(value ProfileSelector) (string, error) {
	selector, err := config.ParseSelector(value.Name + "@" + value.Version)
	if err != nil || selector.ID != value.Name || selector.Version != value.Version {
		return "", fmt.Errorf("%w: AuditProfile selector is invalid", ErrInvalid)
	}
	return selector.String(), nil
}

func acceptsMediaType(accepted []string, actual string) bool {
	for _, candidate := range accepted {
		if candidate == actual {
			return true
		}
	}
	return false
}
