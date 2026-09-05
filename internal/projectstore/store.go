package projectstore

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

var (
	resourceIDPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`)
	idempotencyKeyPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)
	digestPattern         = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

type Repository interface {
	Create(context.Context, CreateParams) (Project, bool, error)
	Get(context.Context, string, string) (Project, error)
	List(context.Context, ListParams) ([]Project, error)
	Update(context.Context, UpdateParams) (Project, error)
}

type PostgresStore struct{ db persistencepostgres.DBTX }

var _ Repository = (*PostgresStore)(nil)

func NewPostgresStore(db persistencepostgres.DBTX) *PostgresStore {
	return &PostgresStore{db: db}
}

func (s *PostgresStore) Create(ctx context.Context, params CreateParams) (Project, bool, error) {
	if err := validateCreate(params); err != nil {
		return Project{}, false, err
	}
	project, err := scanProject(s.db.QueryRow(ctx, `
INSERT INTO projects (
    project_id, owner_id, kind, name, description,
    request_idempotency_key, request_digest
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT DO NOTHING
RETURNING project_id, owner_id, kind, name, description,
          http_target_url, http_target_credential_id, http_target_credential_kind,
          revision, created_at, updated_at`,
		params.ProjectID, params.OwnerID, params.Kind, params.Name, params.Description,
		params.IdempotencyKey, params.RequestDigest,
	))
	if err == nil {
		return project, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Project{}, false, fmt.Errorf("create Project: %w", err)
	}

	var existingID, existingDigest string
	err = s.db.QueryRow(ctx, `
SELECT project_id, request_digest
FROM projects
WHERE owner_id = $1 AND request_idempotency_key = $2`,
		params.OwnerID, params.IdempotencyKey,
	).Scan(&existingID, &existingDigest)
	if errors.Is(err, pgx.ErrNoRows) {
		return Project{}, false, ErrConflict
	}
	if err != nil {
		return Project{}, false, fmt.Errorf("resolve Project idempotency: %w", err)
	}
	if existingDigest != params.RequestDigest {
		return Project{}, false, ErrConflict
	}
	project, err = s.Get(ctx, params.OwnerID, existingID)
	return project, false, err
}

func (s *PostgresStore) Get(ctx context.Context, ownerID, projectID string) (Project, error) {
	if err := validateIdentity(ownerID, projectID); err != nil {
		return Project{}, err
	}
	project, err := scanProject(s.db.QueryRow(ctx, `
SELECT project_id, owner_id, kind, name, description,
       http_target_url, http_target_credential_id, http_target_credential_kind,
       revision, created_at, updated_at
FROM projects
WHERE owner_id = $1 AND project_id = $2`, ownerID, projectID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Project{}, ErrNotFound
	}
	if err != nil {
		return Project{}, fmt.Errorf("read Project: %w", err)
	}
	return project, nil
}

func (s *PostgresStore) List(ctx context.Context, params ListParams) ([]Project, error) {
	if err := validateList(params); err != nil {
		return nil, err
	}
	var kind *string
	if params.Kind != nil {
		value := string(*params.Kind)
		kind = &value
	}
	rows, err := s.db.Query(ctx, `
SELECT project_id, owner_id, kind, name, description,
       http_target_url, http_target_credential_id, http_target_credential_kind,
       revision, created_at, updated_at
FROM projects
WHERE owner_id = $1
  AND ($2::text IS NULL OR kind = $2)
  AND ($3::timestamptz IS NULL OR (created_at, project_id) < ($3, $4))
ORDER BY created_at DESC, project_id DESC
LIMIT $5`, params.OwnerID, kind, params.BeforeCreatedAt, params.BeforeProjectID, params.Limit)
	if err != nil {
		return nil, fmt.Errorf("list Projects: %w", err)
	}
	defer rows.Close()
	projects := make([]Project, 0)
	for rows.Next() {
		project, scanErr := scanProject(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan Project page: %w", scanErr)
		}
		projects = append(projects, project)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Project page: %w", err)
	}
	return projects, nil
}

func (s *PostgresStore) Update(ctx context.Context, params UpdateParams) (Project, error) {
	if err := validateUpdate(params); err != nil {
		return Project{}, err
	}
	project, err := scanProject(s.db.QueryRow(ctx, `
UPDATE projects
SET name = $1, description = $2,
    http_target_url = $3, http_target_credential_id = $4, http_target_credential_kind = $5,
    revision = revision + 1,
    updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
WHERE owner_id = $6 AND project_id = $7 AND revision = $8
RETURNING project_id, owner_id, kind, name, description,
          http_target_url, http_target_credential_id, http_target_credential_kind,
          revision, created_at, updated_at`,
		params.Name, params.Description, targetURL(params.HTTPTarget), targetCredentialID(params.HTTPTarget),
		targetCredentialKind(params.HTTPTarget), params.OwnerID, params.ProjectID, params.ExpectedRevision,
	))
	if err == nil {
		return project, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Project{}, fmt.Errorf("update Project: %w", err)
	}
	var exists bool
	err = s.db.QueryRow(ctx, `
SELECT EXISTS (SELECT 1 FROM projects WHERE owner_id = $1 AND project_id = $2)`,
		params.OwnerID, params.ProjectID,
	).Scan(&exists)
	if err != nil {
		return Project{}, fmt.Errorf("resolve Project update: %w", err)
	}
	if !exists {
		return Project{}, ErrNotFound
	}
	return Project{}, ErrPrecondition
}

type scanner interface{ Scan(...any) error }

func scanProject(row scanner) (Project, error) {
	var project Project
	var revision int64
	var targetURL, credentialID, credentialKind *string
	err := row.Scan(
		&project.ProjectID, &project.OwnerID, &project.Kind, &project.Name,
		&project.Description, &targetURL, &credentialID, &credentialKind,
		&revision, &project.CreatedAt, &project.UpdatedAt,
	)
	if err == nil {
		if revision <= 0 {
			return Project{}, errors.New("stored Project revision is invalid")
		}
		project.Revision = uint64(revision)
		if targetURL != nil {
			project.HTTPTarget = &contracts.HTTPOriginTargetRef{URL: *targetURL}
			if credentialID != nil && credentialKind != nil {
				project.HTTPTarget.Credential = &contracts.RuntimeCredentialRefV2{
					CredentialID: *credentialID, Kind: contracts.RuntimeCredentialKind(*credentialKind),
				}
			}
			if validateErr := project.HTTPTarget.Validate(); validateErr != nil {
				return Project{}, errors.New("stored Project HTTP target is invalid")
			}
		}
	}
	return project, err
}

func targetURL(target *contracts.HTTPOriginTargetRef) *string {
	if target == nil {
		return nil
	}
	value := target.URL
	return &value
}

func targetCredentialID(target *contracts.HTTPOriginTargetRef) *string {
	if target == nil || target.Credential == nil {
		return nil
	}
	value := target.Credential.CredentialID
	return &value
}

func targetCredentialKind(target *contracts.HTTPOriginTargetRef) *string {
	if target == nil || target.Credential == nil {
		return nil
	}
	value := string(target.Credential.Kind)
	return &value
}

func validateCreate(params CreateParams) error {
	if err := validateIdentity(params.OwnerID, params.ProjectID); err != nil {
		return err
	}
	if !params.Kind.Valid() {
		return invalid("Project kind is invalid")
	}
	if err := validateMetadata(params.Name, params.Description); err != nil {
		return err
	}
	if !idempotencyKeyPattern.MatchString(params.IdempotencyKey) || !digestPattern.MatchString(params.RequestDigest) {
		return invalid("Project idempotency identity is invalid")
	}
	return nil
}

func validateUpdate(params UpdateParams) error {
	if err := validateIdentity(params.OwnerID, params.ProjectID); err != nil {
		return err
	}
	if params.ExpectedRevision == 0 || params.ExpectedRevision > uint64(^uint64(0)>>1) {
		return invalid("Project revision is invalid")
	}
	if err := validateMetadata(params.Name, params.Description); err != nil {
		return err
	}
	if params.HTTPTarget != nil {
		if err := params.HTTPTarget.Validate(); err != nil {
			return invalid("Project HTTP target is invalid")
		}
	}
	return nil
}

func validateList(params ListParams) error {
	if strings.TrimSpace(params.OwnerID) == "" || len(params.OwnerID) > 256 || strings.IndexByte(params.OwnerID, 0) >= 0 {
		return invalid("Project owner is invalid")
	}
	if params.Kind != nil && !params.Kind.Valid() {
		return invalid("Project kind filter is invalid")
	}
	if params.Limit < 1 || params.Limit > MaxPageSize {
		return invalid("Project page limit is invalid")
	}
	if (params.BeforeCreatedAt == nil) != (params.BeforeProjectID == "") {
		return invalid("Project page cursor is invalid")
	}
	if params.BeforeProjectID != "" && !resourceIDPattern.MatchString(params.BeforeProjectID) {
		return invalid("Project page cursor is invalid")
	}
	return nil
}

func validateIdentity(ownerID, projectID string) error {
	if strings.TrimSpace(ownerID) == "" || len(ownerID) > 256 || strings.IndexByte(ownerID, 0) >= 0 ||
		!resourceIDPattern.MatchString(projectID) {
		return invalid("Project identity is invalid")
	}
	return nil
}

func validateMetadata(name, description string) error {
	if !utf8.ValidString(name) || strings.TrimSpace(name) == "" || len(name) > MaxNameBytes ||
		strings.IndexByte(name, 0) >= 0 || !utf8.ValidString(description) ||
		len(description) > MaxDescriptionBytes || strings.IndexByte(description, 0) >= 0 {
		return invalid("Project metadata is invalid")
	}
	return nil
}

func invalid(message string) error { return fmt.Errorf("%w: %s", ErrInvalid, message) }
