package runtimeconfig

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
)

const maximumPageSize = 200

// Repository contains only transaction-neutral primitives. Construct it with
// either a pgx pool or a caller-owned pgx.Tx.
type Repository struct {
	db persistencepostgres.DBTX
}

func NewRepository(db persistencepostgres.DBTX) *Repository { return &Repository{db: db} }

// Store is implemented by both pool-backed and transaction-backed
// repositories. LockBindings is meaningful with a caller-owned transaction.
type Store interface {
	InsertVersion(context.Context, Version) (bool, error)
	GetVersion(context.Context, string, string) (Version, error)
	GetVersionByRef(context.Context, Ref) (Version, error)
	ListVersions(context.Context, string, string, int) ([]Version, error)
	GetPublication(context.Context, string) (Publication, error)
	InsertPublication(context.Context, Publication) (bool, error)
	CreateBinding(context.Context, string, Ref, string, time.Time) (Binding, error)
	GetBinding(context.Context, string) (Binding, error)
	ListBindings(context.Context, string, int) ([]Binding, error)
	Rebind(context.Context, string, uint64, Ref, string, time.Time) (Binding, error)
	DeleteBinding(context.Context, string, uint64) error
	LockBindings(context.Context, []string) ([]Binding, error)
}

var _ Store = (*Repository)(nil)

func (r *Repository) InsertVersion(ctx context.Context, version Version) (bool, error) {
	if err := validateVersion(version); err != nil {
		return false, err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_config_versions (
    name, version, digest, canonical_document, built_in, actor_id, created_at
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT DO NOTHING`,
		version.Ref.Name, version.Ref.Version, version.Ref.Digest, string(version.CanonicalDocument),
		version.BuiltIn, version.ActorID, databaseTime(version.CreatedAt),
	)
	if err != nil {
		return false, classifyWrite(err)
	}
	if command.RowsAffected() != 1 {
		existing, getErr := r.GetVersion(ctx, version.Ref.Name, version.Ref.Version)
		if getErr == nil && existing.Ref == version.Ref && existing.BuiltIn == version.BuiltIn &&
			bytes.Equal(existing.CanonicalDocument, version.CanonicalDocument) {
			return false, nil
		}
		return false, ErrConflict
	}
	return true, nil
}

func (r *Repository) GetVersion(ctx context.Context, name, version string) (Version, error) {
	if err := validateID("RuntimeConfig name", name, 63); err != nil || !versionPattern.MatchString(version) || len(version) > 128 {
		return Version{}, invalid("RuntimeConfig identity is invalid")
	}
	return scanVersion(r.db.QueryRow(ctx, `
SELECT name, version, digest, canonical_document, built_in, actor_id, created_at
FROM runtime_config_versions
WHERE name = $1 AND version = $2`, name, version))
}

func (r *Repository) GetVersionByRef(ctx context.Context, ref Ref) (Version, error) {
	if err := validateRef(ref); err != nil {
		return Version{}, err
	}
	return scanVersion(r.db.QueryRow(ctx, `
SELECT name, version, digest, canonical_document, built_in, actor_id, created_at
FROM runtime_config_versions
WHERE name = $1 AND version = $2 AND digest = $3`, ref.Name, ref.Version, ref.Digest))
}

func (r *Repository) ListVersions(ctx context.Context, afterName, afterVersion string, limit int) ([]Version, error) {
	if (afterName == "") != (afterVersion == "") || limit < 1 || limit > maximumPageSize {
		return nil, invalid("RuntimeConfig page cursor or limit is invalid")
	}
	if afterName != "" {
		if err := validateID("RuntimeConfig cursor name", afterName, 63); err != nil || !versionPattern.MatchString(afterVersion) || len(afterVersion) > 128 {
			return nil, invalid("RuntimeConfig page cursor is invalid")
		}
	}
	rows, err := r.db.Query(ctx, `
SELECT name, version, digest, canonical_document, built_in, actor_id, created_at
FROM runtime_config_versions
WHERE (name, version) > ($1, $2)
ORDER BY name, version
LIMIT $3`, afterName, afterVersion, limit)
	if err != nil {
		return nil, errors.New("list RuntimeConfig versions")
	}
	defer rows.Close()
	result := make([]Version, 0)
	for rows.Next() {
		item, scanErr := scanVersion(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, item)
	}
	if rows.Err() != nil {
		return nil, errors.New("iterate RuntimeConfig versions")
	}
	return result, nil
}

func (r *Repository) GetPublication(ctx context.Context, idempotencyKeyDigest string) (Publication, error) {
	if !digestPattern.MatchString(idempotencyKeyDigest) {
		return Publication{}, invalid("publication idempotency digest is invalid")
	}
	var publication Publication
	err := r.db.QueryRow(ctx, `
SELECT idempotency_key_digest, request_digest,
       config_name, config_version, config_digest, actor_id, published_at
FROM runtime_config_publications
WHERE idempotency_key_digest = $1`, idempotencyKeyDigest).Scan(
		&publication.IdempotencyKeyDigest, &publication.RequestDigest,
		&publication.Ref.Name, &publication.Ref.Version, &publication.Ref.Digest,
		&publication.ActorID, &publication.PublishedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Publication{}, ErrNotFound
	}
	if err != nil {
		return Publication{}, errors.New("get RuntimeConfig publication")
	}
	return publication, nil
}

func (r *Repository) InsertPublication(ctx context.Context, publication Publication) (bool, error) {
	if !digestPattern.MatchString(publication.IdempotencyKeyDigest) || !digestPattern.MatchString(publication.RequestDigest) ||
		validateRef(publication.Ref) != nil || !validActor(publication.ActorID) || publication.PublishedAt.IsZero() {
		return false, invalid("RuntimeConfig publication audit is invalid")
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_config_publications (
    idempotency_key_digest, request_digest,
    config_name, config_version, config_digest, actor_id, published_at
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT DO NOTHING`,
		publication.IdempotencyKeyDigest, publication.RequestDigest,
		publication.Ref.Name, publication.Ref.Version, publication.Ref.Digest,
		publication.ActorID, databaseTime(publication.PublishedAt),
	)
	if err != nil {
		return false, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return true, nil
	}
	existing, getErr := r.GetPublication(ctx, publication.IdempotencyKeyDigest)
	if getErr == nil && existing.RequestDigest == publication.RequestDigest && existing.Ref == publication.Ref {
		return false, nil
	}
	return false, ErrConflict
}

func (r *Repository) CreateBinding(ctx context.Context, label string, ref Ref, actor string, at time.Time) (Binding, error) {
	if err := validateBindingMutation(label, ref, actor, at); err != nil {
		return Binding{}, err
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_label_bindings (
    label, config_name, config_version, config_digest, revision,
    created_by, created_at, updated_by, updated_at
) VALUES ($1, $2, $3, $4, 1, $5, $6, $5, $6)
ON CONFLICT DO NOTHING`, label, ref.Name, ref.Version, ref.Digest, actor, databaseTime(at))
	if err != nil {
		return Binding{}, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return r.GetBinding(ctx, label)
	}
	existing, getErr := r.GetBinding(ctx, label)
	if getErr == nil && existing.Ref == ref {
		return existing, nil
	}
	return Binding{}, ErrConflict
}

func (r *Repository) GetBinding(ctx context.Context, label string) (Binding, error) {
	if err := validateLabel(label); err != nil {
		return Binding{}, err
	}
	return scanBinding(r.db.QueryRow(ctx, bindingSelect+` WHERE label = $1`, label))
}

func (r *Repository) ListBindings(ctx context.Context, afterLabel string, limit int) ([]Binding, error) {
	if limit < 1 || limit > maximumPageSize || (afterLabel != "" && validateLabel(afterLabel) != nil) {
		return nil, invalid("RuntimeConfig binding page is invalid")
	}
	rows, err := r.db.Query(ctx, bindingSelect+` WHERE label > $1 ORDER BY label LIMIT $2`, afterLabel, limit)
	if err != nil {
		return nil, errors.New("list RuntimeConfig bindings")
	}
	defer rows.Close()
	result := make([]Binding, 0)
	for rows.Next() {
		item, scanErr := scanBinding(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result = append(result, item)
	}
	if rows.Err() != nil {
		return nil, errors.New("iterate RuntimeConfig bindings")
	}
	return result, nil
}

func (r *Repository) Rebind(ctx context.Context, label string, expectedRevision uint64, ref Ref, actor string, at time.Time) (Binding, error) {
	if expectedRevision == 0 {
		return Binding{}, invalid("expected RuntimeConfig binding revision is required")
	}
	if expectedRevision == ^uint64(0) {
		return Binding{}, ErrConflict
	}
	if err := validateBindingMutation(label, ref, actor, at); err != nil {
		return Binding{}, err
	}
	command, err := r.db.Exec(ctx, `
UPDATE runtime_label_bindings
SET config_name = $3, config_version = $4, config_digest = $5,
    revision = revision + 1, updated_by = $6, updated_at = $7
WHERE label = $1 AND revision = $2::numeric
  AND (config_name, config_version, config_digest) IS DISTINCT FROM ($3, $4, $5)`,
		label, strconv.FormatUint(expectedRevision, 10), ref.Name, ref.Version, ref.Digest, actor, databaseTime(at))
	if err != nil {
		return Binding{}, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return r.GetBinding(ctx, label)
	}
	existing, getErr := r.GetBinding(ctx, label)
	if errors.Is(getErr, ErrNotFound) {
		return Binding{}, ErrNotFound
	}
	if getErr != nil {
		return Binding{}, getErr
	}
	if existing.Revision != expectedRevision {
		return Binding{}, ErrPrecondition
	}
	if existing.Ref == ref {
		return existing, nil
	}
	return Binding{}, ErrPrecondition
}

func (r *Repository) DeleteBinding(ctx context.Context, label string, expectedRevision uint64) error {
	if err := validateLabel(label); err != nil || expectedRevision == 0 {
		return invalid("RuntimeConfig binding delete is invalid")
	}
	if label == DefaultLabel {
		return ErrReserved
	}
	command, err := r.db.Exec(ctx, `
DELETE FROM runtime_label_bindings WHERE label = $1 AND revision = $2::numeric`,
		label, strconv.FormatUint(expectedRevision, 10))
	if err != nil {
		return classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return nil
	}
	existing, getErr := r.GetBinding(ctx, label)
	if errors.Is(getErr, ErrNotFound) {
		return ErrNotFound
	}
	if getErr != nil {
		return getErr
	}
	if existing.Revision != expectedRevision {
		return ErrPrecondition
	}
	return ErrConflict
}

// LockBindings locks each requested row in lexical label order. It is intended
// for caller-owned transactions that compose Run pinning with other stores.
func (r *Repository) LockBindings(ctx context.Context, labels []string) ([]Binding, error) {
	ordered := append([]string(nil), labels...)
	for _, label := range ordered {
		if err := validateLabel(label); err != nil {
			return nil, err
		}
	}
	sort.Strings(ordered)
	for index := 1; index < len(ordered); index++ {
		if ordered[index] == ordered[index-1] {
			return nil, invalid("RuntimeConfig binding lock set contains a duplicate label")
		}
	}
	result := make([]Binding, 0, len(ordered))
	for _, label := range ordered {
		binding, err := scanBinding(r.db.QueryRow(ctx, bindingSelect+` WHERE label = $1 FOR UPDATE`, label))
		if err != nil {
			return nil, err
		}
		result = append(result, binding)
	}
	return result, nil
}

const bindingSelect = `
SELECT label, config_name, config_version, config_digest, revision::text,
       created_by, created_at, updated_by, updated_at
FROM runtime_label_bindings`

type rowScanner interface{ Scan(...any) error }

func scanVersion(row rowScanner) (Version, error) {
	var result Version
	var canonical string
	err := row.Scan(&result.Ref.Name, &result.Ref.Version, &result.Ref.Digest, &canonical,
		&result.BuiltIn, &result.ActorID, &result.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return Version{}, ErrNotFound
	}
	if err != nil {
		return Version{}, errors.New("read RuntimeConfig version")
	}
	decoded, err := DecodeStoredDocument([]byte(canonical))
	if err != nil || decoded.Ref != result.Ref || decoded.BuiltIn != result.BuiltIn {
		return Version{}, errors.New("stored RuntimeConfig version failed integrity validation")
	}
	result.Spec = decoded.Spec
	result.CanonicalDocument = decoded.CanonicalDocument
	return result, nil
}

func scanBinding(row rowScanner) (Binding, error) {
	var result Binding
	var revision string
	err := row.Scan(&result.Label, &result.Ref.Name, &result.Ref.Version, &result.Ref.Digest,
		&revision, &result.CreatedBy, &result.CreatedAt, &result.UpdatedBy, &result.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return Binding{}, ErrNotFound
	}
	if err != nil {
		return Binding{}, errors.New("read RuntimeConfig binding")
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil {
		return Binding{}, errors.New("stored RuntimeConfig binding revision is invalid")
	}
	result.Revision = parsed
	return result, nil
}

func validateVersion(version Version) error {
	if err := validateRef(version.Ref); err != nil || !validActor(version.ActorID) || version.CreatedAt.IsZero() {
		return invalid("RuntimeConfig version metadata is invalid")
	}
	decoded, err := DecodeStoredDocument(version.CanonicalDocument)
	if err != nil || decoded.Ref != version.Ref || decoded.BuiltIn != version.BuiltIn || !specEqual(decoded.Spec, version.Spec) {
		return invalid("RuntimeConfig normalized document does not match its metadata")
	}
	if version.BuiltIn && (version.Ref.Name != BuiltInName || version.Ref.Version != BuiltInVersion || version.Ref.Digest != BuiltInDigest || !bytes.Equal(version.CanonicalDocument, []byte(BuiltInCanonicalDocument))) {
		return invalid("built-in RuntimeConfig does not match the fixed identity")
	}
	return nil
}

func validateBindingMutation(label string, ref Ref, actor string, at time.Time) error {
	if err := validateLabel(label); err != nil || validateRef(ref) != nil || !validActor(actor) || at.IsZero() {
		return invalid("RuntimeConfig binding mutation is invalid")
	}
	return nil
}

func validActor(actor string) bool { return strings.TrimSpace(actor) != "" && len(actor) <= 256 }

func databaseTime(value time.Time) time.Time { return value.UTC().Truncate(time.Microsecond) }

func specEqual(left, right Spec) bool { return reflect.DeepEqual(left, right) }

func classifyWrite(err error) error {
	var postgresError *pgconn.PgError
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case "23505":
			return ErrConflict
		case "23503", "23514", "22001", "22P02":
			return ErrInvalid
		}
	}
	return errors.New("persist RuntimeConfig state")
}
