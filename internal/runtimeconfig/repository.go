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

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const maximumPageSize = 200

// Repository contains only transaction-neutral primitives. Construct it with
// either a pgx pool or a caller-owned pgx.Tx.
type Repository struct {
	db persistencepostgres.DBTX
}

func NewRepository(db persistencepostgres.DBTX) *Repository { return &Repository{db: db} }

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
		version.BuiltIn, version.ActorID, persistencepostgres.Timestamp(version.CreatedAt),
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

// InsertPublishedVersion stores a published version together with its
// idempotency receipt. It reports false, without error, when the version or
// the idempotency key already exists; the caller resolves that by replay.
func (r *Repository) InsertPublishedVersion(ctx context.Context, version Version, publication Publication) (bool, error) {
	if err := validateVersion(version); err != nil {
		return false, err
	}
	if !contentdigest.Valid(publication.IdempotencyKeyDigest) || !contentdigest.Valid(publication.RequestDigest) ||
		publication.Ref != version.Ref || publication.ActorID != version.ActorID ||
		!publication.PublishedAt.Equal(version.CreatedAt) || version.BuiltIn {
		return false, invalid("RuntimeConfig publication receipt does not match its version")
	}
	command, err := r.db.Exec(ctx, `
INSERT INTO runtime_config_versions (
    name, version, digest, canonical_document, built_in, actor_id, created_at,
    idempotency_key_digest, request_digest
) VALUES ($1, $2, $3, $4, false, $5, $6, $7, $8)
ON CONFLICT DO NOTHING`,
		version.Ref.Name, version.Ref.Version, version.Ref.Digest, string(version.CanonicalDocument),
		version.ActorID, persistencepostgres.Timestamp(version.CreatedAt),
		publication.IdempotencyKeyDigest, publication.RequestDigest,
	)
	if err != nil {
		return false, classifyWrite(err)
	}
	return command.RowsAffected() == 1, nil
}

func (r *Repository) GetVersion(ctx context.Context, name, version string) (Version, error) {
	if err := validateID("RuntimeConfig name", name, 63); err != nil || !contracts.ValidVersion(version) || len(version) > 128 {
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

// GetVersionsByRefs reads each distinct referenced document once. Callers that
// need a consistent view across bindings and versions must own a transaction.
func (r *Repository) GetVersionsByRefs(ctx context.Context, refs []Ref) (map[Ref]Version, error) {
	result := make(map[Ref]Version, len(refs))
	if len(refs) == 0 {
		return result, nil
	}
	names := make([]string, 0, len(refs))
	versions := make([]string, 0, len(refs))
	digests := make([]string, 0, len(refs))
	requested := make(map[Ref]struct{}, len(refs))
	for _, ref := range refs {
		if err := validateRef(ref); err != nil {
			return nil, err
		}
		if _, exists := requested[ref]; exists {
			continue
		}
		requested[ref] = struct{}{}
		names = append(names, ref.Name)
		versions = append(versions, ref.Version)
		digests = append(digests, ref.Digest)
	}
	rows, err := r.db.Query(ctx, `
SELECT name, version, digest, canonical_document, built_in, actor_id, created_at
FROM runtime_config_versions
JOIN unnest($1::text[], $2::text[], $3::text[]) AS requested(name, version, digest)
  USING (name, version, digest)`, names, versions, digests)
	if err != nil {
		return nil, persistencepostgres.WrapError("read referenced RuntimeConfig versions", err)
	}
	defer rows.Close()
	for rows.Next() {
		version, scanErr := scanVersion(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result[version.Ref] = version
	}
	if err := rows.Err(); err != nil {
		return nil, persistencepostgres.WrapError("iterate referenced RuntimeConfig versions", err)
	}
	if len(result) != len(requested) {
		return nil, ErrVersionNotFound
	}
	return result, nil
}

func (r *Repository) ListVersions(ctx context.Context, afterName, afterVersion string, limit int) ([]Version, error) {
	if (afterName == "") != (afterVersion == "") || limit < 1 || limit > maximumPageSize {
		return nil, invalid("RuntimeConfig page cursor or limit is invalid")
	}
	if afterName != "" {
		if err := validateID("RuntimeConfig cursor name", afterName, 63); err != nil || !contracts.ValidVersion(afterVersion) || len(afterVersion) > 128 {
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
		return nil, persistencepostgres.WrapError("list RuntimeConfig versions", err)
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
		return nil, persistencepostgres.WrapError("iterate RuntimeConfig versions", rows.Err())
	}
	return result, nil
}

func (r *Repository) GetPublication(ctx context.Context, idempotencyKeyDigest string) (Publication, error) {
	if !contentdigest.Valid(idempotencyKeyDigest) {
		return Publication{}, invalid("publication idempotency digest is invalid")
	}
	var publication Publication
	err := r.db.QueryRow(ctx, `
SELECT idempotency_key_digest, request_digest,
       name, version, digest, actor_id, created_at
FROM runtime_config_versions
WHERE idempotency_key_digest = $1`, idempotencyKeyDigest).Scan(
		&publication.IdempotencyKeyDigest, &publication.RequestDigest,
		&publication.Ref.Name, &publication.Ref.Version, &publication.Ref.Digest,
		&publication.ActorID, &publication.PublishedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return Publication{}, ErrNotFound
	}
	if err != nil {
		return Publication{}, persistencepostgres.WrapError("get RuntimeConfig publication", err)
	}
	return publication, nil
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
ON CONFLICT DO NOTHING`, label, ref.Name, ref.Version, ref.Digest, actor, persistencepostgres.Timestamp(at))
	if err != nil {
		return Binding{}, classifyWrite(err)
	}
	if command.RowsAffected() == 1 {
		return r.GetBinding(ctx, label)
	}
	if _, getErr := r.GetBinding(ctx, label); getErr != nil && !errors.Is(getErr, ErrNotFound) {
		return Binding{}, getErr
	}
	return Binding{}, ErrPrecondition
}

func (r *Repository) GetBinding(ctx context.Context, label string) (Binding, error) {
	if err := validateLabel(label); err != nil {
		return Binding{}, err
	}
	return scanBinding(r.db.QueryRow(ctx, bindingSelect+` WHERE label = $1`, label))
}

// GetBindingsByLabels reads distinct label bindings with one query.
func (r *Repository) GetBindingsByLabels(ctx context.Context, labels []string) (map[string]Binding, error) {
	result := make(map[string]Binding, len(labels))
	if len(labels) == 0 {
		return result, nil
	}
	requested := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		if err := validateLabel(label); err != nil {
			return nil, err
		}
		requested[label] = struct{}{}
	}
	rows, err := r.db.Query(ctx, bindingSelect+` WHERE label = ANY($1::text[])`, labels)
	if err != nil {
		return nil, persistencepostgres.WrapError("read RuntimeConfig label bindings", err)
	}
	defer rows.Close()
	for rows.Next() {
		binding, scanErr := scanBinding(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		result[binding.Label] = binding
	}
	if err := rows.Err(); err != nil {
		return nil, persistencepostgres.WrapError("iterate RuntimeConfig label bindings", err)
	}
	if len(result) != len(requested) {
		return nil, ErrUnknownLabel
	}
	return result, nil
}

func (r *Repository) ListBindings(ctx context.Context, afterLabel string, limit int) ([]Binding, error) {
	if limit < 1 || limit > maximumPageSize || (afterLabel != "" && validateLabel(afterLabel) != nil) {
		return nil, invalid("RuntimeConfig binding page is invalid")
	}
	rows, err := r.db.Query(ctx, bindingSelect+` WHERE label > $1 ORDER BY label LIMIT $2`, afterLabel, limit)
	if err != nil {
		return nil, persistencepostgres.WrapError("list RuntimeConfig bindings", err)
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
		return nil, persistencepostgres.WrapError("iterate RuntimeConfig bindings", rows.Err())
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
		label, strconv.FormatUint(expectedRevision, 10), ref.Name, ref.Version, ref.Digest, actor, persistencepostgres.Timestamp(at))
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

// LockBindings holds shared locks in lexical label order until the owning
// transaction settles. Independent pins are compatible, but rebind/delete
// cannot change an observed binding. Callers that mutate bindings must instead
// use LockBindingsForUpdate to avoid shared-to-exclusive lock upgrades.
func (r *Repository) LockBindings(ctx context.Context, labels []string) ([]Binding, error) {
	return r.lockBindings(ctx, labels, "FOR SHARE")
}

// LockBindingsForUpdate preserves exclusive binding-before-principal ordering
// for rebind/delete, including the other labels validated by those mutations.
func (r *Repository) LockBindingsForUpdate(ctx context.Context, labels []string) ([]Binding, error) {
	return r.lockBindings(ctx, labels, "FOR UPDATE")
}

func (r *Repository) lockBindings(ctx context.Context, labels []string, mode string) ([]Binding, error) {
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
		binding, err := scanBinding(r.db.QueryRow(ctx, bindingSelect+` WHERE label = $1 `+mode, label))
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
		return Version{}, persistencepostgres.WrapError("read RuntimeConfig version", err)
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
		return Binding{}, persistencepostgres.WrapError("read RuntimeConfig binding", err)
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

func specEqual(left, right Spec) bool { return reflect.DeepEqual(left, right) }

func classifyWrite(err error) error {
	if class := persistencepostgres.ConstraintError(err, ErrConflict, ErrInvalid); class != nil {
		return persistencepostgres.WrapError(class.Error(), errors.Join(class, err))
	}
	return persistencepostgres.WrapError("persist RuntimeConfig state", err)
}
