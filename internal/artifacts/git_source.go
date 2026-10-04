package artifacts

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

type GitSource struct {
	RepositoryURL  string    `json:"repositoryUrl"`
	RequestedRef   *string   `json:"requestedRef"`
	ResolvedCommit string    `json:"resolvedCommit"`
	ImportedAt     time.Time `json:"importedAt"`
}

var gitCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

// RecordGitSource is only used by trusted import composition, in the same
// transaction that creates the version. Exact input forks reuse that version.
func (r *PostgresRepository) RecordGitSource(ctx context.Context, scope Scope, ref ArtifactRef, source GitSource) error {
	if _, ok := r.db.(pgx.Tx); !ok {
		return errors.New("Git provenance requires a caller-owned transaction")
	}
	if err := validateScope(scope); err != nil {
		return err
	}
	if err := validateRef(ref); err != nil {
		return err
	}
	if ref.Revision == nil {
		return ErrExactRevisionRequired
	}
	u, err := url.Parse(source.RepositoryURL)
	if err != nil || len(source.RepositoryURL) > 4096 || u.Host == "" || (u.Scheme != "https" && u.Scheme != "ssh") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || (u.Scheme == "https" && u.User != nil) || !gitCommitPattern.MatchString(source.ResolvedCommit) || source.ImportedAt.IsZero() {
		return ErrArtifactIntegrity
	}
	if u.User != nil {
		if _, hasPassword := u.User.Password(); hasPassword {
			return ErrArtifactIntegrity
		}
	}
	if source.RequestedRef != nil && (len(*source.RequestedRef) == 0 || len(*source.RequestedRef) > 1024 || strings.ContainsAny(*source.RequestedRef, "\x00\r\n")) {
		return ErrArtifactIntegrity
	}
	var revisionFound, recorded bool
	err = r.db.QueryRow(ctx, recordGitSourceSQL,
		scope.kind, scope.id, ref.Namespace, ref.Name, *ref.Revision,
		source.RepositoryURL, source.RequestedRef, source.ResolvedCommit, source.ImportedAt,
	).Scan(&revisionFound, &recorded)
	if err != nil {
		return fmt.Errorf("record immutable artifact Git source: %w", err)
	}
	if !revisionFound {
		return ErrArtifactNotFound
	}
	if !recorded {
		return fmt.Errorf("record immutable artifact Git source: %w", ErrArtifactConflict)
	}
	return nil
}

const gitSourceProjection = `CASE WHEN version.git_resolved_commit IS NULL THEN NULL ELSE
jsonb_build_object('repositoryUrl', version.git_repository_url,
                  'requestedRef', version.git_requested_ref,
                  'resolvedCommit', version.git_resolved_commit,
                  'importedAt', version.git_imported_at) END`
