package postgres

import (
	"context"
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/contracts"
)

var gitCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

// RecordGitSource is only used by trusted import composition, in the same
// transaction that creates the version. Exact input forks reuse that version.
func (r *TxRepository) RecordGitSource(ctx context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef, source artifacts.GitSource) error {
	if err := artifacts.ValidateScope(scope); err != nil {
		return err
	}
	if err := artifacts.ValidateRef(ref); err != nil {
		return err
	}
	if ref.Revision == nil {
		return artifacts.ErrExactRevisionRequired
	}
	u, err := url.Parse(source.RepositoryURL)
	if err != nil || len(source.RepositoryURL) > contracts.MaxGitSourceURLBytes || u.Host == "" || (u.Scheme != "https" && u.Scheme != "ssh") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || (u.Scheme == "https" && u.User != nil) || !gitCommitPattern.MatchString(source.ResolvedCommit) || source.ImportedAt.IsZero() {
		return blob.ErrIntegrity
	}
	if u.User != nil {
		if _, hasPassword := u.User.Password(); hasPassword {
			return blob.ErrIntegrity
		}
	}
	if source.RequestedRef != nil && (len(*source.RequestedRef) == 0 || len(*source.RequestedRef) > 1024 || strings.ContainsAny(*source.RequestedRef, "\x00\r\n")) {
		return blob.ErrIntegrity
	}
	if source.Analysis != nil && !source.Analysis.Valid() {
		return blob.ErrIntegrity
	}
	var revisionFound, recorded bool
	err = r.tx.QueryRow(ctx, recordGitSourceSQL,
		scope.Kind(), scope.ID(), ref.Namespace, ref.Name, *ref.Revision,
		source.RepositoryURL, source.RequestedRef, source.ResolvedCommit, source.ImportedAt, source.Analysis,
	).Scan(&revisionFound, &recorded)
	if err != nil {
		return fmt.Errorf("record immutable artifact Git source: %w", err)
	}
	if !revisionFound {
		return artifacts.ErrArtifactNotFound
	}
	if !recorded {
		return fmt.Errorf("record immutable artifact Git source: %w", artifacts.ErrArtifactConflict)
	}
	return nil
}

const gitSourceProjection = `CASE WHEN version.git_resolved_commit IS NULL THEN NULL ELSE
jsonb_build_object('repositoryUrl', version.git_repository_url,
                  'requestedRef', version.git_requested_ref,
                  'resolvedCommit', version.git_resolved_commit,
                  'importedAt', version.git_imported_at) ||
CASE WHEN version.git_analysis IS NULL THEN '{}'::jsonb ELSE jsonb_build_object('analysis', version.git_analysis) END END`
