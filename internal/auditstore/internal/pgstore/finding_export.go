package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// FindingExportScope holds the parent against lifecycle transitions while a
// bounded synchronous export publishes in its managed Artifact namespace.
func (s *QueryStore) FindingExportScope(ctx context.Context, ownerID, auditID string, lock bool) (string, error) {
	query := findingExportScopeSQL
	if lock {
		query += " FOR SHARE"
	}
	var projectID string
	err := s.db.QueryRow(ctx, query, ownerID, auditID).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", auditstore.ErrNotFound
	}
	return projectID, err
}

// ExportFindings reads only explicitly selected findings in one statement. The
// caller checks missing identities/revisions before expanding contributions.
func (s *QueryStore) ExportFindings(ctx context.Context, ownerID, auditID string, ids []string) (map[string]auditstore.ProjectFinding, error) {
	if len(ids) > auditdomain.MaximumCollectionEntries {
		return nil, auditstore.ErrInvalid
	}
	result := make(map[string]auditstore.ProjectFinding, len(ids))
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, exportFindingsSQL, ownerID, auditID, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		value, err := scanProjectFinding(rows)
		if err != nil {
			return nil, err
		}
		result[value.FindingID] = value
	}
	return result, rows.Err()
}

// ExportContributions bounds each indexed finding probe and the enclosing
// explicit selection; one extra distinct receipt makes the selection invalid.
func (s *QueryStore) ExportContributions(ctx context.Context, auditID string, ids []string) (map[string][]string, error) {
	if len(ids) > auditdomain.MaximumCollectionEntries {
		return nil, auditstore.ErrInvalid
	}
	result := make(map[string][]string, len(ids))
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, exportContributionsSQL, auditID, ids, auditdomain.MaximumCollectionEntries+1)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	seen := map[string]bool{}
	for rows.Next() {
		var finding, receipt string
		if err := rows.Scan(&finding, &receipt); err != nil {
			return nil, err
		}
		seen[receipt] = true
		if len(seen) > auditdomain.MaximumCollectionEntries {
			return nil, auditstore.ErrInvalid
		}
		result[finding] = append(result[finding], receipt)
	}
	return result, rows.Err()
}
