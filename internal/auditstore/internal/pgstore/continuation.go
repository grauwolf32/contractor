package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// ContinuationTasks seeks beyond the last consumed source without hydrating
// unrelated Task inputs and parameters. At most 32 exact source projections
// fit the common 8 MiB document ceiling even for large Tasks.
func (s *QueryStore) ContinuationTasks(ctx context.Context, ownerID, auditID, alias string, sources []string, limit int) ([]auditstore.ContinuationSource, error) {
	if limit < 1 || limit > contracts.MaxSelectedMaterials {
		return nil, auditstore.ErrInvalid
	}
	audit, err := s.get(ctx, ownerID, auditID)
	if err != nil {
		return nil, err
	}
	var previous []byte
	err = s.db.QueryRow(ctx, lastProducerTaskSQL, auditID, alias).Scan(&previous)
	after := 0
	if err == nil {
		var source auditstore.TaskSource
		if strictjson.Decode(previous, &source) != nil || source.Kind != "continuation" || !validTaskSource(source) {
			return nil, errStoredTask
		}
		after = source.ThroughOrdinal
	} else if !errors.Is(err, pgx.ErrNoRows) {
		return nil, err
	}
	rows, err := s.db.Query(ctx, continuationTasksSQL, auditID, after, sources, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanContinuationSources(rows, auditID, audit.TaskCount)
}

func scanContinuationSources(rows pgx.Rows, auditID string, taskCount int) ([]auditstore.ContinuationSource, error) {
	selected := []auditstore.ContinuationSource{}
	for rows.Next() {
		var source auditstore.ContinuationSource
		var hypothesis, result []byte
		if err := rows.Scan(&source.TaskID, &source.Ordinal, &hypothesis, &result, &source.GapCode); err != nil {
			return nil, err
		}
		if !contracts.ValidResourceID(source.TaskID) || source.Ordinal < 1 || source.Ordinal > taskCount || source.Ordinal > auditprofile.MaxAuditItemsTotal || source.GapCode != nil && validateText("source gap", *source.GapCode, 128, true) != nil || result == nil && source.GapCode == nil {
			return nil, errStoredTask
		}
		if hypothesis != nil {
			h, err := auditcontract.DecodeHypothesis(hypothesis)
			if err != nil || h.Origin.AuditID != auditID {
				return nil, errStoredTask
			}
			source.Hypothesis = &h
		}
		if result != nil {
			var material auditcontract.Material
			if len(result) > auditstore.MaxArtifactRefBytes || strictjson.Decode(result, &material) != nil || material.ScopeKind != "audit" || material.ScopeID != auditID {
				return nil, errStoredTask
			}
			exact := contracts.ExactArtifact{Ref: material.Ref, Digest: material.Digest, MediaType: material.MediaType, SizeBytes: material.SizeBytes}
			if !exact.HasMetadata() || exact.Validate() != nil {
				return nil, errStoredTask
			}
			source.Result = &material
		}
		selected = append(selected, source)
	}
	return selected, rows.Err()
}

// Exact observation IDs belong to the accepted continuation source. This is a
// point lookup of at most 32 immutable Task snapshots, never a history rescan.
func readTaskObservations(ctx context.Context, db persistencepostgres.DBTX, auditID string, ids []string) ([]auditstore.ContinuationSource, error) {
	if len(ids) == 0 {
		return []auditstore.ContinuationSource{}, nil
	}
	if len(ids) > contracts.MaxSelectedMaterials {
		return nil, auditstore.ErrInvalid
	}
	rows, err := db.Query(ctx, taskObservationsSQL, auditID, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	selected, err := scanContinuationSources(rows, auditID, auditprofile.MaxAuditItemsTotal)
	if err != nil {
		return nil, err
	}
	if len(selected) != len(ids) {
		return nil, errStoredTask
	}
	return selected, nil
}
