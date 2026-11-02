package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func (s *QueryStore) DependencyResultSource(ctx context.Context, auditID, executionID, subjectKey string) (auditstore.DependencyResultSource, error) {
	for _, id := range []string{auditID, executionID} {
		if validateID("dependency source", id) != nil {
			return auditstore.DependencyResultSource{}, auditstore.ErrInvalid
		}
	}
	if validateText("dependency subject", subjectKey, 512, true) != nil {
		return auditstore.DependencyResultSource{}, auditstore.ErrInvalid
	}
	rows, err := s.db.Query(ctx, dependencyResultSourceSQL, auditID, executionID, subjectKey)
	if err != nil {
		return auditstore.DependencyResultSource{}, err
	}
	defer rows.Close()
	var result auditstore.DependencyResultSource
	count := 0
	for rows.Next() {
		count++
		if count > 1 {
			return auditstore.DependencyResultSource{}, auditstore.ErrDependencySourceAmbiguous
		}
		var sourceRef, retainedRef, links []byte
		var overflow bool
		if err := rows.Scan(&result.ExecutionID, &result.RoundID, &result.ItemID, &result.ItemAttempt, &result.RunID, &sourceRef, &result.Source.Digest, &retainedRef, &result.Retained.Digest, &result.Retained.MediaType, &result.Retained.SizeBytes, &links, &overflow); err != nil {
			return auditstore.DependencyResultSource{}, err
		}
		if overflow {
			return auditstore.DependencyResultSource{}, auditstore.ErrDependencyClosureLimit
		}
		if len(links) > 1<<20 || strictjson.Decode(sourceRef, &result.Source.Ref) != nil || strictjson.Decode(retainedRef, &result.Retained.Ref) != nil || strictjson.Decode(links, &result.Evidence) != nil || len(result.Evidence) > auditstore.MaxArtifactLinksPerCall {
			return auditstore.DependencyResultSource{}, auditstore.ErrInvalid
		}
		result.Source.MediaType = result.Retained.MediaType
		result.Source.SizeBytes = result.Retained.SizeBytes
		if result.Source.Validate() != nil || result.Retained.Validate() != nil || result.Source.Digest != result.Retained.Digest {
			return auditstore.DependencyResultSource{}, auditstore.ErrInvalid
		}
		if len(result.Evidence) > auditdomain.MaximumRetainedDependencies {
			return auditstore.DependencyResultSource{}, auditstore.ErrDependencyClosureLimit
		}
	}
	if err := rows.Err(); err != nil {
		return auditstore.DependencyResultSource{}, err
	}
	if count == 0 {
		return result, auditstore.ErrNotFound
	}
	return result, nil
}
