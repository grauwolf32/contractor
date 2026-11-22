package pgstore

import (
	"context"
	"encoding/json"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"strings"
)

const evidenceStopReserve = auditcontract.MaximumDocumentBytes / 8

// RetainEvidenceLinks commits ordinary exact Artifact links while the caller
// holds LockTaskInputs. Frozen Audit forks own bytes independently of a Run.
func (s *OwnerTxStore) RetainEvidenceLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink) error {
	return s.retainRecordedLinks(ctx, audit, links, insertEvidenceLinksSQL)
}

func (s *OwnerTxStore) RetainJudgmentLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink) error {
	return s.retainRecordedLinks(ctx, audit, links, insertJudgmentLinksSQL)
}

func (s *OwnerTxStore) retainRecordedLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink, statement string) error {
	if _, err := validateArtifactLinks(links); err != nil {
		return err
	}
	encoded, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: links})
	var valid bool
	var cost int64
	if err := s.tx.QueryRow(ctx, readCollectionLinkStateSQL, audit.AuditID, encoded).Scan(&valid, &cost); err != nil {
		return err
	}
	if !valid {
		return auditstore.ErrConflict
	}
	reserve := min(int64(evidenceStopReserve), audit.Limits.MaxEvidenceBytes/8)
	if cost > audit.Limits.MaxEvidenceBytes-audit.RetainedEvidenceBytes-reserve {
		return auditstore.ErrEvidenceBudgetExhausted
	}
	_, err := s.tx.Exec(ctx, statement, audit.AuditID, encoded, cost)
	return err
}

// AdmitEvidenceCollection bounds one invocation using its ordered link prefix.
// The caller holds the Audit row and has not written any new selected bytes.
func (s *OwnerTxStore) AdmitEvidenceCollection(ctx context.Context, auditID, executionID string, newSources int, newBytes int64) error {
	var count int
	var used int64
	var sources int
	prefix := "evidence/" + executionID + "/"
	judgments := "judgment/" + executionID + "/"
	if err := s.tx.QueryRow(ctx, evidenceCollectionSizeSQL, auditID, prefix, prefix+"~", judgments, judgments+"~").Scan(&count, &used, &sources); err != nil {
		return err
	}
	if count+sources+newSources+1 > 256 || used+newBytes > auditcontract.MaximumDocumentBytes-evidenceStopReserve {
		return auditstore.ErrEvidenceBudgetExhausted
	}
	return nil
}

// ValidateEvidenceReplay requires an already linked receipt, preventing an
// unlinked preparatory binding from being mistaken for accepted evidence.
func (s *OwnerTxStore) ValidateEvidenceReplay(ctx context.Context, auditID, key string, ref []byte) error {
	var found bool
	if err := s.tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_artifact_links WHERE audit_id=$1 AND logical_key=$2 AND artifact_ref=$3::jsonb)`, auditID, key, json.RawMessage(ref)).Scan(&found); err != nil {
		return err
	}
	if !found {
		return auditstore.ErrInvalid
	}
	return nil
}

// ListExecutionEvidence returns only acknowledged annotations, bounded by the
// invocation collection limit. Source-only links confer no check membership.
func (s *QueryStore) ListExecutionEvidence(ctx context.Context, auditID, executionID string) ([]auditstore.ArtifactLink, error) {
	return s.listExecutionRecords(ctx, auditID, executionID, "evidence")
}

func (s *QueryStore) ListExecutionJudgments(ctx context.Context, auditID, executionID string) ([]auditstore.ArtifactLink, error) {
	return s.listExecutionRecords(ctx, auditID, executionID, "judgment")
}

func (s *QueryStore) listExecutionRecords(ctx context.Context, auditID, executionID, kind string) ([]auditstore.ArtifactLink, error) {
	if !contracts.ValidResourceID(auditID) || !contracts.ValidResourceID(executionID) {
		return nil, auditstore.ErrInvalid
	}
	prefix := kind + "/" + executionID + "/"
	rows, err := s.db.Query(ctx, listExecutionEvidenceSQL, auditID, prefix, prefix+"~")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []auditstore.ArtifactLink{}
	for rows.Next() {
		var link auditstore.ArtifactLink
		var ref []byte
		if err := rows.Scan(&link.Scope, &link.LogicalKey, &ref, &link.Artifact.Digest, &link.Artifact.MediaType, &link.Artifact.SizeBytes, &link.SourceProvenance); err != nil {
			return nil, err
		}
		if len(result) >= 256 || link.Scope != "audit" || !strings.HasPrefix(link.LogicalKey, prefix) || strictjson.Decode(ref, &link.Artifact.Ref) != nil || validateExactArtifact(kind, link.Artifact, true) != nil {
			return nil, auditstore.ErrInvalid
		}
		result = append(result, link)
	}
	return result, rows.Err()
}
