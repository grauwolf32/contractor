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

const (
	evidenceStopReserve  = auditcontract.MaximumDocumentBytes / 8
	executionRecordLimit = 256
)

type recordedWriteKind int

const (
	evidenceWrite recordedWriteKind = iota
	judgmentWrite
)

// RetainEvidenceLinks commits ordinary exact Artifact links while the caller
// holds LockTaskInputs. Frozen Audit forks own bytes independently of a Run.
func (s *OwnerTxStore) RetainEvidenceLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink) error {
	return s.retainRecordedLinks(ctx, audit, links, insertEvidenceLinksSQL, evidenceWrite)
}

func (s *OwnerTxStore) RetainJudgmentLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink) error {
	return s.retainRecordedLinks(ctx, audit, links, insertJudgmentLinksSQL, judgmentWrite)
}

func (s *OwnerTxStore) retainRecordedLinks(ctx context.Context, audit auditstore.Audit, links []auditstore.ArtifactLink, statement string, kind recordedWriteKind) error {
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
	var reserve int64
	if kind == evidenceWrite {
		reserve = min(int64(evidenceStopReserve), audit.Limits.MaxEvidenceBytes/8)
	}
	if cost > audit.Limits.MaxEvidenceBytes-audit.RetainedEvidenceBytes-reserve {
		return auditstore.ErrEvidenceBudgetExhausted
	}
	_, err := s.tx.Exec(ctx, statement, audit.AuditID, encoded, cost)
	return err
}

// AdmitEvidenceCollection preserves conclusion capacity while admitting notes.
// The caller holds the Audit row and has not written any new selected bytes.
func (s *OwnerTxStore) AdmitEvidenceCollection(ctx context.Context, auditID, executionID string, newSources int, newBytes int64) error {
	return s.admitRecordedCollection(ctx, auditID, executionID, newSources, newBytes, evidenceWrite)
}

// AdmitJudgmentCollection can consume the reserved slots and bytes, within the
// same total execution budget. It never creates a separate quota.
func (s *OwnerTxStore) AdmitJudgmentCollection(ctx context.Context, auditID, executionID string, newBytes int64) error {
	return s.admitRecordedCollection(ctx, auditID, executionID, 0, newBytes, judgmentWrite)
}

func (s *OwnerTxStore) admitRecordedCollection(ctx context.Context, auditID, executionID string, newSources int, newBytes int64, kind recordedWriteKind) error {
	var receiptCount, sourceCount, memberCount int
	var retainedBytes int64
	prefix := "evidence/" + executionID + "/"
	judgments := "judgment/" + executionID + "/"
	if err := s.tx.QueryRow(ctx, evidenceCollectionSizeSQL, auditID, prefix, prefix+"~", judgments, judgments+"~", executionID).Scan(&receiptCount, &retainedBytes, &sourceCount, &memberCount); err != nil {
		return err
	}
	if memberCount < 1 || memberCount > auditcontract.MaxExecutionMembers {
		return auditstore.ErrInvalid
	}
	// Evidence leaves one scalar-result slot per assigned member.
	recordLimit, byteLimit := executionRecordLimit, int64(auditcontract.MaximumDocumentBytes)
	if kind == evidenceWrite {
		recordLimit -= memberCount
		byteLimit -= evidenceStopReserve
	}
	prospectiveRecords := receiptCount + sourceCount + newSources + 1
	prospectiveBytes := retainedBytes + newBytes
	if prospectiveRecords > recordLimit || prospectiveBytes > byteLimit {
		return auditstore.ErrEvidenceBudgetExhausted
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
