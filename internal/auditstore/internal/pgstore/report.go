package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

func (s *ClaimStore) CommitReport(
	ctx context.Context, params auditstore.CommitReportParams,
) (auditstore.Audit, error) {
	if err := validateCommitReport(params); err != nil {
		return auditstore.Audit{}, err
	}
	payload := encodeReportLinks(params.Machine, params.Summary)
	audit, err := scanAudit(s.db.QueryRow(ctx, commitReportSQL+prefixedAuditColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.RoundID, params.ExpectedRoundRevision,
		params.RequestDigest, payload,
	))
	if err == nil {
		return audit, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, fmt.Errorf("commit Audit report: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Audit{}, liveErr
	} else if !live {
		return auditstore.Audit{}, auditstore.ErrClaimLost
	}
	return auditstore.Audit{}, auditstore.ErrPrecondition
}

// encodeReportLinks projects the two report links for jsonb_to_recordset.
func encodeReportLinks(machine, summary auditstore.ArtifactLink) json.RawMessage {
	links := []auditstore.ArtifactLink{machine, summary}
	encodedLinks := make([]artifactLinkJSON, len(links))
	for index, link := range links {
		ref, _ := json.Marshal(link.Artifact.Ref)
		encodedLinks[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
	}
	payload, _ := json.Marshal(encodedLinks)
	return payload
}
