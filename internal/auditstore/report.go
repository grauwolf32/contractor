package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

const (
	ReportMachineLogicalKey = "report/machine"
	ReportSummaryLogicalKey = "report/summary"
)

func (s *PostgresStore) CommitReport(
	ctx context.Context, params CommitReportParams,
) (Audit, error) {
	if err := validateCommitReport(params); err != nil {
		return Audit{}, err
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
		return Audit{}, fmt.Errorf("commit Audit report: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Audit{}, liveErr
	} else if !live {
		return Audit{}, ErrClaimLost
	}
	return Audit{}, ErrPrecondition
}

// encodeReportLinks projects the two report links for jsonb_to_recordset.
func encodeReportLinks(machine, summary ArtifactLink) json.RawMessage {
	links := []ArtifactLink{machine, summary}
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
