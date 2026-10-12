package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *ClaimStore) AcceptInitialRound(ctx context.Context, p auditstore.AcceptInitialRoundParams) (auditstore.Round, bool, error) {
	if err := validateClaimIdentity(p.Claim); err != nil {
		return auditstore.Round{}, false, err
	}
	if p.ExpectedAuditRevision == 0 || p.ExpectedAuditRevision > math.MaxInt64 || validateID("roundID", p.RoundID) != nil ||
		validateExactArtifact("initial manifest", p.Manifest, true) != nil ||
		p.Inventory.LogicalKey != auditstore.InitialInventoryLogicalKey || len(p.Items) == 0 {
		return auditstore.Round{}, false, invalidf("prepared initial Round is invalid")
	}
	if err := validateRoundItems(p.Items); err != nil {
		return auditstore.Round{}, false, err
	}
	for _, item := range p.Items {
		if len(item.ProposalSources) != 0 {
			return auditstore.Round{}, false, invalidf("initial inventory cannot consume proposals")
		}
	}
	retainedBytes, err := validateArtifactLinks([]auditstore.ArtifactLink{p.Inventory})
	if err != nil {
		return auditstore.Round{}, false, err
	}
	identity, err := json.Marshal(struct {
		Schema    string                        `json:"schema"`
		RoundID   string                        `json:"roundId"`
		Manifest  auditstore.ExactArtifact      `json:"manifest"`
		Items     []auditstore.MaterializedItem `json:"items"`
		Inventory auditstore.ArtifactLink       `json:"inventory"`
	}{"contractor.audit.initial-round-acceptance.v1", p.RoundID, p.Manifest, p.Items, p.Inventory})
	if err != nil {
		return auditstore.Round{}, false, err
	}
	digest := auditdomain.DigestBytes(identity)
	replay := func() (auditstore.Round, bool, error) {
		round, getErr := s.getRound(ctx, p.Claim.AuditID, p.RoundID)
		if errors.Is(getErr, auditstore.ErrNotFound) {
			return auditstore.Round{}, false, nil
		}
		if getErr != nil {
			return auditstore.Round{}, false, getErr
		}
		var stored *string
		if getErr = s.db.QueryRow(ctx, acceptedRoundDigestSQL, p.Claim.AuditID, p.RoundID).Scan(&stored); getErr != nil {
			return auditstore.Round{}, true, getErr
		}
		if stored == nil || *stored != digest {
			return auditstore.Round{}, true, auditstore.ErrConflict
		}
		return round, true, nil
	}
	if round, found, replayErr := replay(); found || replayErr != nil {
		return round, false, replayErr
	}
	// Identical row encoding keeps review, coverage and item authority exactly
	// the same as the direct-input start transaction.
	items := make([]materializedItemJSON, len(p.Items))
	for i, item := range p.Items {
		ref, _ := json.Marshal(item.Task.Ref)
		origin, _ := json.Marshal(item.Origin)
		approval := item.ApprovalKind
		if approval == "" {
			approval = auditstore.ItemApprovalNone
		}
		items[i] = materializedItemJSON{ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: ref, TaskDigest: item.Task.Digest,
			Origin: origin, WorkflowRole: item.WorkflowRole, InitialState: string(item.InitialState),
			ApprovalKind: string(approval), ApprovalDigest: item.ApprovalDigest,
			Status: string(item.Coverage.Status), Requested: nonNilStrings(item.Coverage.Requested),
			Completed: nonNilStrings(item.Coverage.Completed), Gaps: nonNilStrings(item.Coverage.Gaps),
			Rationale: item.Coverage.Rationale, ProposalSources: []proposalItemSourceJSON{}}
	}
	encodedItems, _ := json.Marshal(items)
	encodedRef, _ := json.Marshal(p.Manifest.Ref)
	links, _ := prepareRetainedLinks(auditstore.CollectParams{Retained: []auditstore.ArtifactLink{p.Inventory}})
	_, err = scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, acceptInitialRoundSQL,
		p.Claim.AuditID, p.Claim.HolderID, p.Claim.Epoch, p.ExpectedAuditRevision, p.RoundID, encodedRef,
		encodedItems, p.Manifest.Digest,
		links, retainedBytes, digest,
	))
	if err == nil {
		round, getErr := s.getRound(ctx, p.Claim.AuditID, p.RoundID)
		return round, true, getErr
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Round{}, false, auditstore.ErrProjectDeleting
	}
	if errors.Is(err, pgx.ErrNoRows) || persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
		if round, found, replayErr := replay(); found || replayErr != nil {
			return round, false, replayErr
		}
		if live, claimErr := s.claimLive(ctx, p.Claim); claimErr != nil {
			return auditstore.Round{}, false, claimErr
		} else if !live {
			return auditstore.Round{}, false, auditstore.ErrClaimLost
		}
		return auditstore.Round{}, false, auditstore.ErrPrecondition
	}
	return auditstore.Round{}, false, fmt.Errorf("accept prepared initial Audit Round: %w", err)
}
