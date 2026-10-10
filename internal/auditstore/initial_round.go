package auditstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const InitialInventoryLogicalKey = "initial-inventory"

// AcceptInitialRoundParams contains only derived state. The original baseline,
// deadline and preparation receipts remain unchanged when it is accepted.
type AcceptInitialRoundParams struct {
	Claim                 ControllerClaim
	ExpectedAuditRevision uint64
	RoundID               string
	Manifest              ExactArtifact
	Items                 []MaterializedItem
	Inventory             ArtifactLink
}

func (s *PostgresStore) AcceptInitialRound(ctx context.Context, p AcceptInitialRoundParams) (Round, bool, error) {
	if err := validateClaimIdentity(p.Claim); err != nil {
		return Round{}, false, err
	}
	if p.ExpectedAuditRevision == 0 || p.ExpectedAuditRevision > math.MaxInt64 || validateID("roundID", p.RoundID) != nil ||
		validateExactArtifact("initial manifest", p.Manifest, true) != nil ||
		p.Inventory.LogicalKey != InitialInventoryLogicalKey || len(p.Items) == 0 {
		return Round{}, false, invalidf("prepared initial Round is invalid")
	}
	if err := validateRoundItems(p.Items); err != nil {
		return Round{}, false, err
	}
	for _, item := range p.Items {
		if len(item.ProposalSources) != 0 {
			return Round{}, false, invalidf("initial inventory cannot consume proposals")
		}
	}
	retainedBytes, err := validateArtifactLinks([]ArtifactLink{p.Inventory})
	if err != nil {
		return Round{}, false, err
	}
	identity, err := json.Marshal(struct {
		Schema    string             `json:"schema"`
		RoundID   string             `json:"roundId"`
		Manifest  ExactArtifact      `json:"manifest"`
		Items     []MaterializedItem `json:"items"`
		Inventory ArtifactLink       `json:"inventory"`
	}{"contractor.audit.initial-round-acceptance.v1", p.RoundID, p.Manifest, p.Items, p.Inventory})
	if err != nil {
		return Round{}, false, err
	}
	digest := auditdomain.DigestBytes(identity)
	replay := func() (Round, bool, error) {
		round, getErr := s.GetRound(ctx, p.Claim.AuditID, p.RoundID)
		if errors.Is(getErr, ErrNotFound) {
			return Round{}, false, nil
		}
		if getErr != nil {
			return Round{}, false, getErr
		}
		var stored *string
		if getErr = s.db.QueryRow(ctx, acceptedRoundDigestSQL, p.Claim.AuditID, p.RoundID).Scan(&stored); getErr != nil {
			return Round{}, true, getErr
		}
		if stored == nil || *stored != digest {
			return Round{}, true, ErrConflict
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
			approval = ItemApprovalNone
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
	links, _ := prepareRetainedLinks(CollectParams{Retained: []ArtifactLink{p.Inventory}})
	_, err = scanAudit(s.db.QueryRow(ctx, acceptInitialRoundSQL,
		p.Claim.AuditID, p.Claim.HolderID, p.Claim.Epoch, p.ExpectedAuditRevision, p.RoundID, encodedRef,
		encodedItems, p.Manifest.Digest,
		links, retainedBytes, digest,
	))
	if err == nil {
		round, getErr := s.GetRound(ctx, p.Claim.AuditID, p.RoundID)
		return round, true, getErr
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Round{}, false, ErrProjectDeleting
	}
	if errors.Is(err, pgx.ErrNoRows) || persistencepostgres.SQLState(err) == persistencepostgres.SQLStateUniqueViolation {
		if round, found, replayErr := replay(); found || replayErr != nil {
			return round, false, replayErr
		}
		if live, claimErr := s.claimLive(ctx, p.Claim); claimErr != nil {
			return Round{}, false, claimErr
		} else if !live {
			return Round{}, false, ErrClaimLost
		}
		return Round{}, false, ErrPrecondition
	}
	return Round{}, false, fmt.Errorf("accept prepared initial Audit Round: %w", err)
}
