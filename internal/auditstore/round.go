package auditstore

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

type materializedItemJSON struct {
	ItemID          string                   `json:"item_id"`
	ItemKey         string                   `json:"item_key"`
	Ordinal         int                      `json:"ordinal"`
	Kind            string                   `json:"kind"`
	SubjectKey      string                   `json:"subject_key"`
	TaskRef         json.RawMessage          `json:"task_ref"`
	TaskDigest      string                   `json:"task_digest"`
	Origin          json.RawMessage          `json:"origin"`
	WorkflowRole    string                   `json:"workflow_role"`
	InitialState    string                   `json:"initial_state"`
	ApprovalKind    string                   `json:"approval_kind"`
	ApprovalDigest  string                   `json:"approval_digest,omitempty"`
	Status          string                   `json:"status"`
	Requested       []string                 `json:"requested"`
	Completed       []string                 `json:"completed"`
	Gaps            []string                 `json:"gaps"`
	Rationale       string                   `json:"rationale"`
	ProposalSources []proposalItemSourceJSON `json:"proposal_sources"`
}

type proposalItemSourceJSON struct {
	ItemID               string        `json:"item_id"`
	ReceiptID            string        `json:"receipt_id"`
	ProposedCheckOrdinal int           `json:"proposed_check_ordinal"`
	Proposal             ExactArtifact `json:"proposal"`
}

func (s *PostgresStore) MaterializeRound(
	ctx context.Context,
	params MaterializeRoundParams,
) (Audit, bool, error) {
	if err := validateMaterialize(params); err != nil {
		return Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	encodedManifestRef, _ := json.Marshal(params.Manifest.Ref)
	items := make([]materializedItemJSON, len(params.Items))
	for index, item := range params.Items {
		encodedTaskRef, _ := json.Marshal(item.Task.Ref)
		encodedOrigin, _ := json.Marshal(item.Origin)
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = ItemApprovalNone
		}
		items[index] = materializedItemJSON{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: encodedTaskRef,
			TaskDigest: item.Task.Digest, Origin: encodedOrigin, WorkflowRole: item.WorkflowRole,
			InitialState: string(item.InitialState), ApprovalKind: string(approvalKind),
			ApprovalDigest: item.ApprovalDigest, Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
			ProposalSources: []proposalItemSourceJSON{},
		}
	}
	encodedItems, _ := json.Marshal(items)
	links := make([]artifactLinkJSON, len(params.InitialRetained))
	for index, link := range params.InitialRetained {
		ref, _ := json.Marshal(link.Artifact.Ref)
		links[index] = artifactLinkJSON{
			LogicalKey: link.LogicalKey, ArtifactRef: ref,
			ArtifactDigest: link.Artifact.Digest, MediaType: link.Artifact.MediaType,
			SizeBytes: link.Artifact.SizeBytes, SourceProvenance: link.SourceProvenance,
			DisplayRef: link.DisplayRef,
		}
	}
	encodedLinks, _ := json.Marshal(links)
	retainedBytes, _ := validateArtifactLinks(params.InitialRetained)
	response, _ := json.Marshal(map[string]string{"auditId": params.AuditID, "roundId": params.RoundID})
	audit, err := scanAudit(s.db.QueryRow(ctx, materializeRoundSQL,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		[]byte(params.BaselineSnapshot), optionalDeadline(params.DeadlineAt), encodedItems,
		params.Manifest.Digest, params.IdempotencyKey, params.RequestDigest, response,
		encodedLinks, retainedBytes,
	))
	if err == nil {
		return audit, true, nil
	}
	switch persistencepostgres.SQLState(err) {
	case "55000":
		return Audit{}, false, ErrProjectDeleting
	case "23505":
		return s.replayAudit(ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("materialize Audit round: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, "audit.start", params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.Get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, ErrNotFound) {
		return Audit{}, false, ErrNotFound
	} else if getErr != nil {
		return Audit{}, false, getErr
	}
	return Audit{}, false, ErrPrecondition
}

func (s *PostgresStore) AcceptNextRound(
	ctx context.Context,
	params AcceptRoundParams,
) (Round, bool, error) {
	if err := validateAcceptRound(params); err != nil {
		return Round{}, false, err
	}
	if replay, found, err := s.lookupAcceptedRoundReplay(ctx, params); err != nil || found {
		return replay, false, err
	}
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return Round{}, false, err
	}
	encodedManifestRef, _ := json.Marshal(params.Manifest.Ref)
	items := make([]materializedItemJSON, len(params.Items))
	sources := make([]proposalItemSourceJSON, 0, len(params.Items))
	for index, item := range params.Items {
		encodedTaskRef, _ := json.Marshal(item.Task.Ref)
		encodedOrigin, _ := json.Marshal(item.Origin)
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = ItemApprovalNone
		}
		itemSources := make([]proposalItemSourceJSON, len(item.ProposalSources))
		for sourceIndex, source := range item.ProposalSources {
			itemSources[sourceIndex] = proposalItemSourceJSON{
				ItemID: item.ItemID, ReceiptID: source.ReceiptID,
				ProposedCheckOrdinal: source.ProposedCheckOrdinal, Proposal: source.Proposal,
			}
			sources = append(sources, itemSources[sourceIndex])
		}
		items[index] = materializedItemJSON{
			ItemID: item.ItemID, ItemKey: item.ItemKey, Ordinal: item.Ordinal,
			Kind: item.Kind, SubjectKey: item.SubjectKey, TaskRef: encodedTaskRef,
			TaskDigest: item.Task.Digest, Origin: encodedOrigin, WorkflowRole: item.WorkflowRole,
			InitialState: string(item.InitialState), ApprovalKind: string(approvalKind),
			ApprovalDigest: item.ApprovalDigest, Status: string(item.Coverage.Status),
			Requested: nonNilStrings(item.Coverage.Requested), Completed: nonNilStrings(item.Coverage.Completed),
			Gaps: nonNilStrings(item.Coverage.Gaps), Rationale: item.Coverage.Rationale,
			ProposalSources: itemSources,
		}
	}
	encodedItems, _ := json.Marshal(items)
	encodedSources, _ := json.Marshal(sources)
	round, err := scanRound(s.db.QueryRow(ctx, acceptNextRoundSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.PreviousRoundID,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		params.Manifest.Digest, encodedItems, encodedSources, acceptanceDigest,
	))
	if err == nil {
		return round, true, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Round{}, false, ErrProjectDeleting
	}
	if persistencepostgres.SQLState(err) == "23505" || errors.Is(err, pgx.ErrNoRows) {
		if replay, found, replayErr := s.lookupAcceptedRoundReplay(ctx, params); replayErr != nil || found {
			return replay, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return Round{}, false, liveErr
		} else if !live {
			return Round{}, false, ErrClaimLost
		}
		if persistencepostgres.SQLState(err) == "23505" {
			return Round{}, false, ErrConflict
		}
		return Round{}, false, ErrPrecondition
	}
	return Round{}, false, fmt.Errorf("accept next Audit round: %w", err)
}

func (s *PostgresStore) lookupAcceptedRoundReplay(
	ctx context.Context, params AcceptRoundParams,
) (Round, bool, error) {
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return Round{}, false, err
	}
	round, err := s.GetRound(ctx, params.Claim.AuditID, params.RoundID)
	if errors.Is(err, ErrNotFound) {
		return Round{}, false, nil
	}
	if err != nil {
		return Round{}, false, err
	}
	if round.Ordinal != params.RoundOrdinal || round.ExpectedItemCount != len(params.Items) ||
		round.Manifest.Digest != params.Manifest.Digest || !sameRoundArtifactRef(round.Manifest.Ref, params.Manifest.Ref) {
		return Round{}, true, ErrConflict
	}
	var storedDigest *string
	if err := s.db.QueryRow(
		ctx, acceptedRoundDigestSQL, params.Claim.AuditID, params.RoundID,
	).Scan(&storedDigest); err != nil {
		return Round{}, true, err
	}
	if storedDigest == nil || *storedDigest != acceptanceDigest {
		return Round{}, true, ErrConflict
	}
	return round, true, nil
}

func roundAcceptanceDigest(params AcceptRoundParams) (string, error) {
	encoded, err := json.Marshal(struct {
		Schema          string             `json:"schema"`
		PreviousRoundID string             `json:"previousRoundId"`
		RoundID         string             `json:"roundId"`
		RoundOrdinal    int                `json:"roundOrdinal"`
		Manifest        ExactArtifact      `json:"manifest"`
		Items           []MaterializedItem `json:"items"`
	}{
		Schema: "contractor.audit.round-acceptance.v1", PreviousRoundID: params.PreviousRoundID,
		RoundID: params.RoundID, RoundOrdinal: params.RoundOrdinal,
		Manifest: params.Manifest, Items: params.Items,
	})
	if err != nil {
		return "", fmt.Errorf("encode Audit Round acceptance identity: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func sameRoundArtifactRef(left, right contracts.ArtifactRef) bool {
	return left.Namespace == right.Namespace && left.Name == right.Name &&
		left.Revision != nil && right.Revision != nil && *left.Revision == *right.Revision
}

func (s *PostgresStore) TransitionRound(
	ctx context.Context,
	params RoundTransitionParams,
) (Round, error) {
	if err := validateRoundTransition(params); err != nil {
		return Round{}, err
	}
	round, err := scanRound(s.db.QueryRow(ctx, transitionRoundSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.RoundID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState),
	))
	if err == nil {
		return round, nil
	}
	if persistencepostgres.SQLState(err) == "55000" {
		return Round{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Round{}, fmt.Errorf("transition Audit round: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Round{}, liveErr
	} else if !live {
		return Round{}, ErrClaimLost
	}
	existing, getErr := s.GetRound(ctx, params.Claim.AuditID, params.RoundID)
	if getErr != nil {
		return Round{}, getErr
	}
	if existing.State == params.TargetState && existing.Revision == params.ExpectedRevision+1 {
		return existing, nil
	}
	return Round{}, ErrPrecondition
}

func optionalDeadline(value time.Time) *time.Time {
	if value.IsZero() {
		return nil
	}
	return &value
}
