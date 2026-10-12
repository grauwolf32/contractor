package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
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
	ItemID               string                   `json:"item_id"`
	ReceiptID            string                   `json:"receipt_id"`
	ProposedCheckOrdinal int                      `json:"proposed_check_ordinal"`
	Proposal             auditstore.ExactArtifact `json:"proposal"`
}

func (s *OwnerStore) MaterializeRound(
	ctx context.Context,
	params auditstore.MaterializeRoundParams,
) (auditstore.Audit, bool, error) {
	if err := validateMaterialize(params); err != nil {
		return auditstore.Audit{}, false, err
	}
	if replay, found, err := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationStart, params.IdempotencyKey, params.RequestDigest,
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
			approvalKind = auditstore.ItemApprovalNone
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
	audit, err := scanAudit(persistencepostgres.SavepointQueryRow(ctx, s.db, materializeRoundSQL,
		params.OwnerID, params.AuditID, params.ExpectedRevision,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		[]byte(params.BaselineSnapshot), optionalDeadline(params.DeadlineAt), encodedItems,
		params.Manifest.Digest, params.IdempotencyKey, params.RequestDigest, response,
		encodedLinks, retainedBytes, nil,
	))
	if err == nil {
		return audit, true, nil
	}
	switch persistencepostgres.Classify(err) {
	case persistencepostgres.ClassDeleting:
		return auditstore.Audit{}, false, auditstore.ErrProjectDeleting
	case persistencepostgres.ClassDuplicate:
		return s.replayAudit(ctx, params.OwnerID, auditstore.MutationStart, params.IdempotencyKey, params.RequestDigest)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, fmt.Errorf("materialize Audit round: %w", err)
	}
	if replay, found, replayErr := s.lookupAuditReplay(
		ctx, params.OwnerID, auditstore.MutationStart, params.IdempotencyKey, params.RequestDigest,
	); replayErr != nil || found {
		return replay, false, replayErr
	}
	if _, getErr := s.get(ctx, params.OwnerID, params.AuditID); errors.Is(getErr, auditstore.ErrNotFound) {
		return auditstore.Audit{}, false, auditstore.ErrNotFound
	} else if getErr != nil {
		return auditstore.Audit{}, false, getErr
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
}

func (s *ClaimStore) AcceptNextRound(
	ctx context.Context,
	params auditstore.AcceptRoundParams,
) (auditstore.Round, bool, error) {
	if err := validateAcceptRound(params); err != nil {
		return auditstore.Round{}, false, err
	}
	if replay, found, err := s.lookupAcceptedRoundReplay(ctx, params); err != nil || found {
		return replay, false, err
	}
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return auditstore.Round{}, false, err
	}
	encodedManifestRef, _ := json.Marshal(params.Manifest.Ref)
	items := make([]materializedItemJSON, len(params.Items))
	sources := make([]proposalItemSourceJSON, 0, len(params.Items))
	for index, item := range params.Items {
		encodedTaskRef, _ := json.Marshal(item.Task.Ref)
		encodedOrigin, _ := json.Marshal(item.Origin)
		approvalKind := item.ApprovalKind
		if approvalKind == "" {
			approvalKind = auditstore.ItemApprovalNone
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
	round, err := scanRound(persistencepostgres.SavepointQueryRow(ctx, s.db, acceptNextRoundSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExpectedAuditRevision, params.PreviousRoundID,
		params.RoundID, params.RoundOrdinal, encodedManifestRef,
		params.Manifest.Digest, encodedItems, encodedSources, acceptanceDigest,
	))
	if err == nil {
		return round, true, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Round{}, false, auditstore.ErrProjectDeleting
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate || errors.Is(err, pgx.ErrNoRows) {
		if replay, found, replayErr := s.lookupAcceptedRoundReplay(ctx, params); replayErr != nil || found {
			return replay, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return auditstore.Round{}, false, liveErr
		} else if !live {
			return auditstore.Round{}, false, auditstore.ErrClaimLost
		}
		if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
			return auditstore.Round{}, false, auditstore.ErrConflict
		}
		return auditstore.Round{}, false, auditstore.ErrPrecondition
	}
	return auditstore.Round{}, false, fmt.Errorf("accept next Audit round: %w", err)
}

func (s *ClaimStore) lookupAcceptedRoundReplay(
	ctx context.Context, params auditstore.AcceptRoundParams,
) (auditstore.Round, bool, error) {
	acceptanceDigest, err := roundAcceptanceDigest(params)
	if err != nil {
		return auditstore.Round{}, false, err
	}
	round, err := s.getRound(ctx, params.Claim.AuditID, params.RoundID)
	if errors.Is(err, auditstore.ErrNotFound) {
		return auditstore.Round{}, false, nil
	}
	if err != nil {
		return auditstore.Round{}, false, err
	}
	if round.Ordinal != params.RoundOrdinal || round.ExpectedItemCount != len(params.Items) ||
		round.Manifest.Digest != params.Manifest.Digest || !round.Manifest.Ref.SameExact(params.Manifest.Ref) {
		return auditstore.Round{}, true, auditstore.ErrConflict
	}
	var storedDigest *string
	if err := s.db.QueryRow(
		ctx, acceptedRoundDigestSQL, params.Claim.AuditID, params.RoundID,
	).Scan(&storedDigest); err != nil {
		return auditstore.Round{}, true, err
	}
	if storedDigest == nil || *storedDigest != acceptanceDigest {
		return auditstore.Round{}, true, auditstore.ErrConflict
	}
	return round, true, nil
}

func roundAcceptanceDigest(params auditstore.AcceptRoundParams) (string, error) {
	encoded, err := json.Marshal(struct {
		Schema          string                        `json:"schema"`
		PreviousRoundID string                        `json:"previousRoundId"`
		RoundID         string                        `json:"roundId"`
		RoundOrdinal    int                           `json:"roundOrdinal"`
		Manifest        auditstore.ExactArtifact      `json:"manifest"`
		Items           []auditstore.MaterializedItem `json:"items"`
	}{
		Schema: "contractor.audit.round-acceptance.v1", PreviousRoundID: params.PreviousRoundID,
		RoundID: params.RoundID, RoundOrdinal: params.RoundOrdinal,
		Manifest: params.Manifest, Items: params.Items,
	})
	if err != nil {
		return "", fmt.Errorf("encode Audit Round acceptance identity: %w", err)
	}
	return contentdigest.Bytes(encoded), nil
}

func (s *ClaimStore) TransitionRound(
	ctx context.Context,
	params auditstore.RoundTransitionParams,
) (auditstore.Round, error) {
	if err := validateRoundTransition(params); err != nil {
		return auditstore.Round{}, err
	}
	round, err := scanRound(s.db.QueryRow(ctx, transitionRoundSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.RoundID, params.ExpectedRevision,
		string(params.ExpectedState), string(params.TargetState),
	))
	if err == nil {
		return round, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Round{}, auditstore.ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Round{}, fmt.Errorf("transition Audit round: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Round{}, liveErr
	} else if !live {
		return auditstore.Round{}, auditstore.ErrClaimLost
	}
	existing, getErr := s.getRound(ctx, params.Claim.AuditID, params.RoundID)
	if getErr != nil {
		return auditstore.Round{}, getErr
	}
	if existing.State == params.TargetState && existing.Revision == params.ExpectedRevision+1 {
		return existing, nil
	}
	return auditstore.Round{}, auditstore.ErrPrecondition
}

func optionalDeadline(value time.Time) *time.Time {
	if value.IsZero() {
		return nil
	}
	return &value
}
