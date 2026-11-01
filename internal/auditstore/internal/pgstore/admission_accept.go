package pgstore

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// acceptPreparedRound publishes membership already source-verified and staged
// under this claim. The final statement changes two parent rows; it never reads
// or rebuilds the containing item collection.
func (s *ClaimStore) acceptPreparedRound(ctx context.Context, p auditstore.AcceptRoundParams) (auditstore.Round, bool, error) {
	if validateClaimIdentity(p.Claim) != nil || validateID("Round", p.RoundID) != nil || validateID("previous Round", p.PreviousRoundID) != nil || p.Generation == "" || p.PreparedRoot == nil || p.PreparedRoot.Validate() != nil || !p.PreparedRoot.HasMetadata() || p.PreparedRoot.Verify(p.Manifest) != nil || p.Manifest.MediaType != auditdomain.PreparedRoundRootMediaType || len(p.Items) != 0 || len(p.Retained) != 0 {
		return auditstore.Round{}, false, auditstore.ErrInvalid
	}
	var result auditstore.Round
	changed := false
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := auditClaims.Lock(ctx, tx, p.Claim.AuditID, p.Claim.HolderID, int64(p.Claim.Epoch)); err != nil {
			if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
				return auditstore.ErrClaimLost
			}
			return err
		}
		var state auditstore.AdmissionProgress
		var data []byte
		if err := tx.QueryRow(ctx, `SELECT admission_checkpoint FROM audit_rounds WHERE audit_id=$1 AND round_id=$2`, p.Claim.AuditID, p.PreviousRoundID).Scan(&data); err != nil {
			return err
		}
		if strictjson.Decode(data, &state) != nil || state.Generation != p.Generation || state.RoundID != p.RoundID || state.RoundOrdinal != p.RoundOrdinal || state.Root == nil || state.Root.Verify(*p.PreparedRoot) != nil || state.StagedCount != state.SelectedCount || state.StagedPages != state.PublishedPages {
			return auditstore.ErrInvalid
		}
		if _, err := encodeAdmissionProgress(state); err != nil {
			return err
		}
		round, err := scanRound(tx.QueryRow(ctx, `SELECT `+prefixedRoundColumns("round")+` FROM audit_rounds AS round WHERE audit_id=$1 AND round_id=$2 FOR UPDATE`, p.Claim.AuditID, p.RoundID))
		if err != nil {
			return err
		}
		if round.Manifest.Digest != p.PreparedRoot.Digest || !round.Manifest.Ref.SameExact(p.PreparedRoot.Ref) || round.ExpectedItemCount != state.SelectedCount {
			return auditstore.ErrConflict
		}
		if round.State != auditstore.RoundProposed {
			result = round
			return nil
		}
		ref, _ := json.Marshal(p.PreparedRoot.Ref)
		result, err = scanRound(tx.QueryRow(ctx, acceptPreparedRoundSQL, p.Claim.AuditID, p.PreviousRoundID, p.RoundID, p.ExpectedAuditRevision, p.Generation, string(ref), p.PreparedRoot.Digest, state.SelectedCount, state.StagedPages, state.PagesDigest, p.RoundOrdinal))
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrPrecondition
		}
		changed = err == nil
		return err
	})
	return result, changed, err
}

// ReadSelectedPreparedManifest hydrates only the immutable execution entries
// selected for a batch, after summing cheap stored byte and descriptor costs.
// The accepted root/generation identifies their already-verified page chain.
func (s *QueryStore) ReadSelectedPreparedManifest(ctx context.Context, auditID, roundID string, itemIDs []string, maxBytes, maxRelated int) ([]auditdomain.PreparedExecutionItem, error) {
	result := []auditdomain.PreparedExecutionItem{}
	if validateID("Audit", auditID) != nil || validateID("Round", roundID) != nil || len(itemIDs) < 1 || len(itemIDs) > auditstore.MaxReconcileRows || maxBytes < 1 || maxBytes > auditstore.MaxAdmissionMetadataBytes || maxRelated < 1 || maxRelated > auditstore.MaxAdmissionRelated {
		return result, auditstore.ErrInvalid
	}
	seen := map[string]bool{}
	for _, id := range itemIDs {
		if validateID("item", id) != nil || seen[id] {
			return result, auditstore.ErrInvalid
		}
		seen[id] = true
	}
	rows, err := s.db.Query(ctx, selectedPreparedHeadersSQL, auditID, roundID, itemIDs)
	if err != nil {
		return result, err
	}
	type header struct {
		id                      string
		ordinal, bytes, related int
	}
	var headers []header
	for rows.Next() {
		var h header
		if err := rows.Scan(&h.id, &h.ordinal, &h.bytes, &h.related); err != nil {
			rows.Close()
			return result, err
		}
		headers = append(headers, h)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return result, err
	}
	if len(headers) != len(itemIDs) {
		return result, auditstore.ErrInvalid
	}
	ids := []string{}
	usedBytes, usedRelated := 0, 0
	for _, h := range headers {
		if h.bytes > maxBytes-usedBytes || h.related > maxRelated-usedRelated {
			break
		}
		usedBytes += h.bytes
		usedRelated += h.related
		ids = append(ids, h.id)
	}
	if len(ids) == 0 {
		return result, auditstore.ErrAdmissionSliceFull
	}
	rows, err = s.db.Query(ctx, selectedPreparedItemsSQL, auditID, roundID, ids)
	if err != nil {
		return result, err
	}
	defer rows.Close()
	for rows.Next() {
		var data []byte
		var ordinal int
		if err := rows.Scan(&data, &ordinal); err != nil {
			return result, err
		}
		var item auditdomain.PreparedExecutionItem
		if strictjson.Decode(data, &item) != nil || auditdomain.ValidatePreparedExecutionItem(item) != nil {
			return result, auditstore.ErrInvalid
		}
		item.Ordinal = ordinal
		result = append(result, item)
	}
	if err := rows.Err(); err != nil {
		return result, err
	}
	if len(result) != len(ids) {
		return result, auditstore.ErrConflict
	}
	return result, nil
}
