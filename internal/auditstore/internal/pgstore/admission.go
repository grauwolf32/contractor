package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/strictjson"
	"github.com/jackc/pgx/v5"
)

type ClaimedAdmission struct{ db persistencepostgres.DBTX }

func NewClaimedAdmission(db persistencepostgres.DBTX) *ClaimedAdmission {
	return &ClaimedAdmission{db: db}
}

type admissionWork struct {
	primary, related, metadata int
	payload                    int64
}

func (w *admissionWork) add(primary, related, metadata int, payload int64) error {
	if primary < 0 || related < 0 || metadata < 0 || payload < 0 ||
		primary > auditstore.MaxAdmissionPrimary-w.primary || related > auditstore.MaxAdmissionRelated-w.related ||
		metadata > auditstore.MaxAdmissionMetadataBytes-w.metadata || payload > auditstore.MaxAdmissionPayloadBytes-w.payload {
		return fmt.Errorf("%w: admission slice work limit", auditstore.ErrAdmissionSliceFull)
	}
	w.primary += primary
	w.related += related
	w.metadata += metadata
	w.payload += payload
	return nil
}

// AdmissionSession exposes only the artifact and read ports of one claimed
// transaction. Neither an owner request nor a nested callback obtains controller
// commands or a second pool connection.
type AdmissionSession struct {
	State         auditstore.AdmissionProgress
	Artifacts     *artifacts.Service
	Reads         auditstore.AdmissionReader
	tx            pgx.Tx
	reader        *admissionReader
	validateRoute func(auditdomain.AcceptedPlan, *auditdomain.RetainedDependencyManifest, string) error
}

// WithCandidate lets a rejected candidate roll back speculative retained forks
// and task artifacts while its outer slice still commits a rejection decision.
func (s *AdmissionSession) WithCandidate(ctx context.Context, fn func(context.Context, *artifacts.Service, auditstore.AdmissionReader) (bool, error)) (bool, error) {
	if fn == nil {
		return false, auditstore.ErrInvalid
	}
	child, err := s.tx.Begin(ctx)
	if err != nil {
		return false, err
	}
	defer func() {
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
		_ = child.Rollback(cleanup)
	}()
	reader := *s.reader
	reader.QueryStore = NewQueryStore(child)
	reader.db = child
	keep, err := fn(ctx, artifacts.NewService(artifacts.NewPostgresRepository(child)), &reader)
	if err != nil || !keep {
		return false, err
	}
	if err := child.Commit(ctx); err != nil {
		return false, err
	}
	return true, nil
}

// WithStep anchors the durable intent before invoking any producer callback.
// Subsequent callbacks perform one bounded changing phase. All automatic
// transaction replays keep that same intent/deadline and exact natural keys.
func (s *ClaimedAdmission) WithStep(ctx context.Context, params auditstore.AdmissionStepParams,
	fn func(context.Context, *AdmissionSession) (auditstore.AdmissionMutation, error),
) (auditstore.AdmissionProgress, error) {
	var result auditstore.AdmissionProgress
	if validateClaimIdentity(params.Claim) != nil || validateID("previous Round", params.PreviousRoundID) != nil || fn == nil {
		return result, auditstore.ErrInvalid
	}
	now := params.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := auditClaims.Lock(ctx, tx, params.Claim.AuditID, params.Claim.HolderID, int64(params.Claim.Epoch)); err != nil {
			if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
				return auditstore.ErrClaimLost
			}
			return err
		}
		var owner, project, profile string
		var baseline, checkpoint []byte
		var ordinal int
		err := tx.QueryRow(ctx, admissionFrameSQL, params.Claim.AuditID, params.PreviousRoundID).Scan(&owner, &project, &profile, &baseline, &ordinal, &checkpoint)
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.ErrPrecondition
		}
		if err != nil {
			return err
		}
		var state auditstore.AdmissionProgress
		if checkpoint == nil {
			state = auditstore.AdmissionProgress{Schema: auditstore.AdmissionSchema, Revision: 1,
				AuditID: params.Claim.AuditID, PreviousRoundID: params.PreviousRoundID, RoundOrdinal: ordinal + 1,
				ProfileDigest: profile, BaselineDigest: auditdomain.DigestBytes(baseline),
				Budget: json.RawMessage(`{}`), DeadlineAt: now.Add(auditstore.AdmissionValidationTimeout),
				PagesDigest: auditdomain.DigestBytes(nil)}
			state.RoundID = auditdomain.DeterministicID("round", state.AuditID, fmt.Sprint(state.RoundOrdinal))
			if err := tx.QueryRow(ctx, `SELECT clock_timestamp()`).Scan(&state.SourceCutoffAt); err != nil {
				return err
			}
			state.Generation = auditdomain.DeterministicID("admission", state.AuditID, state.PreviousRoundID, state.SourceCutoffAt.Format(time.RFC3339Nano))
			if err := tx.QueryRow(ctx, admissionSourceUpperSQL, state.AuditID, state.SourceCutoffAt).Scan(&state.SourceUpper.CreatedAt, &state.SourceUpper.ReceiptID); err != nil {
				return err
			}
			state.SourceUpper.Ordinal = 512
			encoded, err := encodeAdmissionProgress(state)
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE audit_rounds SET admission_checkpoint=$3::text::json WHERE audit_id=$1 AND round_id=$2 AND admission_checkpoint IS NULL`, state.AuditID, state.PreviousRoundID, string(encoded)); err != nil {
				return err
			}
			state.Changed = true
			result = state
			return nil
		}
		if len(checkpoint) > auditstore.MaxAdmissionProgressBytes || strictjson.Decode(checkpoint, &state) != nil || state.Schema != auditstore.AdmissionSchema ||
			state.AuditID != params.Claim.AuditID || state.PreviousRoundID != params.PreviousRoundID || state.ProfileDigest != profile || state.BaselineDigest != auditdomain.DigestBytes(baseline) {
			return auditstore.ErrInvalid
		}
		if _, err := encodeAdmissionProgress(state); err != nil {
			return err
		}
		if !now.Before(state.DeadlineAt) {
			return auditstore.ErrAdmissionDeadline
		}
		reader := newAdmissionReader(tx, state, owner, project)
		session := &AdmissionSession{State: state, Artifacts: artifacts.NewService(artifacts.NewPostgresRepository(tx)), Reads: reader, tx: tx, reader: reader}
		mutation, err := fn(ctx, session)
		if err != nil {
			return err
		}
		result, err = session.commit(ctx, mutation)
		return err
	})
	return result, err
}

func encodeAdmissionProgress(state auditstore.AdmissionProgress) ([]byte, error) {
	if state.Schema != auditstore.AdmissionSchema || state.Revision == 0 || state.Generation == "" || state.DeadlineAt.IsZero() ||
		state.SelectedCount < 0 || state.SelectedCount > auditdomain.MaximumItems || state.StagedCount < 0 || state.StagedCount > state.SelectedCount ||
		state.PublishedPages < 0 || state.StagedPages < 0 || state.StagedPages > state.PublishedPages || validateJSONObject("admission budget", state.Budget, 4096) != nil || state.SourceCutoffAt.IsZero() || state.RoundOrdinal < 2 || state.RoundOrdinal > 33 || state.RoundOrdinal == 33 && state.SelectedCount > 0 ||
		state.StopCode != "" && state.StopCode != "round_budget_exhausted" && state.StopCode != "item_budget_exhausted" && state.StopCode != "evidence_budget_exhausted" && state.StopCode != "submission_budget_exhausted" ||
		(state.BlockedCode == "") != (state.BlockedReceiptID == "") || state.Cursor.Ordinal < 0 || state.Cursor.Ordinal > 512 {
		return nil, auditstore.ErrInvalid
	}
	if state.PublishedPages > state.SelectedCount || state.StagedPages > state.StagedCount ||
		(state.PublishedPages == 0) != (state.SelectedCount == 0) || (state.StagedPages == 0) != (state.StagedCount == 0) ||
		!contentdigest.Valid(state.PagesDigest) || state.PublishedPages == 0 && state.PagesDigest != auditdomain.DigestBytes(nil) ||
		state.Root == nil && (state.StagedPages != 0 || state.StagedCount != 0) {
		return nil, auditstore.ErrInvalid
	}
	if state.Root != nil && (!state.Exhausted || state.SelectedCount == 0 || state.Root.Validate() != nil || !state.Root.HasMetadata() || state.Root.MediaType != auditdomain.PreparedRoundRootMediaType || state.Root.SizeBytes < 1 || state.Root.SizeBytes > auditdomain.MaximumPreparedRoundRootBytes || state.Root.Ref.Namespace != auditdomain.ArtifactNamespace(state.AuditID)) {
		return nil, auditstore.ErrInvalid
	}
	if state.Root != nil && (state.StagedPages == state.PublishedPages) != (state.StagedCount == state.SelectedCount) {
		return nil, auditstore.ErrInvalid
	}
	encoded, err := json.Marshal(state)
	if err != nil || len(encoded) > auditstore.MaxAdmissionProgressBytes {
		return nil, auditstore.ErrInvalid
	}
	return encoded, nil
}
