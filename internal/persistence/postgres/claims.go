package postgres

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// A controller claim leases one subject of a controller, such as an Audit or
// an Eval experiment, to one controller instance (PER-R24). Each controller
// keeps one claim row per subject in its own table, with the shape every
// claim table shares: the subject, a growing epoch, and the holder with the
// start and end of its lease. The controller leases subjects through
// LeaseSQL, proves the lease in each write through LiveClaimSQL,
// LiveCondition or Lock, and gives it back with Release. Every lease advances
// the epoch, so a write of a holder whose lease expired or was taken over
// finds no live claim.

// The bounds of a controller lease.
const (
	minControllerLease = time.Second
	maxControllerLease = 5 * time.Minute
)

// ErrControllerClaimLost reports that the holder no longer holds the claim at
// its epoch: the lease expired, was released or was taken over.
var ErrControllerClaimLost = errors.New("controller claim lost")

// ControllerClaims is the claim table of one controller.
type ControllerClaims struct {
	table   string
	subject string
}

// NewControllerClaims names a controller's claim table and its subject
// column. Both are fixed SQL identifiers of the schema, never input.
func NewControllerClaims(table, subject string) ControllerClaims {
	return ControllerClaims{table: table, subject: subject}
}

// ControllerClaim is one lease of a subject.
type ControllerClaim struct {
	SubjectID string
	HolderID  string
	Epoch     int64
	ClaimedAt time.Time
	ExpiresAt time.Time
}

// ValidControllerLease reports whether lease is within the bounds of a
// controller lease.
func ValidControllerLease(lease time.Duration) bool {
	return lease >= minControllerLease && lease <= maxControllerLease
}

// LeaseSQL completes a statement whose WITH list ends with a CTE named
// candidates that selects the ctid, as claim_tid, of claim rows that meet
// AvailableCondition, locked FOR UPDATE SKIP LOCKED. It leases them to holder $1 for $2
// milliseconds, advancing each epoch, and returns the new claims in lease
// order for ScanControllerClaim.
func (c ControllerClaims) LeaseSQL() string {
	return `, leased AS (
    UPDATE ` + c.table + ` AS claim
       SET epoch = claim.epoch + 1,
           holder_id = $1,
           claimed_at = clock_timestamp(),
           expires_at = clock_timestamp() + $2::bigint * interval '1 millisecond'
      FROM candidates
     WHERE claim.ctid = candidates.claim_tid
    RETURNING claim.` + c.subject + ` AS subject_id, claim.holder_id, claim.epoch,
              claim.claimed_at, claim.expires_at
)
SELECT subject_id, holder_id, epoch, claimed_at, expires_at
  FROM leased
 ORDER BY claimed_at, subject_id`
}

// AvailableCondition is the condition under which the claim row named alias
// can be leased: it has no holder, or its lease has ended.
func (c ControllerClaims) AvailableCondition(alias string) string {
	return `(` + alias + `.holder_id IS NULL OR ` + alias + `.expires_at <= clock_timestamp())`
}

// LiveClaimSQL is a query, for a MATERIALIZED CTE, that locks the claim row
// of the subject bound to subject while the holder bound to holder holds it
// at the epoch bound to epoch, and yields no row otherwise.
func (c ControllerClaims) LiveClaimSQL(subject, holder, epoch string) string {
	return `SELECT claim.` + c.subject + `
      FROM ` + c.table + ` AS claim
     WHERE claim.` + c.subject + ` = ` + subject + `
       AND ` + c.LiveCondition("claim", holder, epoch) + `
     FOR UPDATE OF claim`
}

// LiveCondition is the condition under which the claim row named alias is
// held by the holder bound to holder at the epoch bound to epoch and its
// lease lasts, for a statement that joins the claim table itself.
func (c ControllerClaims) LiveCondition(alias, holder, epoch string) string {
	return c.HeldCondition(alias, holder, epoch) + ` AND ` + alias + `.expires_at > clock_timestamp()`
}

// HeldCondition is the condition under which the claim row named alias is
// held by the holder bound to holder at the epoch bound to epoch, whether or
// not its lease lasts. Only a step that ends the holder's work with its
// subject, such as a release or a purge, accepts an expired lease: no other
// holder can have leased the subject without advancing the epoch.
func (c ControllerClaims) HeldCondition(alias, holder, epoch string) string {
	return alias + `.holder_id = ` + holder + ` AND ` + alias + `.epoch = ` + epoch
}

// Lock locks the live claim row of subjectID held by holderID at epoch until
// the transaction of db ends, or reports ErrControllerClaimLost.
func (c ControllerClaims) Lock(ctx context.Context, db DBTX, subjectID, holderID string, epoch int64) error {
	var locked string
	err := db.QueryRow(ctx, c.LiveClaimSQL("$1", "$2", "$3"), subjectID, holderID, epoch).Scan(&locked)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrControllerClaimLost
	}
	if err != nil {
		return WrapError("lock controller claim", err)
	}
	return nil
}

// Live reports whether holderID still holds subjectID at epoch, without
// locking the claim.
func (c ControllerClaims) Live(ctx context.Context, db DBTX, subjectID, holderID string, epoch int64) (bool, error) {
	var live bool
	err := db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM `+c.table+` AS claim
     WHERE claim.`+c.subject+` = $1 AND `+c.LiveCondition("claim", "$2", "$3")+`
)`, subjectID, holderID, epoch).Scan(&live)
	if err != nil {
		return false, WrapError("check controller claim", err)
	}
	return live, nil
}

// Release gives the claim of subjectID back when holderID holds it at epoch,
// expired or not, or reports ErrControllerClaimLost. The epoch stays, so the
// next lease advances it.
func (c ControllerClaims) Release(ctx context.Context, db DBTX, subjectID, holderID string, epoch int64) error {
	tag, err := db.Exec(ctx, `
UPDATE `+c.table+` AS claim
   SET holder_id = NULL, claimed_at = NULL, expires_at = NULL
 WHERE claim.`+c.subject+` = $1 AND `+c.HeldCondition("claim", "$2", "$3"), subjectID, holderID, epoch)
	if err != nil {
		return WrapError("release controller claim", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrControllerClaimLost
	}
	return nil
}

// ScanControllerClaim scans one claim that LeaseSQL returns.
func ScanControllerClaim(row pgx.Row) (ControllerClaim, error) {
	var result ControllerClaim
	if err := row.Scan(&result.SubjectID, &result.HolderID, &result.Epoch, &result.ClaimedAt, &result.ExpiresAt); err != nil {
		return ControllerClaim{}, err
	}
	if result.Epoch < 1 || !result.ExpiresAt.After(result.ClaimedAt) {
		return ControllerClaim{}, errors.New("stored controller claim is invalid")
	}
	return result, nil
}
