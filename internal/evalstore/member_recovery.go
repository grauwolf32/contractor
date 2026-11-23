package evalstore

import (
	"context"
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/jackc/pgx/v5"
)

// MemberRecovery is the recovery authority over one member inside one
// caller-owned transaction (EVM-R246). Recovery steps accept a deleting
// workspace, because accepted work must be reconciled, and every step needs
// the coordinator's current claim, also when it reconciles an external
// submission (EVM-R110, PER-R25).
//
// The first step of the transaction takes the authority: the evaluation
// Project row FOR SHARE, the experiment row FOR UPDATE and the live claim row
// FOR UPDATE, in that order; LockMember and RegisterExecutionProject lock the
// execution workspace's Project between the first and the second. A later
// RegisterExecutionProject step locks a workspace Project that its own
// transaction created, which no other transaction can lock yet. The
// transaction holds these row locks until it ends, so meanwhile no other
// transaction can fence or cancel the experiment, start the Project's
// deletion or lease the claim to another holder. A later step of the same
// transaction does not take them again. It proves in one statement that the
// claim is still live at its epoch, as every write must, and rereads the
// experiment's state and deletion request, which an earlier statement of the
// same transaction may have changed.
//
// A MemberRecovery is no authorization of its own: it belongs to the
// transaction of the store that made it, and once that transaction or
// savepoint has ended every step fails, because its pgx.Tx is closed. Its
// first step must not run inside a savepoint that is rolled back before a
// later step, which would release the locks the later step relies on; every
// caller takes it directly on its transaction.
type MemberRecovery struct {
	store      *Store
	scope      Scope
	id, member string
	claim      Claim
	held       bool
}

// RecoverMember returns the recovery authority over member of experiment id
// under claim, bound to this store's transaction. It reads nothing until its
// first step.
func (s *Store) RecoverMember(scope Scope, id, member string, claim Claim) *MemberRecovery {
	return &MemberRecovery{store: s, scope: scope, id: id, member: member, claim: claim}
}

// errRecoveryLockOrder reports a LockMember call after the transaction already
// locked the experiment, which would lock a Project after it.
var errRecoveryLockOrder = errors.New("evalstore member recovery must lock its Projects in its first step")

// recoveryFence is what a recovery step needs of its experiment's state.
type recoveryFence struct {
	state             evaldomain.State
	deletionRequested bool
}

// authority takes the recovery authority in the first step of the
// transaction and proves it again in a later step.
func (r *MemberRecovery) authority(ctx context.Context) (recoveryFence, error) {
	if !r.held {
		if _, err := r.store.project(ctx, r.scope, true); err != nil {
			return recoveryFence{}, err
		}
	}
	return r.afterProjectLocks(ctx)
}

// afterProjectLocks locks the experiment and proves the claim once every
// Project lock of the first step is held, or proves the authority again in a
// later step.
func (r *MemberRecovery) afterProjectLocks(ctx context.Context) (recoveryFence, error) {
	if r.held {
		return r.refresh(ctx)
	}
	e, err := r.store.locked(ctx, r.scope, r.id)
	if err != nil {
		return recoveryFence{}, err
	}
	if err = r.store.checkClaim(ctx, r.id, r.claim); err != nil {
		return recoveryFence{}, err
	}
	r.held = true
	return recoveryFence{state: e.State, deletionRequested: e.DeletionRequestedAt != nil}, nil
}

// refresh proves in a later step that the claim the first step locked is
// still live at its epoch and rereads the experiment's state and deletion
// request. Both rows stay locked by this transaction, so it takes no lock.
func (r *MemberRecovery) refresh(ctx context.Context) (recoveryFence, error) {
	var fence recoveryFence
	var deletion *time.Time
	err := r.store.db.QueryRow(ctx, `SELECT e.state, e.deletion_requested_at
FROM eval_experiments e
JOIN eval_controller_claims claim USING (experiment_id)
WHERE e.experiment_id = $1
    AND `+evalClaims.LiveCondition("claim", "$2", "$3"), r.id, r.claim.HolderID, r.claim.Epoch).Scan(&fence.state, &deletion)
	if errors.Is(err, pgx.ErrNoRows) {
		return recoveryFence{}, ErrClaimLost
	}
	if err != nil {
		return recoveryFence{}, err
	}
	fence.deletionRequested = deletion != nil
	return fence, nil
}
