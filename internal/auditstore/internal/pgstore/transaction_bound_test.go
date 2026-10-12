package pgstore

import (
	"reflect"
	"testing"
)

// TestStartStagingLockRequiresTransaction pins PER-R18 for the Audit store:
// a step whose row lock must last until the caller commits exists only on a
// seam's store bound to the caller's transaction, so a pool-backed store
// cannot even name it.
func TestStartStagingLockRequiresTransaction(t *testing.T) {
	for _, step := range []struct {
		pool, tx any
		method   string
	}{
		{(*OwnerStore)(nil), (*OwnerTxStore)(nil), "LockForStartStaging"},
		{(*RunDeletion)(nil), (*RunDeletionTx)(nil), "LockRunDeletionAudits"},
		{(*ReviewStore)(nil), (*ReviewTxStore)(nil), "RenewItemReviewsForResume"},
	} {
		if _, found := reflect.TypeOf(step.pool).MethodByName(step.method); found {
			t.Errorf("%T.%s is available outside a transaction", step.pool, step.method)
		}
		if _, found := reflect.TypeOf(step.tx).MethodByName(step.method); !found {
			t.Errorf("%T.%s is missing", step.tx, step.method)
		}
	}
}
