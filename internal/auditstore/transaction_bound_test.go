package auditstore

import (
	"reflect"
	"testing"
)

// TestStartStagingLockRequiresTransaction pins PER-R18 for the Audit store:
// a step whose row lock must last until the caller commits exists only on
// the store bound to the caller's transaction, so a pool-backed store cannot
// even name it.
func TestStartStagingLockRequiresTransaction(t *testing.T) {
	for _, method := range []string{"LockForStartStaging", "LockRunDeletionAudits", "RenewItemReviewsForResume"} {
		if _, found := reflect.TypeOf((*PostgresStore)(nil)).MethodByName(method); found {
			t.Errorf("PostgresStore.%s is available outside a transaction", method)
		}
		if _, found := reflect.TypeOf((*PostgresTxStore)(nil)).MethodByName(method); !found {
			t.Errorf("PostgresTxStore.%s is missing", method)
		}
	}
}
