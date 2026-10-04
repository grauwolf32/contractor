//go:build integration

package findingintake

import "time"

// collectionAttemptTimeout mirrors the Controller's default operation
// timeout, which bounds one collection attempt. Collection commits bounded
// transactions, so an attempt only has to commit one of them, with ample
// margin even under the race detector.
const collectionAttemptTimeout = 10 * time.Second
