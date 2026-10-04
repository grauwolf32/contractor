//go:build race

package findingintake

import "time"

// Race instrumentation slows the 2,001-proposal database fixture. Preserve
// the production 10-second assertion in non-race runs and allow a 2x budget
// when checking correctness under -race.
const largeDirectCollectionTestTimeout = 20 * time.Second
