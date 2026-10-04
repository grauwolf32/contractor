//go:build !race

package findingintake

import "time"

// The production collection attempt has a 10-second default deadline.
const largeDirectCollectionTestTimeout = 10 * time.Second
