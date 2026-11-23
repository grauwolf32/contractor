// Package disktest controls explicitly opted-in fixtures with large disk writes.
package disktest

import (
	"os"
	"testing"
)

// Require skips a disk-heavy fixture before it creates resources unless the caller
// explicitly sets CONTRACTOR_TEST_HEAVY_IO=1. Full gates use the same default.
func Require(t testing.TB) {
	t.Helper()
	switch os.Getenv("CONTRACTOR_TEST_HEAVY_IO") {
	case "1":
		return
	case "", "0":
		t.Skip("disk-heavy fixture disabled; explicitly set CONTRACTOR_TEST_HEAVY_IO=1 to run")
	default:
		t.Fatal("CONTRACTOR_TEST_HEAVY_IO must be 0 or 1")
	}
}
