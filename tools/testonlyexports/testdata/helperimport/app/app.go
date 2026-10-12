// Package app is production code that imports a test helper package, which
// the allowlist rule forbids.
package app

import "example.com/helperimport/apptest"

// Run has a production caller in cmd/run.
func Run() string { return apptest.Name() }
