// Package coretest is a test helper package: its exported helpers are not
// reported, and its references are test references.
package coretest

import "example.com/fixture/core"

// HelperExport is called by nothing.
func HelperExport() { core.UsedByHelper() }
