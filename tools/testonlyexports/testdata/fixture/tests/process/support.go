// Package process stands for the process tests under tests/, whose
// non-test files are test code too.
package process

import "example.com/fixture/core"

// ProcessExport is called by nothing.
func ProcessExport() { core.UsedByProcessTests() }
