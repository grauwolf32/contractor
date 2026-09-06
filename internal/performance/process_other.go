//go:build !linux

package performance

func readOSProcess() (Process, Reason) { return Process{}, UnsupportedPlatform }
