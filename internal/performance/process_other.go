//go:build !linux && !darwin && !windows && !freebsd && !openbsd && !solaris

package performance

func readOSProcess() (Process, Reason) { return Process{}, UnsupportedPlatform }
