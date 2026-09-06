//go:build linux

package performance

import (
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"
)

func readOSProcess() (Process, Reason) {
	var value Process
	var reason Reason
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) == nil {
		user := float64(usage.Utime.Sec) + float64(usage.Utime.Usec)/1e6
		system := float64(usage.Stime.Sec) + float64(usage.Stime.Usec)/1e6
		value.CPUUserSeconds, value.CPUSystemSeconds = &user, &system
	} else {
		reason = ReadFailed
	}
	file, err := os.Open("/proc/self/statm")
	if err != nil {
		return value, ReadFailed
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 257))
	if err != nil || len(data) > 256 {
		return value, ReadFailed
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return value, ReadFailed
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	pageSize := uint64(os.Getpagesize())
	if err != nil || pages > (1<<53-1)/pageSize {
		return value, ReadFailed
	}
	rss := pages * pageSize // current RSS, never process-lifetime ru_maxrss
	value.RSSBytes = &rss
	return value, reason
}
