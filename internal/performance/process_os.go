//go:build linux || darwin || windows || freebsd || openbsd || solaris

package performance

import (
	"math"
	"os"

	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/process"
)

type processMetrics interface {
	Times() (*cpu.TimesStat, error)
	MemoryInfo() (*process.MemoryInfoStat, error)
}

func readOSProcess() (Process, Reason) {
	current, err := process.NewProcess(int32(os.Getpid()))
	if err != nil {
		return Process{}, ReadFailed
	}
	return readProcessMetrics(current)
}

// Read only this process, preserving independently available CPU and RSS.
// MemoryInfo.RSS is current resident memory; HWM is never an observation.
func readProcessMetrics(current processMetrics) (Process, Reason) {
	var value Process
	var reason Reason
	times, err := current.Times()
	if err != nil || times == nil {
		reason = ReadFailed
	} else {
		for _, field := range []struct {
			source float64
			target **float64
		}{{times.User, &value.CPUUserSeconds}, {times.System, &value.CPUSystemSeconds}} {
			if field.source < 0 || math.IsNaN(field.source) || math.IsInf(field.source, 0) {
				reason = ReadFailed
			} else {
				n := field.source
				*field.target = &n
			}
		}
	}
	memory, err := current.MemoryInfo()
	if err != nil || memory == nil || memory.RSS == 0 || memory.RSS > 1<<53-1 {
		// A live self process cannot have zero RSS. Some platform APIs return
		// zero-filled records on failure; never publish that as an observation.
		reason = ReadFailed
	} else {
		rss := memory.RSS
		value.RSSBytes = &rss
	}
	return value, reason
}
