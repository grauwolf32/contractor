package evalstore

import (
	"log/slog"
	"sync"
	"time"
)

const (
	// purgeLogEvery is how many passes of one purge lie between two progress
	// records: about ten minutes of an experiment purge, at one pass per Eval
	// coordinator tick.
	purgeLogEvery = 600
	// purgeSlowAge is the purge age from which a progress record is a
	// warning: the reopen condition of the accepted purge rate (EVM-R243).
	purgeSlowAge = time.Hour
	// purgeMeterIdle is how long a purge stays counted without a pass. A
	// purge this Server stops seeing was finished by the Project purge, or
	// is continued by another Server, or waits for drain.
	purgeMeterIdle = 10 * time.Minute
)

// PurgeMeter is the detection signal of the Eval purge rate that EVM-R243
// accepts. For each experiment and Project purge it counts the passes this
// Server commits, the rows and bytes they charged, and logs them with the
// purge's age since its deletion request: when this Server commits its first
// pass, every purgeLogEvery passes, as a warning once the purge is older than
// purgeSlowAge, and when the purge finishes. Counts are per process and
// start again after a restart; a purge without a pass for purgeMeterIdle is
// forgotten, so the meter holds at most the purges of that window.
type PurgeMeter struct {
	logger *slog.Logger
	now    func() time.Time
	mu     sync.Mutex
	purges map[purgeKey]*purgeTally
}

type purgeKey struct{ kind, id string }

type purgeTally struct {
	passes, rows, bytes int64
	last                time.Time
}

// NewPurgeMeter returns a meter that logs to logger and reads the time from
// now.
func NewPurgeMeter(logger *slog.Logger, now func() time.Time) *PurgeMeter {
	return &PurgeMeter{logger: logger, now: now, purges: map[purgeKey]*purgeTally{}}
}

// Experiment records a committed pass of an experiment's purge.
func (m *PurgeMeter) Experiment(id string, requested time.Time, pass PurgeProgress) {
	m.record(purgeKey{"experiment", id}, requested, pass)
}

// Project records the committed Eval pass of a Project purge transaction.
func (m *PurgeMeter) Project(id string, requested time.Time, pass PurgeProgress) {
	m.record(purgeKey{"project", id}, requested, pass)
}

func (m *PurgeMeter) record(key purgeKey, requested time.Time, pass PurgeProgress) {
	if pass.Rows == 0 && !pass.Done {
		return
	}
	now := m.now()
	m.mu.Lock()
	defer m.mu.Unlock()
	for other, tally := range m.purges {
		if now.Sub(tally.last) > purgeMeterIdle {
			delete(m.purges, other)
		}
	}
	tally := m.purges[key]
	if tally == nil && pass.Rows == 0 {
		return // A pass that found nothing of a purge this Server has not seen.
	}
	if tally == nil {
		tally = &purgeTally{}
		m.purges[key] = tally
	}
	if pass.Rows > 0 {
		tally.passes++
		tally.rows += pass.Rows
		tally.bytes += pass.Bytes
		tally.last = now
	}
	age := now.Sub(requested)
	attrs := []any{key.kind + "_id", key.id, "passes", tally.passes, "rows", tally.rows, "bytes", tally.bytes, "age", age}
	switch {
	case pass.Done:
		delete(m.purges, key)
		m.logger.Info("Eval purge finished", attrs...)
	case tally.passes == 1:
		m.logger.Info("Eval purge started", attrs...)
	case tally.passes%purgeLogEvery == 0 && age >= purgeSlowAge:
		m.logger.Warn("Eval purge is slow", attrs...)
	case tally.passes%purgeLogEvery == 0:
		m.logger.Info("Eval purge progress", attrs...)
	}
}
