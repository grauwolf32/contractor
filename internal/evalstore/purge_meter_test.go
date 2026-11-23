package evalstore

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"testing"
	"time"
)

type meterRecord struct {
	Level, Msg              string
	ExperimentID, ProjectID string
	Passes, Rows, Bytes     int64
	Age                     time.Duration
}

func meterRecords(t *testing.T, logs *bytes.Buffer) []meterRecord {
	t.Helper()
	var out []meterRecord
	decoder := json.NewDecoder(logs)
	for decoder.More() {
		var record struct {
			Level        string `json:"level"`
			Msg          string `json:"msg"`
			ExperimentID string `json:"experiment_id"`
			ProjectID    string `json:"project_id"`
			Passes       int64  `json:"passes"`
			Rows         int64  `json:"rows"`
			Bytes        int64  `json:"bytes"`
			Age          int64  `json:"age"`
		}
		if err := decoder.Decode(&record); err != nil {
			t.Fatal(err)
		}
		out = append(out, meterRecord{record.Level, record.Msg, record.ExperimentID, record.ProjectID,
			record.Passes, record.Rows, record.Bytes, time.Duration(record.Age)})
	}
	return out
}

// The purge meter logs the first pass this Server commits, every
// purgeLogEvery-th pass, a warning once the purge is older than purgeSlowAge,
// and the pass that finishes the purge, with the pass count, rows, bytes and
// age; passes that removed nothing are not counted, and a purge without a
// pass for purgeMeterIdle is forgotten (EVM-R243).
func TestPurgeMeterLogsPassCountAndAge(t *testing.T) {
	var logs bytes.Buffer
	requested := time.Date(2026, 10, 24, 12, 0, 0, 0, time.UTC)
	now := requested
	meter := NewPurgeMeter(slog.New(slog.NewJSONHandler(&logs, nil)), func() time.Time { return now })
	pass := PurgeProgress{Rows: 1000, Bytes: 4096}

	meter.Experiment("small", requested, PurgeProgress{Rows: 10, Bytes: 100, Done: true})
	meter.Experiment("waiting", requested, PurgeProgress{})
	meter.Project("empty", requested, PurgeProgress{Done: true})
	for range purgeLogEvery {
		now = now.Add(time.Second)
		meter.Experiment("large", requested, pass)
	}
	meter.Experiment("large", requested, PurgeProgress{})
	// Slower passes, none more than purgeMeterIdle apart, reach purgeSlowAge
	// at the next progress record.
	step := (purgeSlowAge - purgeLogEvery*time.Second) / purgeLogEvery
	for range purgeLogEvery {
		now = now.Add(step)
		meter.Experiment("large", requested, pass)
	}
	meter.Experiment("large", requested, PurgeProgress{Rows: 1, Bytes: 1, Done: true})
	meter.Project("project", requested, pass)
	meter.Project("project", requested, PurgeProgress{Done: true})
	meter.Project("stale", requested, pass)
	now = now.Add(purgeMeterIdle + time.Second)
	meter.Project("project", requested, pass)
	meter.Project("stale", requested, pass)

	want := []meterRecord{
		{"INFO", "Eval purge finished", "small", "", 1, 10, 100, 0},
		{"INFO", "Eval purge started", "large", "", 1, 1000, 4096, time.Second},
		{"INFO", "Eval purge progress", "large", "", purgeLogEvery, purgeLogEvery * 1000, purgeLogEvery * 4096, purgeLogEvery * time.Second},
		{"WARN", "Eval purge is slow", "large", "", 2 * purgeLogEvery, 2 * purgeLogEvery * 1000, 2 * purgeLogEvery * 4096, purgeSlowAge},
		{"INFO", "Eval purge finished", "large", "", 2*purgeLogEvery + 1, 2*purgeLogEvery*1000 + 1, 2*purgeLogEvery*4096 + 1, purgeSlowAge},
		{"INFO", "Eval purge started", "", "project", 1, 1000, 4096, purgeSlowAge},
		{"INFO", "Eval purge finished", "", "project", 1, 1000, 4096, purgeSlowAge},
		{"INFO", "Eval purge started", "", "stale", 1, 1000, 4096, purgeSlowAge},
		{"INFO", "Eval purge started", "", "project", 1, 1000, 4096, purgeSlowAge + purgeMeterIdle + time.Second},
		{"INFO", "Eval purge started", "", "stale", 1, 1000, 4096, purgeSlowAge + purgeMeterIdle + time.Second},
	}
	got := meterRecords(t, &logs)
	if len(got) != len(want) {
		t.Fatalf("records = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("record %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	if len(meter.purges) != 2 {
		t.Fatalf("meter holds %d purges, want the two with a recent pass", len(meter.purges))
	}
}
