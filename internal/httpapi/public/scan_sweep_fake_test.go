package public

import (
	"context"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/scanservice"
	"github.com/grauwolf32/contractor/internal/scanstore"
	"time"
)

type fakeScanSweeps struct{}

func (*fakeScanSweeps) Create(context.Context, string, string, scanservice.Request) (scanstore.Sweep, bool, error) {
	return fakeSweep(), false, nil
}
func (*fakeScanSweeps) Get(context.Context, string, string) (scanstore.Sweep, error) {
	return fakeSweep(), nil
}
func (*fakeScanSweeps) List(context.Context, string, string, int) ([]scanstore.Sweep, error) {
	return []scanstore.Sweep{fakeSweep()}, nil
}
func (*fakeScanSweeps) Members(context.Context, string, string, int, int) ([]scanstore.Member, error) {
	return []scanstore.Member{{Ordinal: 0, ProjectID: "project-one", Target: "", State: "skipped", Reason: "target_missing", Settled: true}}, nil
}
func (*fakeScanSweeps) Cancel(context.Context, string, string) (scanstore.Sweep, error) {
	s := fakeSweep()
	s.State = "cancelled"
	return s, nil
}
func (*fakeScanSweeps) Purge(context.Context, string, string) error { return nil }
func fakeSweep() scanstore.Sweep {
	now := time.Date(2026, 10, 23, 0, 0, 0, 0, time.UTC)
	return scanstore.Sweep{SweepID: "sweep-one", Workflow: "nuclei-web@1", SnapshotDigest: contentdigest.Bytes([]byte("sweep")), State: "completed", Counts: scanstore.Counts{Total: 1, Skipped: 1}, CreatedAt: now, SettledAt: &now}
}
