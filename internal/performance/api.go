package performance

import (
	"context"
	"crypto/rand"
	"errors"
	"math"
	"time"
)

var ErrHistoryStep = errors.New("unsupported performance history step")

type APIDiagnostics struct {
	SkippedSamples  uint64  `json:"skippedSamples"`
	RejectedSamples uint64  `json:"rejectedSamples"`
	SkippedMinutes  uint64  `json:"skippedMinutes"`
	DroppedMinutes  uint64  `json:"droppedMinutes"`
	PendingMinutes  int     `json:"pendingMinutes"`
	CollectorReason *Reason `json:"collectorReason,omitempty"`
	WriterReason    *Reason `json:"writerReason,omitempty"`
}

type SnapshotResponse struct {
	Enabled                     bool           `json:"enabled"`
	Generation                  string         `json:"generation"`
	ObservedAt                  time.Time      `json:"observedAt"`
	SampleIntervalSeconds       uint32         `json:"sampleIntervalSeconds"`
	DatabaseIntervalSeconds     uint32         `json:"databaseIntervalSeconds"`
	DatabaseSizeIntervalSeconds uint32         `json:"databaseSizeIntervalSeconds"`
	Current                     *Sample        `json:"current,omitempty"`
	Diagnostics                 APIDiagnostics `json:"diagnostics"`
}

type FineHistoryPoint struct {
	Kind string `json:"kind"`
	Sample
}

type AggregateHistoryPoint struct {
	Kind string `json:"kind"`
	HistoryPoint
}

type HistoryResponse struct {
	From   time.Time `json:"from"`
	To     time.Time `json:"to"`
	Step   string    `json:"step"`
	Points any       `json:"points"`
}

// ReadService is a read-only facade over the independently owned collectors.
// Snapshot never performs I/O; only durable History calls its repository.
type ReadService struct {
	enabled     bool
	generation  string
	collector   *Collector
	diagnostics *Diagnostics
	history     *HistoryRepository
	now         func() time.Time
}

func NewReadService(
	enabled bool,
	collector *Collector,
	diagnostics *Diagnostics,
	history *HistoryRepository,
	now func() time.Time,
) *ReadService {
	if now == nil {
		now = time.Now
	}
	generation := rand.Text()
	if collector != nil {
		generation = collector.Snapshot().Generation
	}
	return &ReadService{
		enabled: enabled, generation: generation, collector: collector,
		diagnostics: diagnostics, history: history, now: now,
	}
}

func (s *ReadService) Snapshot() SnapshotResponse {
	result := SnapshotResponse{
		Enabled: s.enabled, Generation: s.generation, ObservedAt: s.now().UTC(),
		SampleIntervalSeconds:       uint32(SampleInterval / time.Second),
		DatabaseIntervalSeconds:     uint32(DatabaseInterval / time.Second),
		DatabaseSizeIntervalSeconds: uint32(DatabaseSizeInterval / time.Second),
	}
	if s.collector != nil {
		view := s.collector.Snapshot()
		result.Generation = view.Generation
		result.Current = view.Current
		result.Diagnostics.SkippedSamples = view.SkippedSamples
		result.Diagnostics.RejectedSamples = view.RejectedSamples
		if view.RejectedSamples != 0 {
			reason := RecordLimit
			result.Diagnostics.CollectorReason = &reason
		}
	}
	if s.diagnostics != nil {
		view := s.diagnostics.Snapshot()
		result.Diagnostics.SkippedMinutes = view.SkippedMinutes
		result.Diagnostics.DroppedMinutes = view.DroppedMinutes
		result.Diagnostics.PendingMinutes = view.PendingMinutes
		result.Diagnostics.WriterReason = view.WriteReason
	}
	if !s.enabled {
		result.Current = nil
	}
	return result
}

func (s *ReadService) History(
	ctx context.Context,
	from time.Time,
	to time.Time,
	step string,
) (HistoryResponse, error) {
	from, to = from.UTC(), to.UTC()
	result := HistoryResponse{From: from, To: to, Step: step}
	if from.IsZero() || !to.After(from) {
		return HistoryResponse{}, ErrHistoryRange
	}
	switch step {
	case "15s":
		now := s.now()
		// The requested window may start a few milliseconds before the
		// Server's moving cutoff while the request is in flight. Bound the
		// duration and require overlap with the retained hour instead of
		// making an exact one-hour browser range fail due to transit time.
		if to.Sub(from) > time.Hour || !to.After(now.Add(-time.Hour)) || !from.Before(now) ||
			math.Ceil(to.Sub(from).Seconds()/SampleInterval.Seconds()) > MaxHistoryPoints {
			return HistoryResponse{}, ErrHistoryRange
		}
		points := make([]FineHistoryPoint, 0)
		if s.collector != nil {
			for _, sample := range s.collector.History(from, to) {
				points = append(points, FineHistoryPoint{Kind: "sample", Sample: sample})
			}
		}
		result.Points = points
		return result, nil
	case "1m", "5m", "1h":
		if s.history == nil {
			return HistoryResponse{}, errors.New("performance history repository is unavailable")
		}
		steps := map[string]time.Duration{"1m": time.Minute, "5m": 5 * time.Minute, "1h": time.Hour}
		rows, err := s.history.Read(ctx, from, to, steps[step])
		if err != nil {
			return HistoryResponse{}, err
		}
		points := make([]AggregateHistoryPoint, 0, len(rows))
		for _, row := range rows {
			points = append(points, AggregateHistoryPoint{Kind: "aggregate", HistoryPoint: row})
		}
		result.Points = points
		return result, nil
	default:
		return HistoryResponse{}, ErrHistoryStep
	}
}
