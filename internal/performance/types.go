// Package performance collects bounded in-memory operational measurements.
// Construction and lifecycle are explicit; importing it starts no work.
package performance

import "time"

const (
	SampleInterval       = 15 * time.Second
	DatabaseInterval     = time.Minute
	DatabaseSizeInterval = 5 * time.Minute
	LiveFrames           = 240
	MaxRecordBytes       = 32 * 1024
	MaxLiveBytes         = 8 * 1024 * 1024
	MaxHistoryPoints     = 1000
	HistoryRetention     = 7 * 24 * time.Hour
	MaxPendingMinutes    = 10
	MaxCleanupRows       = 1000
)

type Status string

const (
	OK          Status = "ok"
	Partial     Status = "partial"
	Unavailable Status = "unavailable"
)

// Reason is an allowlisted diagnostic, never a raw OS/driver error.
type Reason string

const (
	UnsupportedPlatform Reason = "unsupported_platform"
	ReadFailed          Reason = "read_failed"
	SamplingGap         Reason = "sampling_gap"
	CounterReset        Reason = "counter_reset"
	MissingBaseline     Reason = "missing_baseline"
	PermissionDenied    Reason = "permission_denied"
	StatisticsDisabled  Reason = "statistics_disabled"
	DatabaseUnavailable Reason = "database_unavailable"
	BudgetExceeded      Reason = "budget_exceeded"
	RecordLimit         Reason = "record_limit"
)

type Surface string

const (
	Public  Surface = "public"
	Private Surface = "private"
)

// HTTP counts have exactly 2 surfaces x 10 methods x 6 status classes. These
// dimensions intentionally cannot carry a URL, tenant, Run ID, or exception.
func HTTPMethods() [10]string {
	return [10]string{"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH", "other"}
}
func HTTPStatusClasses() [6]string {
	return [6]string{"1xx", "2xx", "3xx", "4xx", "5xx", "no_response"}
}
func HTTPBucketBoundsSeconds() [13]float64 {
	return [13]float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60}
}

// Histogram stores cumulative counts. Bucket 13 is +Inf, encoded as a count
// rather than a non-finite JSON number. Merge counts/sum/buckets, not quantiles.
type Histogram struct {
	Count      uint64     `json:"count"`
	SumSeconds float64    `json:"sumSeconds"`
	Buckets    [14]uint64 `json:"buckets"`
}

type Coverage struct {
	StartedAt       time.Time `json:"startedAt"`
	EndedAt         time.Time `json:"endedAt"`
	DurationSeconds float64   `json:"durationSeconds"`
	ExpectedSamples uint64    `json:"expectedSamples"`
	ObservedSamples uint64    `json:"observedSamples"`
}

type Freshness struct {
	Status          Status     `json:"status"`
	Reason          *Reason    `json:"reason,omitempty"`
	ObservedAt      *time.Time `json:"observedAt,omitempty"`
	LastAttemptAt   time.Time  `json:"lastAttemptAt"`
	IntervalSeconds uint32     `json:"intervalSeconds"`
	Coverage        Coverage   `json:"coverage"`
}

type HTTPSurface struct {
	Surface  Surface       `json:"surface"`
	InFlight uint64        `json:"inFlight"`
	Counts   [10][6]uint64 `json:"counts"`
	Duration Histogram     `json:"duration"`
}
type HTTP struct {
	Freshness Freshness      `json:"freshness"`
	Surfaces  [2]HTTPSurface `json:"surfaces"`
}

type Process struct {
	Freshness        Freshness         `json:"freshness"`
	CPUUserSeconds   *float64          `json:"cpuUserSeconds,omitempty"`
	CPUSystemSeconds *float64          `json:"cpuSystemSeconds,omitempty"`
	CPUCores         *float64          `json:"cpuCores,omitempty"`
	RSSBytes         *uint64           `json:"rssBytes,omitempty"`
	HeapLiveBytes    *uint64           `json:"heapLiveBytes,omitempty"`
	Goroutines       *uint64           `json:"goroutines,omitempty"`
	GCCycles         *uint64           `json:"gcCycles,omitempty"`
	GCPauseSeconds   *float64          `json:"gcPauseSeconds,omitempty"`
	GCPauses         *GCPauseHistogram `json:"gcPauses,omitempty"`
}

const MaxGCPauseBounds = 256

// Counts are non-cumulative bucket populations. Bounds are finite upper edges;
// the final count covers +Inf, which is never emitted as a JSON number.
type GCPauseHistogram struct {
	BoundsSeconds []float64 `json:"boundsSeconds"`
	Counts        []uint64  `json:"counts"`
}

type Pool struct {
	Freshness               Freshness `json:"freshness"`
	AcquiredConnections     *uint32   `json:"acquiredConnections,omitempty"`
	IdleConnections         *uint32   `json:"idleConnections,omitempty"`
	TotalConnections        *uint32   `json:"totalConnections,omitempty"`
	MaxConnections          *uint32   `json:"maxConnections,omitempty"`
	AcquireCount            *uint64   `json:"acquireCount,omitempty"`
	AcquireDurationSeconds  *float64  `json:"acquireDurationSeconds,omitempty"`
	EmptyAcquireCount       *uint64   `json:"emptyAcquireCount,omitempty"`
	EmptyAcquireWaitSeconds *float64  `json:"emptyAcquireWaitSeconds,omitempty"`
	CanceledAcquireCount    *uint64   `json:"canceledAcquireCount,omitempty"`
}

type Database struct {
	Freshness                     Freshness      `json:"freshness"`
	StatsReset                    *time.Time     `json:"statsReset,omitempty"`
	Commits                       *uint64        `json:"commits,omitempty"`
	Rollbacks                     *uint64        `json:"rollbacks,omitempty"`
	Deadlocks                     *uint64        `json:"deadlocks,omitempty"`
	TempFiles                     *uint64        `json:"tempFiles,omitempty"`
	TempBytes                     *uint64        `json:"tempBytes,omitempty"`
	BlocksRead                    *uint64        `json:"blocksRead,omitempty"`
	BlocksHit                     *uint64        `json:"blocksHit,omitempty"`
	ActiveConnections             *uint64        `json:"activeConnections,omitempty"`
	IdleConnections               *uint64        `json:"idleConnections,omitempty"`
	IdleInTransactionConnections  *uint64        `json:"idleInTransactionConnections,omitempty"`
	LockWaitingConnections        *uint64        `json:"lockWaitingConnections,omitempty"`
	LongestTransactionSeconds     *float64       `json:"longestTransactionSeconds,omitempty"`
	LongestIdleTransactionSeconds *float64       `json:"longestIdleTransactionSeconds,omitempty"`
	EstimatedLiveTuples           *uint64        `json:"estimatedLiveTuples,omitempty"`
	EstimatedDeadTuples           *uint64        `json:"estimatedDeadTuples,omitempty"`
	ClientConnections             *uint64        `json:"clientConnections,omitempty"`
	HiddenConnections             *uint64        `json:"hiddenConnections,omitempty"`
	AutovacuumWorkers             *uint64        `json:"autovacuumWorkers,omitempty"`
	VacuumCount                   *uint64        `json:"vacuumCount,omitempty"`
	AutovacuumCount               *uint64        `json:"autovacuumCount,omitempty"`
	LastVacuumAt                  *time.Time     `json:"lastVacuumAt,omitempty"`
	LastAutovacuumAt              *time.Time     `json:"lastAutovacuumAt,omitempty"`
	Rates                         *DatabaseRates `json:"rates,omitempty"`
}

// Rates cover real observation intervals; BufferHitRatio is PostgreSQL buffers,
// not operating-system cache, physical I/O or an application-only measurement.
type DatabaseRates struct {
	IntervalSeconds float64  `json:"intervalSeconds"`
	Commits         float64  `json:"commitsPerSecond"`
	Rollbacks       float64  `json:"rollbacksPerSecond"`
	Deadlocks       float64  `json:"deadlocksPerSecond"`
	TempFiles       float64  `json:"tempFilesPerSecond"`
	TempBytes       float64  `json:"tempBytesPerSecond"`
	BlocksRead      float64  `json:"blocksReadPerSecond"`
	BlocksHit       float64  `json:"blocksHitPerSecond"`
	BufferHitRatio  *float64 `json:"bufferHitRatio,omitempty"`
}

type DatabaseSize struct {
	Freshness Freshness `json:"freshness"`
	SizeBytes *uint64   `json:"sizeBytes,omitempty"`
}

type Sample struct {
	Version      int           `json:"version"`
	Generation   string        `json:"generation"`
	ObservedAt   time.Time     `json:"observedAt"`
	HTTP         *HTTP         `json:"http,omitempty"`
	Process      *Process      `json:"process,omitempty"`
	Pool         *Pool         `json:"pool,omitempty"`
	Database     *Database     `json:"database,omitempty"`
	DatabaseSize *DatabaseSize `json:"databaseSize,omitempty"`
}
