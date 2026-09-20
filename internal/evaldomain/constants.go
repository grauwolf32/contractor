package evaldomain

import "time"

// Collection bounds apply to one member's retained observation. They are
// independent of execution budgets and never change Run/Audit token limits.
const (
	MaxReportSources       = 128
	MaxChartBins           = 20
	MaxProgressBuckets     = 200
	ProgressGap            = 5 * time.Second
	CollectionBatchSize    = 100
	MaxInventoryExecutions = 1024
	MaxMetricSnapshots     = 1024
	MaxEvidenceGaps        = 1024
	MaxCollectionBytes     = 16 << 20
)

const (
	RecordKindResult        = "result"
	RecordKindAssessment    = "assessment"
	ResultSchemaVersion     = "contractor.eval-result-input/v1"
	AssessmentSchemaVersion = "contractor.eval-assessment-input/v1"

	NativeCollectorActor  = "system:eval-collector"
	NativeCollectorSource = "native-collector"

	FreshnessPending = "pending"
	FreshnessCurrent = "current"
	FreshnessStale   = "stale"
)
