package contracts

// Stage request and result bounds belong to separate wire contracts. Keep
// their identities separate even when their current numeric values coincide.
const (
	MaxStageRequestBytes       = 256 * 1024
	MaxStageResultBytes        = 256 * 1024
	MaxStageResultSummaryBytes = 64 * 1024
	MaxStageResultArtifacts    = 128
	// This bounds a WorkerResult payload at the Planner boundary; the complete
	// WorkerCompletion envelope has its own MaxWorkerCompletionBytes limit.
	MaxWorkerResultPayloadBytes = 256 * 1024
)
