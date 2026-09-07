package contracts

// DroppedSpanCounts counts local discard decisions, not confirmed remote loss.
type DroppedSpanCounts struct {
	QueueOverflow     uint64 `json:"queueOverflow"`
	EncodingFailed    uint64 `json:"encodingFailed"`
	NonRetryable      uint64 `json:"nonRetryable"`
	RetryExhausted    uint64 `json:"retryExhausted"`
	CollectorRejected uint64 `json:"collectorRejected"`
	DeadlineExceeded  uint64 `json:"deadlineExceeded"`
	Cancelled         uint64 `json:"cancelled"`
	Shutdown          uint64 `json:"shutdown"`
}
