package reporting

import "github.com/grauwolf32/contractor/internal/contracts"

// WorkerModelUsage contains known usage sums. A nonzero unavailable count
// makes complete token usage unavailable, including while a call is in flight.
type WorkerModelUsage struct {
	ModelCalls            uint64  `json:"modelCalls" strictjson:"required"`
	InputTokens           uint64  `json:"inputTokens" strictjson:"required"`
	OutputTokens          uint64  `json:"outputTokens" strictjson:"required"`
	TotalTokens           uint64  `json:"totalTokens" strictjson:"required"`
	TokenUsageUnavailable uint64  `json:"tokenUsageUnavailable" strictjson:"required"`
	DurationMS            *uint64 `json:"durationMs,omitempty"`
}

type WorkerModelPhases struct {
	Main          WorkerModelUsage `json:"main" strictjson:"required"`
	Compaction    WorkerModelUsage `json:"compaction" strictjson:"required"`
	Finalization  WorkerModelUsage `json:"finalization" strictjson:"required"`
	Summarization WorkerModelUsage `json:"summarization" strictjson:"required"`
}

type CompactionPromptMeasurement struct {
	Generation   uint64  `json:"generation" strictjson:"required"`
	PromptTokens *uint64 `json:"promptTokens,omitempty"`
}

// WorkerCompactionProgress describes one logical invocation's latest attempt,
// independently of terminal summarization and the enclosing Run outcome.
type WorkerCompactionProgress struct {
	InvocationID       string                       `json:"invocationId" strictjson:"required"`
	Generation         uint64                       `json:"generation" strictjson:"required"`
	Phase              string                       `json:"phase" strictjson:"required"`
	Attempts           uint64                       `json:"attempts" strictjson:"required"`
	Failures           uint64                       `json:"failures" strictjson:"required"`
	ArchiveReads       uint64                       `json:"archiveReads" strictjson:"required"`
	InputBytes         uint64                       `json:"inputBytes" strictjson:"required"`
	SummaryBytes       uint64                       `json:"summaryBytes" strictjson:"required"`
	ReasoningAvailable uint64                       `json:"reasoningAvailable" strictjson:"required"`
	ReasoningIncluded  uint64                       `json:"reasoningIncluded" strictjson:"required"`
	ReasoningTruncated uint64                       `json:"reasoningTruncated" strictjson:"required"`
	LastCause          string                       `json:"lastCause,omitempty"`
	BeforePrompt       *CompactionPromptMeasurement `json:"beforePrompt,omitempty"`
	AfterPrompt        *CompactionPromptMeasurement `json:"afterPrompt,omitempty"`
}

func (u WorkerModelUsage) validate() error {
	for _, n := range []uint64{u.ModelCalls, u.InputTokens, u.OutputTokens, u.TotalTokens, u.TokenUsageUnavailable} {
		if n > contracts.MaxExactInteger {
			return contracts.Invalidf("model phase counter exceeds its bound")
		}
	}
	if u.DurationMS != nil && *u.DurationMS > contracts.MaxExactInteger {
		return contracts.Invalidf("model phase duration exceeds its bound")
	}
	if u.TokenUsageUnavailable > u.ModelCalls {
		return contracts.Invalidf("model phase missing usage exceeds calls")
	}
	return nil
}

func (p WorkerModelPhases) Validate() error {
	for _, u := range []WorkerModelUsage{p.Main, p.Compaction, p.Finalization, p.Summarization} {
		if err := u.validate(); err != nil {
			return err
		}
	}
	return nil
}

func (p WorkerCompactionProgress) Validate() error {
	if contracts.ValidateWorkerSubtaskID(p.InvocationID) != nil || p.Generation > contracts.MaxWorkerCompactionsPerInvocation || p.Attempts > contracts.MaxWorkerCompactionsPerInvocation+1 || p.Failures > contracts.MaxWorkerCompactionsPerInvocation+1 {
		return contracts.Invalidf("compaction invocation or generation is invalid")
	}
	pending := uint64(0)
	switch p.Phase {
	case "requested", "running":
		pending = 1
	case "idle":
		if p.Attempts != 0 {
			return contracts.Invalidf("idle compaction has attempts")
		}
	case "applied", "failed":
	default:
		return contracts.Invalidf("unknown compaction phase")
	}
	if p.Attempts != p.Generation+p.Failures+pending {
		return contracts.Invalidf("compaction outcome counts are inconsistent")
	}
	if (p.Phase == "failed") != (p.LastCause != "") || p.LastCause != "" && !contracts.ValidWorkerFailureCode(p.LastCause) || p.Phase == "failed" && p.Failures == 0 {
		return contracts.Invalidf("compaction failure cause is inconsistent")
	}
	for _, n := range []uint64{p.ArchiveReads, p.InputBytes, p.SummaryBytes, p.ReasoningAvailable, p.ReasoningIncluded, p.ReasoningTruncated} {
		if n > contracts.MaxExactInteger {
			return contracts.Invalidf("compaction counter exceeds its bound")
		}
	}
	if p.ReasoningIncluded > p.ReasoningAvailable || p.ReasoningTruncated > p.ReasoningAvailable {
		return contracts.Invalidf("compaction reasoning counts are inconsistent")
	}
	for _, m := range []*CompactionPromptMeasurement{p.BeforePrompt, p.AfterPrompt} {
		if m != nil && (m.Generation > contracts.MaxWorkerCompactionsPerInvocation || m.PromptTokens != nil && *m.PromptTokens > contracts.MaxExactInteger) {
			return contracts.Invalidf("invalid compaction prompt measurement")
		}
	}
	if p.Phase != "idle" && p.BeforePrompt == nil {
		return contracts.Invalidf("compaction requires a before measurement")
	}
	if p.Phase == "applied" {
		if p.Generation == 0 || p.InputBytes == 0 || p.SummaryBytes == 0 || p.AfterPrompt == nil || p.BeforePrompt.Generation != p.Generation-1 || p.AfterPrompt.Generation != p.Generation {
			return contracts.Invalidf("applied compaction measurements are inconsistent")
		}
	} else if p.AfterPrompt != nil || p.BeforePrompt != nil && p.BeforePrompt.Generation != p.Generation {
		return contracts.Invalidf("unapplied compaction measurements are inconsistent")
	}
	return nil
}

func (p WorkerCompactionProgress) Clone() WorkerCompactionProgress {
	copyMeasurement := func(source *CompactionPromptMeasurement) *CompactionPromptMeasurement {
		if source == nil {
			return nil
		}
		value := *source
		if value.PromptTokens != nil {
			tokens := *value.PromptTokens
			value.PromptTokens = &tokens
		}
		return &value
	}
	p.BeforePrompt = copyMeasurement(p.BeforePrompt)
	p.AfterPrompt = copyMeasurement(p.AfterPrompt)
	return p
}

func (p WorkerModelPhases) Clone() WorkerModelPhases {
	for _, usage := range []*WorkerModelUsage{&p.Main, &p.Compaction, &p.Finalization, &p.Summarization} {
		if usage.DurationMS != nil {
			value := *usage.DurationMS
			usage.DurationMS = &value
		}
	}
	return p
}
