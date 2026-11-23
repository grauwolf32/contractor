package telemetry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const (
	DefaultRuntimeDiagnosticLimit = 5
	MaxRuntimeDiagnosticLimit     = 10
	MaxDiagnosticPageBytes        = 128 << 10
	maxRuntimeDiagnosticBytes     = 2 << 20
	// Divide the input budget across all selected rows, including lookahead.
	maxRuntimeParticipantBytes = maxRuntimeDiagnosticBytes / (MaxRuntimeDiagnosticLimit + 1)
	maxDiagnosticStopBytes     = 4096
)

// RuntimeDiagnosticParams selects one bounded, owner-authorized attempt page.
// Generation and After are opaque-cursor facts supplied by the HTTP boundary.
type RuntimeDiagnosticParams struct {
	OwnerID, RunID, StageExecutionID string
	Generation, After                string
	Limit                            int
}

// RuntimeAdapterDiagnostic uses decimal strings for exact uint64 counters in
// browser clients. It exposes neither adapter configuration nor credentials.
type RuntimeAdapterDiagnostic struct {
	Ref              string                       `json:"ref"`
	Operations       string                       `json:"operations"`
	FailedOperations string                       `json:"failedOperations"`
	FlushAttempted   *bool                        `json:"flushAttempted,omitempty"`
	FlushSucceeded   *bool                        `json:"flushSucceeded,omitempty"`
	LastErrorCode    *string                      `json:"lastErrorCode,omitempty"`
	DroppedSpans     *DroppedSpanDiagnosticCounts `json:"droppedSpans,omitempty"`
}

type DroppedSpanDiagnosticCounts struct {
	QueueOverflow     string `json:"queueOverflow"`
	EncodingFailed    string `json:"encodingFailed"`
	NonRetryable      string `json:"nonRetryable"`
	RetryExhausted    string `json:"retryExhausted"`
	CollectorRejected string `json:"collectorRejected"`
	DeadlineExceeded  string `json:"deadlineExceeded"`
	Cancelled         string `json:"cancelled"`
	Shutdown          string `json:"shutdown"`
}

type RuntimeDiagnostic struct {
	Compaction          *reporting.WorkerCompactionProgress `json:"compaction,omitempty"`
	ModelPhases         *reporting.WorkerModelPhases        `json:"modelPhases,omitempty"`
	LogicalAgent        string                              `json:"logicalAgent"`
	Status              string                              `json:"status"`
	Reason              string                              `json:"reason,omitempty"`
	Complete            *bool                               `json:"complete,omitempty"`
	DurationMS          *string                             `json:"durationMs,omitempty"`
	StopReason          *string                             `json:"stopReason,omitempty"`
	StopReasonTruncated bool                                `json:"stopReasonTruncated"`
	Adapters            []RuntimeAdapterDiagnostic          `json:"adapters"`
}

type RuntimeDiagnosticPage struct {
	Status     string              `json:"status"`
	Reason     string              `json:"reason,omitempty"`
	Items      []RuntimeDiagnostic `json:"items"`
	Generation string              `json:"-"`
	More       bool                `json:"-"`
	After      string              `json:"-"`
}

func (r *Repository) ReadRuntimeDiagnostics(ctx context.Context, params RuntimeDiagnosticParams) (RuntimeDiagnosticPage, error) {
	result := RuntimeDiagnosticPage{Status: "available", Items: []RuntimeDiagnostic{}}
	if err := validateDiagnosticIdentity(params.OwnerID, params.RunID, params.StageExecutionID); err != nil {
		return result, err
	}
	if params.Limit < 1 || params.Limit > MaxRuntimeDiagnosticLimit ||
		(params.After == "") != (params.Generation == "") {
		return result, fmt.Errorf("%w: invalid diagnostic page", ErrInvalid)
	}
	rows, err := r.db.Query(ctx, runtimeDiagnosticPageSQL,
		params.OwnerID, params.RunID, params.StageExecutionID,
		params.After, params.Limit+1, maxRuntimeParticipantBytes, contracts.NamePattern, contracts.MaxNameLength)
	if err != nil {
		return result, fmt.Errorf("read Runtime diagnostics: %w", err)
	}
	defer rows.Close()
	found := false
	usedBytes := 0
	for rows.Next() {
		var terminal bool
		var version, tuple, kind, name *string
		var updated *time.Time
		var document, progress []byte
		var stopTruncated, invalidNames bool
		if err := rows.Scan(&terminal, &version, &updated, &tuple, &kind, &name, &document, &stopTruncated, &invalidNames, &progress); err != nil {
			return result, fmt.Errorf("scan Runtime diagnostics: %w", err)
		}
		found = true
		if version == nil || updated == nil || tuple == nil {
			result.Status, result.Reason = diagnosticUnavailable(terminal)
			continue
		}
		result.Generation = diagnosticGeneration(*updated, *tuple)
		if params.Generation != "" && params.Generation != result.Generation {
			return result, fmt.Errorf("%w: diagnostic generation changed", ErrConflict)
		}
		if *version != contracts.APIVersion || kind == nil || *kind != "object" {
			result.Status, result.Reason = "unavailable", "invalid_report"
			continue
		}
		if name == nil {
			if invalidNames {
				result.Status, result.Reason = "unavailable", "invalid_report"
			}
			continue
		}
		if invalidNames {
			result.Reason = "invalid_report"
		}
		if !contracts.ValidName(*name) {
			result.Reason = "invalid_report"
			continue
		}
		if len(result.Items) == params.Limit {
			result.More = true
			break
		}
		entry := projectRuntimeDiagnostic(*name, document, stopTruncated)
		var observation RuntimeDiagnostic
		if err := strictjson.DecodeNonNull(progress, &observation); err == nil && (observation.Compaction == nil || observation.Compaction.Validate() == nil) && (observation.ModelPhases == nil || observation.ModelPhases.Validate() == nil) {
			entry.Compaction = observation.Compaction
			entry.ModelPhases = observation.ModelPhases
		} else {
			result.Reason = "invalid_report"
		}
		encoded, encodeErr := json.Marshal(entry)
		if encodeErr != nil {
			return result, fmt.Errorf("encode Runtime diagnostic: %w", encodeErr)
		}
		// Leave space for the fixed envelope and bounded sealed cursor. Every
		// individual entry is below this limit by its typed field bounds.
		if usedBytes+len(encoded)+1 > MaxDiagnosticPageBytes-4096 {
			result.More = true
			break
		}
		usedBytes += len(encoded) + 1
		result.Items = append(result.Items, entry)
		result.After = *name
	}
	if err := rows.Err(); err != nil {
		return result, fmt.Errorf("iterate Runtime diagnostics: %w", err)
	}
	if !found {
		return result, ErrNotFound
	}
	if result.Status == "pending" && r.workerProgress != nil {
		return r.readLiveWorkerDiagnostics(ctx, params)
	}
	if result.Status != "available" {
		result.Items, result.More, result.After = []RuntimeDiagnostic{}, false, ""
	}
	return result, nil
}

func validateDiagnosticIdentity(owner, run, stage string) error {
	if strings.TrimSpace(owner) == "" || len(owner) > contracts.MaxResourceIDLength ||
		!contracts.ValidResourceID(run) || !contracts.ValidResourceID(stage) {
		return fmt.Errorf("%w: invalid diagnostic identity", ErrInvalid)
	}
	return nil
}

func diagnosticGeneration(updated time.Time, tuple string) string {
	digest := sha256.Sum256([]byte(updated.UTC().Format(time.RFC3339Nano) + ":" + tuple))
	return hex.EncodeToString(digest[:])
}

func diagnosticUnavailable(terminal bool) (string, string) {
	if terminal {
		return "unavailable", "report_missing"
	}
	return "pending", "report_missing"
}

func projectRuntimeDiagnostic(name string, data []byte, stopTruncated bool) RuntimeDiagnostic {
	result := RuntimeDiagnostic{LogicalAgent: name, Status: "unavailable", Reason: "invalid_report", Adapters: []RuntimeAdapterDiagnostic{}}
	var report reporting.RuntimeReport
	if len(data) == 0 || len(data) > maxRuntimeParticipantBytes || strictjson.DecodeNonNull(data, &report) != nil || report.Validate() != nil {
		return result
	}
	result.Status, result.Reason = "available", ""
	result.Complete = &report.Complete
	if !report.Complete {
		result.Status = "partial"
	}
	if report.DurationMS != nil {
		value := strconv.FormatInt(*report.DurationMS, 10)
		result.DurationMS = &value
	}
	if report.StopReason != nil {
		value := diagnosticURLPattern.ReplaceAllString(*report.StopReason, "[REDACTED_URL]")
		result.StopReasonTruncated = stopTruncated || len(value) > maxDiagnosticStopBytes
		value = truncateUTF8(value, maxDiagnosticStopBytes)
		result.StopReason = &value
	}
	refs := make([]runtimesettings.RuntimeAdapterRef, 0, len(report.Adapters))
	for ref := range report.Adapters {
		refs = append(refs, ref)
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i] < refs[j] })
	for _, ref := range refs {
		metrics := report.Adapters[ref]
		entry := RuntimeAdapterDiagnostic{Ref: string(ref),
			Operations: strconv.FormatUint(metrics.Operations, 10), FailedOperations: strconv.FormatUint(metrics.FailedOperations, 10),
			FlushAttempted: metrics.FlushAttempted, FlushSucceeded: metrics.FlushSucceeded, LastErrorCode: metrics.LastErrorCode}
		if drops := metrics.DroppedSpans; drops != nil {
			entry.DroppedSpans = &DroppedSpanDiagnosticCounts{
				QueueOverflow: strconv.FormatUint(drops.QueueOverflow, 10), EncodingFailed: strconv.FormatUint(drops.EncodingFailed, 10),
				NonRetryable: strconv.FormatUint(drops.NonRetryable, 10), RetryExhausted: strconv.FormatUint(drops.RetryExhausted, 10),
				CollectorRejected: strconv.FormatUint(drops.CollectorRejected, 10), DeadlineExceeded: strconv.FormatUint(drops.DeadlineExceeded, 10),
				Cancelled: strconv.FormatUint(drops.Cancelled, 10), Shutdown: strconv.FormatUint(drops.Shutdown, 10)}
		}
		result.Adapters = append(result.Adapters, entry)
	}
	return result
}
