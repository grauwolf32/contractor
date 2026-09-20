package scan

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner"
	"github.com/grauwolf32/contractor/internal/scanplan"
)

type Coverage struct {
	Candidates  int  `json:"candidates"`
	Selected    int  `json:"selected"`
	Skipped     int  `json:"skipped"`
	Unavailable int  `json:"unavailable"`
	Completed   int  `json:"completed"`
	Failed      int  `json:"failed"`
	Incomplete  int  `json:"incomplete"`
	Unknown     int  `json:"unknown"`
	Complete    bool `json:"complete"`
}

type Report struct {
	SchemaVersion int                     `json:"schemaVersion"`
	Plan          contracts.ArtifactRef   `json:"plan"`
	PlanID        string                  `json:"planId"`
	Jobs          []planner.ScanJobRecord `json:"jobs"`
	Coverage      Coverage                `json:"coverage"`
}

type workerReport struct {
	SchemaVersion  int                              `json:"schemaVersion"`
	Tool           string                           `json:"tool"`
	InputDigest    string                           `json:"inputDigest"`
	InputArtifacts map[string]contracts.ArtifactRef `json:"inputArtifacts"`
	Observation    map[string]json.RawMessage       `json:"observation"`
}

var workerDigest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func (p *execution) observe(ctx context.Context, job scanplan.ScanJob, request contracts.StageContentRequest, record planner.ScanJobRecord, completion contracts.WorkerCompletion) planner.ScanJobRecord {
	if planner.ValidateWorkerCompletion(completion, job.ID) != nil {
		record.Status, record.Code = planner.ScanJobUnknown, "scan_invalid_worker_result"
		return record
	}
	failedReport := completion.Failure != nil && completion.Failure.Code == "tool_execution_failed"
	if completion.Failure != nil && !failedReport {
		switch completion.Failure.Code {
		case "worker_busy", "worker_draining", "worker_unavailable", "scan_unavailable":
			record.Status, record.Code = planner.ScanJobUnavailable, "scan_worker_unavailable"
		case "tool_timeout", "tool_cancelled", "tool_output_invalid":
			record.Status, record.Code = planner.ScanJobIncomplete, "scan_incomplete"
		case "tool_outcome_unknown", "tool_report_failed", "tool_input_conflict":
			record.Status, record.Code = planner.ScanJobUnknown, "scan_outcome_unknown"
		default:
			record.Status, record.Code = planner.ScanJobFailed, "scan_failed"
		}
		return record
	}
	target := p.jobOutput(job)
	readCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), journalIOTimeout)
	defer cancel()
	var ref contracts.ArtifactRef
	if failedReport {
		var err error
		ref, err = p.factory.artifacts.Resolve(readCtx, p.invocation.RunID, target)
		if err != nil {
			record.Status, record.Code = planner.ScanJobFailed, "scan_failed"
			return record
		}
	} else {
		result := completion.Result
		var ok bool
		ref, ok = result.Artifacts[job.Execution.ResultArtifact]
		if !ok || len(result.Artifacts) != 1 {
			record.Status, record.Code = planner.ScanJobUnknown, "scan_invalid_worker_result"
			return record
		}
	}
	if ref.ValidateExact() != nil || ref.Namespace != target.Namespace || ref.Name != target.Name {
		record.Status, record.Code = planner.ScanJobUnknown, "scan_invalid_worker_result"
		return record
	}
	payload, err := p.factory.artifacts.Read(readCtx, p.invocation.RunID, ref, planner.MaxScanReportBytes)
	if err != nil || payload.MediaType != "application/json" {
		record.Status, record.Code = planner.ScanJobIncomplete, "scan_report_unavailable"
		return record
	}
	var report workerReport
	if !strictWorkerReport(payload.Data, &report) || report.SchemaVersion != 1 || report.Tool != job.Tool || !workerDigest.MatchString(report.InputDigest) || !reflect.DeepEqual(report.InputArtifacts, request.Artifacts) {
		record.Status, record.Code = planner.ScanJobIncomplete, "scan_report_invalid"
		return record
	}
	exact := planner.CloneArtifactRef(ref)
	record.Report = &exact
	record.Status, record.Code = observationStatus(report.Observation)
	if failedReport && record.Status == planner.ScanJobCompleted {
		record.Status, record.Code = planner.ScanJobUnknown, "scan_invalid_worker_result"
	}
	if completion.Result != nil && completion.Result.Observations.Truncated && record.Status == planner.ScanJobCompleted {
		record.Status, record.Code = planner.ScanJobIncomplete, "scan_incomplete"
	}
	return record
}

func observationStatus(observation map[string]json.RawMessage) (string, string) {
	invalid := func() (string, string) { return planner.ScanJobIncomplete, "scan_report_invalid" }
	var status, errorCode string
	if json.Unmarshal(observation["status"], &status) != nil || (status != "completed" && status != "failed") {
		return invalid()
	}
	if raw, ok := observation["errorCode"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		if json.Unmarshal(raw, &errorCode) != nil || errorCode == "" {
			return invalid()
		}
	}
	// Scanner timeouts may have no exit code or a signal exit code. Keep those
	// technical outcomes distinct from an ordinary nonzero scanner exit.
	switch errorCode {
	case "scanner_unavailable":
		return planner.ScanJobUnavailable, "scan_worker_unavailable"
	case "scan_timeout", "output_limit_exceeded", "scan_incomplete", "scan_request_failed", "invalid_scanner_output":
		return planner.ScanJobIncomplete, "scan_incomplete"
	}
	var exitCode int
	if json.Unmarshal(observation["exitCode"], &exitCode) != nil || bytes.Equal(bytes.TrimSpace(observation["exitCode"]), []byte("null")) {
		return invalid()
	}
	if status == "failed" || exitCode != 0 {
		return planner.ScanJobFailed, "scan_failed"
	}
	incomplete := errorCode != ""
	names := make([]string, 0, len(observation))
	for name := range observation {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		raw := observation[name]
		switch {
		case strings.HasSuffix(name, "Truncated") || name == "outputLimitExceeded" || name == "scanComplete":
			var value bool
			if json.Unmarshal(raw, &value) != nil || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				return invalid()
			}
			if name == "scanComplete" {
				incomplete = incomplete || !value
			} else {
				incomplete = incomplete || value
			}
		case name == "invalidResultLines":
			var value int
			if json.Unmarshal(raw, &value) != nil || value < 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				return invalid()
			}
			incomplete = incomplete || value > 0
		}
	}
	if incomplete {
		return planner.ScanJobIncomplete, "scan_incomplete"
	}
	return planner.ScanJobCompleted, ""
}

func (p *execution) finish(ctx context.Context, identity planner.ScanSessionIdentity, plan scanplan.ScanPlan, state planner.ScanState) (contracts.StageContentResult, error) {
	if p.audit != nil {
		history, err := p.auditHistory(ctx)
		if err != nil {
			return contracts.StageContentResult{}, err
		}
		found := false
		for index := range history {
			if history[index].StageExecutionID == p.invocation.StageExecutionID {
				history[index].State = state
				found = true
			}
		}
		if !found {
			return contracts.StageContentResult{}, scanError("scan_history_invalid", nil)
		}
		return p.finishAudit(ctx, identity, history)
	}
	empty := contracts.StageContentResult{}
	coverage := Coverage{Candidates: len(plan.Candidates), Selected: len(plan.Jobs)}
	for _, candidate := range plan.Candidates {
		switch candidate.Selection {
		case "skipped":
			coverage.Skipped++
		case "unavailable":
			coverage.Unavailable++
		}
	}
	for _, record := range state.Jobs {
		switch record.Status {
		case planner.ScanJobCompleted:
			coverage.Completed++
		case planner.ScanJobFailed:
			coverage.Failed++
		case planner.ScanJobIncomplete:
			coverage.Incomplete++
		case planner.ScanJobUnavailable:
			coverage.Unavailable++
		case planner.ScanJobUnknown:
			coverage.Unknown++
		default:
			return empty, scanError("scan_session_invalid", nil)
		}
	}
	coverage.Complete = coverage.Selected > 0 && coverage.Completed == coverage.Selected && coverage.Skipped == 0 && coverage.Unavailable == 0 && len(plan.PreparationGaps) == 0 && (plan.PreparationCoverage == nil || plan.PreparationCoverage.Complete)
	if state.Plan == nil {
		return empty, scanError("scan_session_invalid", nil)
	}
	report := Report{SchemaVersion: 1, Plan: planner.CloneArtifactRef(*state.Plan), PlanID: plan.ID, Jobs: state.Jobs, Coverage: coverage}
	data, err := contracts.MarshalPrivateCanonical(report)
	if err != nil || len(data) > scanplan.MaxPlanBytes {
		return empty, scanError("scan_report_invalid", err)
	}
	slot := p.invocation.Stage.Result.Artifacts["report"].From
	target := contracts.ArtifactRef{Namespace: slot.Namespace, Name: slot.Name + "." + stableHash(p.invocation.StageExecutionID)[:16]}
	writeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), journalIOTimeout)
	defer cancel()
	ref, err := p.factory.artifacts.Create(writeCtx, p.invocation.RunID, target, artifacts.Payload{MediaType: "application/json", Data: data})
	if err != nil {
		return empty, scanError("scan_report_write_failed", err)
	}
	result := contracts.StageContentResult{APIVersion: contracts.APIVersion, Outcome: contracts.StageSucceeded, Summary: fmt.Sprintf("Scan jobs: %d completed, %d failed, %d incomplete, %d unknown, %d unavailable; %d candidates skipped. Inspect reports for evidence and coverage limits.", coverage.Completed, coverage.Failed, coverage.Incomplete, coverage.Unknown, coverage.Unavailable, coverage.Skipped), Artifacts: map[string]contracts.ArtifactRef{"report": ref}}
	if coverage.Selected == 0 || coverage.Completed != coverage.Selected {
		result.Outcome = contracts.StageFailed
		result.Error = &contracts.TerminationError{Code: "scan_incomplete", Message: "The scan plan did not complete all selected jobs", Retryable: false}
	}
	if validation := planner.ValidateCandidate(writeCtx, p.invocation.RunID, p.invocation.Stage.Result.Artifacts, result, p.factory.inspector); validation != nil {
		return empty, validation
	}
	if err := p.factory.sessions.CompleteScan(writeCtx, identity, planner.Completion{Result: &result}); err != nil {
		return empty, scanError("scan_completion_write_failed", err)
	}
	return planner.CloneStageResult(result), nil
}

func strictWorkerReport(data []byte, out *workerReport) bool {
	if len(data) > planner.MaxScanReportBytes || !utf8.Valid(data) {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	nodes := 0
	var read func(int) bool
	read = func(depth int) bool {
		nodes++
		if depth > 64 || nodes > 100000 {
			return false
		}
		token, err := decoder.Token()
		if err != nil {
			return false
		}
		switch token {
		case json.Delim('{'):
			seen := map[string]bool{}
			for decoder.More() {
				raw, err := decoder.Token()
				key, ok := raw.(string)
				if err != nil || !ok || seen[key] {
					return false
				}
				seen[key] = true
				if !read(depth + 1) {
					return false
				}
			}
			end, err := decoder.Token()
			return err == nil && end == json.Delim('}')
		case json.Delim('['):
			for decoder.More() {
				if !read(depth + 1) {
					return false
				}
			}
			end, err := decoder.Token()
			return err == nil && end == json.Delim(']')
		}
		return true
	}
	if !read(1) {
		return false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(data, &fields) != nil || len(fields) != 5 {
		return false
	}
	for _, name := range []string{"schemaVersion", "tool", "inputDigest", "inputArtifacts", "observation"} {
		if raw, ok := fields[name]; !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return false
		}
	}
	decoder = json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if decoder.Decode(out) != nil || out.Observation == nil || out.InputArtifacts == nil {
		return false
	}
	return true
}
