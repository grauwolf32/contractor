package projectsitemap

import (
	"bytes"
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// These typed projections consume the existing schemaVersion 1 tool report.
// Unknown fields are rejected instead of accepting a raw scanner stdout format.
type scannerEnvelope struct {
	SchemaVersion  int                              `json:"schemaVersion" strictjson:"required"`
	Tool           string                           `json:"tool" strictjson:"required"`
	InputDigest    string                           `json:"inputDigest" strictjson:"required"`
	InputArtifacts map[string]contracts.ArtifactRef `json:"inputArtifacts" strictjson:"required"`
	Observation    json.RawMessage                  `json:"observation" strictjson:"required"`
}

type processReport struct {
	Status              string          `json:"status" strictjson:"required"`
	ExitCode            *int64          `json:"exitCode" strictjson:"required,nullable"`
	ErrorCode           *string         `json:"errorCode" strictjson:"required,nullable"`
	Stdout              string          `json:"stdout" strictjson:"required"`
	Stderr              string          `json:"stderr" strictjson:"required"`
	StdoutTruncated     bool            `json:"stdoutTruncated" strictjson:"required"`
	StderrTruncated     bool            `json:"stderrTruncated" strictjson:"required"`
	OutputLimitExceeded bool            `json:"outputLimitExceeded" strictjson:"required"`
	DurationMS          int64           `json:"durationMs" strictjson:"required"`
	Scanner             string          `json:"scanner" strictjson:"required"`
	DiagnosticsRedacted bool            `json:"diagnosticsRedacted" strictjson:"required"`
	Results             json.RawMessage `json:"results" strictjson:"required"`
	ResultsTruncated    bool            `json:"resultsTruncated" strictjson:"required"`
	InvalidResultLines  uint64          `json:"invalidResultLines" strictjson:"required"`
}

type ffufReport struct {
	processReport
	WordlistArtifact  contracts.ArtifactRef `json:"wordlistArtifact" strictjson:"required"`
	WordlistEntries   int                   `json:"wordlistEntries" strictjson:"required"`
	PayloadsAttempted *uint64               `json:"payloadsAttempted" strictjson:"required,nullable"`
	RequestErrors     *uint64               `json:"requestErrors" strictjson:"required,nullable"`
	ScanComplete      bool                  `json:"scanComplete" strictjson:"required"`
}

type ffufMatch struct {
	Input struct {
		FUZZ string `json:"FUZZ" strictjson:"required"`
	} `json:"input" strictjson:"required"`
	Position         int    `json:"position" strictjson:"required"`
	Status           int    `json:"status" strictjson:"required"`
	Length           int64  `json:"length" strictjson:"required"`
	Words            int64  `json:"words" strictjson:"required"`
	Lines            int64  `json:"lines" strictjson:"required"`
	URL              string `json:"url" strictjson:"required"`
	DurationNS       int64  `json:"durationNs" strictjson:"required"`
	ContentType      string `json:"contentType" strictjson:"required"`
	RedirectLocation string `json:"redirectLocation" strictjson:"required"`
}

type katanaReport struct {
	processReport
	Artifacts         map[string]contracts.ArtifactRef `json:"artifacts" strictjson:"required"`
	TargetsArtifact   *contracts.ArtifactRef           `json:"targetsArtifact" strictjson:"required,nullable"`
	TargetsDigest     *string                          `json:"targetsDigest" strictjson:"required,nullable"`
	ArtifactErrorCode *string                          `json:"artifactErrorCode,omitempty"`
	Source            struct {
		Seed   string `json:"seed" strictjson:"required"`
		Origin string `json:"origin" strictjson:"required"`
	} `json:"source" strictjson:"required"`
	Limits struct {
		MaxDepth              int `json:"maxDepth" strictjson:"required"`
		MaxPages              int `json:"maxPages" strictjson:"required"`
		RateLimit             int `json:"rateLimit" strictjson:"required"`
		TimeoutSeconds        int `json:"timeoutSeconds" strictjson:"required"`
		CrawlDurationSeconds  int `json:"crawlDurationSeconds" strictjson:"required"`
		RequestTimeoutSeconds int `json:"requestTimeoutSeconds" strictjson:"required"`
		MaxTargets            int `json:"maxTargets" strictjson:"required"`
		MaxTargetBytes        int `json:"maxTargetBytes" strictjson:"required"`
		MaxResponseBytes      int `json:"maxResponseBytes" strictjson:"required"`
	} `json:"limits" strictjson:"required"`
	DiscoveryComplete bool   `json:"discoveryComplete" strictjson:"required"`
	RequestErrors     uint64 `json:"requestErrors" strictjson:"required"`
	Coverage          struct {
		Observations       uint64   `json:"observations" strictjson:"required"`
		Responses          uint64   `json:"responses" strictjson:"required"`
		RequestErrors      uint64   `json:"requestErrors" strictjson:"required"`
		DepthLimited       uint64   `json:"depthLimited" strictjson:"required"`
		OutOfScope         uint64   `json:"outOfScope" strictjson:"required"`
		UnsupportedMethods uint64   `json:"unsupportedMethods" strictjson:"required"`
		Unvisited          uint64   `json:"unvisited" strictjson:"required"`
		InvalidResultLines uint64   `json:"invalidResultLines" strictjson:"required"`
		Duplicates         uint64   `json:"duplicates" strictjson:"required"`
		UniqueTargets      uint64   `json:"uniqueTargets" strictjson:"required"`
		ExportedTargets    uint64   `json:"exportedTargets" strictjson:"required"`
		OmittedTargets     uint64   `json:"omittedTargets" strictjson:"required"`
		Limitations        []string `json:"limitations" strictjson:"required"`
	} `json:"coverage" strictjson:"required"`
}

type katanaMatch struct {
	URL        string `json:"url" strictjson:"required"`
	Method     string `json:"method" strictjson:"required"`
	StatusCode int    `json:"statusCode" strictjson:"required"`
	Source     string `json:"source" strictjson:"required"`
}

func scannerReport(data []byte, kind, at, sourceID string) (Mutation, error) {
	if len(data) > maxScannerReportBytes || !utf8.Valid(data) || !validTime(at) {
		return Mutation{}, ErrInvalid
	}
	if _, err := strictjson.DecodeBounded(data, strictjson.Bounds{MaxDepth: 16, MaxNodes: 100000, NamesAreNodes: true}, func(number string) (json.Number, bool) { return json.Number(number), true }); err != nil {
		return Mutation{}, ErrInvalid
	}
	var report scannerEnvelope
	if strictjson.DecodeNonNull(data, &report) != nil || report.SchemaVersion != 1 || report.Tool != "scan_"+kind || !contentdigest.Valid(report.InputDigest) {
		return Mutation{}, ErrInvalid
	}
	for name, ref := range report.InputArtifacts {
		if !contracts.ValidName(name) || ref.ValidateExact() != nil {
			return Mutation{}, ErrInvalid
		}
	}
	command := Mutation{Operation: "observe", Observations: []Observation{}, Gaps: []Gap{}}
	switch kind {
	case "ffuf":
		var parsed ffufReport
		if strictjson.DecodeNonNull(report.Observation, &parsed) != nil || !validProcessReport(parsed.processReport, kind) || parsed.WordlistArtifact.ValidateExact() != nil || parsed.WordlistEntries < 1 || parsed.WordlistEntries > 10000 {
			return Mutation{}, ErrInvalid
		}
		if parsed.PayloadsAttempted != nil && *parsed.PayloadsAttempted > uint64(parsed.WordlistEntries) || parsed.RequestErrors != nil && *parsed.RequestErrors > uint64(parsed.WordlistEntries) {
			return Mutation{}, ErrInvalid
		}
		if parsed.ScanComplete && (parsed.Status != "completed" || parsed.ExitCode == nil || *parsed.ExitCode != 0 || parsed.InvalidResultLines != 0 || parsed.PayloadsAttempted == nil || *parsed.PayloadsAttempted != uint64(parsed.WordlistEntries) || parsed.RequestErrors == nil || *parsed.RequestErrors != 0) {
			return Mutation{}, ErrInvalid
		}
		var rows []ffufMatch
		if decodeReportRows(parsed.Results, &rows) != nil || len(rows) > MaxBatch {
			return Mutation{}, ErrInvalid
		}
		for _, row := range rows {
			if row.Position < 1 || row.Position > parsed.WordlistEntries || len(row.Input.FUZZ) > 4096 || row.Length < 0 || row.Words < 0 || row.Lines < 0 || row.DurationNS < 0 || !validOutcome("response", row.Status) || !validScannerURL(row.URL) {
				return Mutation{}, ErrInvalid
			}
			command.Observations = append(command.Observations, Observation{URL: row.URL, Method: "GET", Outcome: "response", Status: row.Status, At: at})
		}
		appendReportGaps(&command, parsed.processReport, sourceID)
		if !parsed.ScanComplete {
			appendReportGap(&command, sourceID, "incomplete", 1)
		}
		if parsed.RequestErrors != nil {
			appendReportGap(&command, sourceID, "request-errors", *parsed.RequestErrors)
		}
	case "katana":
		var parsed katanaReport
		if strictjson.DecodeNonNull(report.Observation, &parsed) != nil || !validProcessReport(parsed.processReport, kind) || !validKatanaReport(parsed) {
			return Mutation{}, ErrInvalid
		}
		var rows []katanaMatch
		if decodeReportRows(parsed.Results, &rows) != nil || len(rows) > MaxBatch || uint64(len(rows)) != parsed.Coverage.ExportedTargets {
			return Mutation{}, ErrInvalid
		}
		seen := map[string]bool{}
		var targets strings.Builder
		for _, row := range rows {
			origin, _, _, err := projectURL(row.URL)
			parent, _, _, parentErr := projectURL(row.Source)
			if row.Method != "GET" || !validOutcome("response", row.StatusCode) || !validScannerURL(row.URL) || !validScannerURL(row.Source) || err != nil || parentErr != nil || origin != parsed.Source.Origin || parent != parsed.Source.Origin || seen[row.URL] {
				return Mutation{}, ErrInvalid
			}
			seen[row.URL] = true
			targets.WriteString(row.URL + "\n")
			command.Observations = append(command.Observations, Observation{URL: row.URL, Method: "GET", Outcome: "response", Status: row.StatusCode, At: at})
		}
		if len(rows) > 0 && (parsed.TargetsDigest == nil || *parsed.TargetsDigest != contentdigest.Bytes([]byte(targets.String()))) || len(rows) == 0 && parsed.TargetsDigest != nil {
			return Mutation{}, ErrInvalid
		}
		appendReportGaps(&command, parsed.processReport, sourceID)
		if parsed.Status == "failed" && parsed.ErrorCode != nil && *parsed.ErrorCode != "scan_artifact_failed" {
			appendReportGap(&command, sourceID, "incomplete", 1)
		}
		appendReportGap(&command, sourceID, "request-errors", parsed.RequestErrors)
		appendReportGap(&command, sourceID, "members", parsed.Coverage.OmittedTargets)
		appendReportGap(&command, sourceID, "projection", parsed.Coverage.DepthLimited+parsed.Coverage.OutOfScope+parsed.Coverage.UnsupportedMethods+parsed.Coverage.Unvisited)
	default:
		return Mutation{}, ErrInvalid
	}
	if len(jsonBytes(command)) > MaxBatchBytes {
		return Mutation{}, ErrInvalid
	}
	return command, nil
}

func validProcessReport(report processReport, kind string) bool {
	return report.Scanner == kind && slices.Contains([]string{"completed", "failed"}, report.Status) && report.DiagnosticsRedacted && report.Stdout == "" && report.Stderr == "" && report.DurationMS >= 0 && report.InvalidResultLines <= MaxOmissionCount &&
		(report.ErrorCode == nil && report.Status == "completed" && report.ExitCode != nil && *report.ExitCode == 0 || report.ErrorCode != nil && *report.ErrorCode != "" && report.Status == "failed")
}

func decodeReportRows(data json.RawMessage, target any) error {
	var compact bytes.Buffer
	if err := json.Compact(&compact, data); err != nil || compact.Len() > maxScannerResultBytes || bytes.Equal(compact.Bytes(), []byte("null")) || strictjson.DecodeNonNull(data, target) != nil {
		return ErrInvalid
	}
	return nil
}

func validScannerURL(raw string) bool {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.User != nil || parsed.Fragment != "" {
		return false
	}
	_, _, parameters, err := projectURL(raw)
	return err == nil && len(parameters) <= MaxBatch
}

func appendReportGaps(command *Mutation, report processReport, sourceID string) {
	appendReportGap(command, sourceID, "projection", report.InvalidResultLines)
	if report.ResultsTruncated || report.OutputLimitExceeded {
		appendReportGap(command, sourceID, "report-truncated", 1)
	}
}

func appendReportGap(command *Mutation, sourceID, reason string, count uint64) {
	if count == 0 {
		return
	}
	for i := range command.Gaps {
		if command.Gaps[i].Reason == reason {
			command.Gaps[i].Count = saturated(command.Gaps[i].Count, count)
			return
		}
	}
	command.Gaps = append(command.Gaps, Gap{SourceID: sourceID, Reason: reason, Count: count})
}

func validKatanaReport(report katanaReport) bool {
	limits := report.Limits
	origin, _, _, err := projectURL(report.Source.Seed)
	if err != nil || !validScannerURL(report.Source.Seed) || origin != report.Source.Origin || report.DiscoveryComplete || limits.MaxDepth < 1 || limits.MaxDepth > 5 || limits.MaxPages < 1 || limits.MaxPages > 1000 || limits.RateLimit < 1 || limits.RateLimit > 1000 || limits.TimeoutSeconds < 1 || limits.TimeoutSeconds > 3600 || limits.CrawlDurationSeconds != max(1, limits.TimeoutSeconds-2) || limits.RequestTimeoutSeconds != min(10, max(1, limits.TimeoutSeconds/3)) || limits.MaxTargets != MaxBatch || limits.MaxTargetBytes != maxScannerResultBytes || limits.MaxResponseBytes != 1<<20 {
		return false
	}
	coverage := report.Coverage
	for _, count := range []uint64{coverage.Observations, coverage.Responses, coverage.RequestErrors, coverage.DepthLimited, coverage.OutOfScope, coverage.UnsupportedMethods, coverage.Unvisited, coverage.InvalidResultLines, coverage.Duplicates, coverage.UniqueTargets, coverage.ExportedTargets, coverage.OmittedTargets} {
		if count > MaxOmissionCount {
			return false
		}
	}
	if coverage.RequestErrors != report.RequestErrors || coverage.InvalidResultLines != report.InvalidResultLines || coverage.UniqueTargets != coverage.ExportedTargets+coverage.OmittedTargets || coverage.Responses != coverage.UniqueTargets+coverage.Duplicates || report.ResultsTruncated != (coverage.OmittedTargets > 0) {
		return false
	}
	if coverage.Observations != coverage.Responses+coverage.RequestErrors+coverage.DepthLimited+coverage.OutOfScope+coverage.UnsupportedMethods+coverage.Unvisited+coverage.InvalidResultLines {
		return false
	}
	allowed := []string{"bounded_crawl", "exhaustion_unverified", "response_size_limit", "default_extension_filter", "depth_limit", "page_limit", "request_errors", "out_of_scope_observations", "unsupported_methods", "unvisited_observations", "invalid_output", "target_export_limit", "process_incomplete"}
	seen := map[string]bool{}
	for _, limitation := range coverage.Limitations {
		if !slices.Contains(allowed, limitation) || seen[limitation] {
			return false
		}
		seen[limitation] = true
	}
	for _, required := range allowed[:4] {
		if !seen[required] {
			return false
		}
	}
	for _, condition := range []struct {
		reason  string
		present bool
	}{
		{"depth_limit", coverage.DepthLimited > 0},
		{"page_limit", coverage.Responses+coverage.RequestErrors >= uint64(limits.MaxPages)},
		{"request_errors", coverage.RequestErrors > 0},
		{"out_of_scope_observations", coverage.OutOfScope > 0},
		{"unsupported_methods", coverage.UnsupportedMethods > 0},
		{"unvisited_observations", coverage.Unvisited > 0},
		{"invalid_output", coverage.InvalidResultLines > 0},
		{"target_export_limit", coverage.OmittedTargets > 0},
	} {
		if seen[condition.reason] != condition.present {
			return false
		}
	}
	if report.TargetsArtifact == nil {
		return len(report.Artifacts) == 0 && (coverage.ExportedTargets == 0 || report.ArtifactErrorCode != nil && *report.ArtifactErrorCode == "scan_artifact_failed")
	}
	target, found := report.Artifacts["targets"]
	return report.TargetsArtifact.ValidateExact() == nil && found && len(report.Artifacts) == 1 && target.SameExact(*report.TargetsArtifact) && report.ArtifactErrorCode == nil
}
