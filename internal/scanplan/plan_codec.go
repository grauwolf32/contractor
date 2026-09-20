package scanplan

import (
	"bytes"
	"encoding/json"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

var planDigestPattern = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
var planIdentityPattern = regexp.MustCompile(`^(plan|candidate|job)-[a-f0-9]{64}$`)
var planCodePattern = regexp.MustCompile(`^[a-z][a-z0-9_]{0,127}$`)
var planRequestIDPattern = regexp.MustCompile(`^request-[a-f0-9]{64}$`)
var planLineIDPattern = regexp.MustCompile(`^line-[0-9]{4,7}$`)
var planPointerPattern = regexp.MustCompile(`^#(?:/(?:[^~]|~[01])*)*$`)

func MarshalPlan(plan ScanPlan) ([]byte, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	data, err := contracts.MarshalPrivateCanonical(plan)
	if err != nil || len(data) > MaxPlanBytes {
		return nil, failure("scan_plan_limit_exceeded")
	}
	return data, nil
}

func DecodePlan(data []byte) (ScanPlan, error) {
	empty := ScanPlan{}
	if len(data) > MaxPlanBytes || !utf8.Valid(data) {
		return empty, failure("invalid_scan_plan")
	}
	if _, err := parseJSONDocument(data); err != nil {
		return empty, failure("invalid_scan_plan")
	}
	if !planJSONShape(data, reflect.TypeOf(empty)) {
		return empty, failure("invalid_scan_plan_shape")
	}
	var plan ScanPlan
	if json.Unmarshal(data, &plan) != nil {
		return empty, failure("invalid_scan_plan")
	}
	if err := plan.Validate(); err != nil {
		return empty, err
	}
	return plan, nil
}

func (p ScanPlan) Validate() error {
	bad := func() error { return failure("invalid_scan_plan") }
	if p.SchemaVersion != 1 || p.Source.Artifact.ValidateExact() != nil || !planDigestPattern.MatchString(p.Source.ContentDigest) || p.Policy.Validate() != nil || p.Candidates == nil || p.Jobs == nil || p.PreparationGaps == nil || len(p.Candidates) > MaxPlanCandidates || len(p.Jobs) > p.Policy.MaxJobs || len(p.PreparationGaps) > contracts.MaxHTTPRequestSetGaps {
		return bad()
	}
	if !utf8.ValidString(*p.Source.Artifact.Revision) || !validPlanPreparation(p) {
		return bad()
	}
	policies := map[string]contracts.ScanToolPolicy{}
	previous := ""
	for _, tool := range p.Policy.Tools {
		if tool.Worker <= previous || !sort.StringsAreSorted(tool.TestParameters) {
			return bad()
		}
		previous = tool.Worker
		policies[tool.Worker] = tool
	}
	candidates := map[string]ScanCandidate{}
	workerTools, toolWorkers := map[string]string{}, map[string]string{}
	workerSources := map[string]map[string]bool{}
	allSources, selectedSources := map[string]bool{}, map[string]bool{}
	selected := 0
	previous = ""
	for _, candidate := range p.Candidates {
		if !strings.HasPrefix(candidate.ID, "candidate-") || !planIdentityPattern.MatchString(candidate.ID) || candidate.ID <= previous || candidate.SourceIDs == nil || len(candidate.SourceIDs) < 1 || len(candidate.SourceIDs) > 1000 || policies[candidate.Worker].Worker == "" || !knownScanner(candidate.Tool) {
			return bad()
		}
		previous = candidate.ID
		if previousTool := workerTools[candidate.Worker]; previousTool != "" && previousTool != candidate.Tool {
			return bad()
		}
		if previousWorker := toolWorkers[candidate.Tool]; previousWorker != "" && previousWorker != candidate.Worker {
			return bad()
		}
		workerTools[candidate.Worker], toolWorkers[candidate.Tool] = candidate.Tool, candidate.Worker
		if !validPlanToolPolicy(policies[candidate.Worker], candidate.Tool) {
			return bad()
		}
		if workerSources[candidate.Worker] == nil {
			workerSources[candidate.Worker] = map[string]bool{}
		}
		priorSource := ""
		for _, source := range candidate.SourceIDs {
			if source <= priorSource || !validPlanSourceID(source, p.Source.MediaType) || workerSources[candidate.Worker][source] {
				return bad()
			}
			priorSource = source
			workerSources[candidate.Worker][source] = true
			allSources[source] = true
			if len(allSources) > 1000 {
				return bad()
			}
			if candidate.Selection == "selected" {
				selectedSources[source] = true
			}
		}
		switch candidate.Selection {
		case "selected":
			if candidate.Code != "" {
				return bad()
			}
			selected++
		case "skipped", "unavailable":
			if !planCodePattern.MatchString(candidate.Code) {
				return bad()
			}
		default:
			return bad()
		}
		candidates[candidate.ID] = candidate
	}
	if selected != len(p.Jobs) || len(allSources) > 1000 || len(selectedSources) > p.Policy.MaxInputs {
		return bad()
	}
	for worker := range policies {
		if len(workerSources[worker]) != len(allSources) {
			return bad()
		}
	}
	if p.Source.MediaType == TargetListMediaType && len(allSources) == 0 {
		return bad()
	}
	if p.PreparationCoverage != nil && (len(allSources) > p.PreparationCoverage.Prepared || (len(allSources) == 0) != (p.PreparationCoverage.Prepared == 0)) {
		return bad()
	}
	counts, seconds := map[string]int{}, map[string]int{}
	seenTools := map[string]string{}
	totalSeconds := 0
	previous = ""
	for _, job := range p.Jobs {
		candidate, ok := candidates[job.CandidateID]
		if !ok || candidate.Selection != "selected" || job.CandidateID <= previous || candidate.Worker != job.Worker || candidate.Tool != job.Tool || job.Execution.Tool != job.Tool || job.Execution.Validate() != nil || job.Execution.ResultArtifact != "report" || job.TimeoutSeconds != job.Execution.TimeoutSeconds || job.ID != jobIdentity(p.ID, job.CandidateID) || contracts.ValidateArtifactName(job.Namespace) != nil || !planDigestPattern.MatchString(job.TemplateDigest) {
			return bad()
		}
		parts := strings.Split(job.TemplateRef, "@")
		if len(parts) != 2 || (contracts.AgentTemplateRef{TemplateID: parts[0], Version: parts[1], Digest: job.TemplateDigest}).ValidateRef() != nil || !validPlanExecution(job.Execution) || job.Parameters == nil || job.Artifacts == nil || len(job.Parameters) > 1 || len(job.Artifacts) > 1 {
			return bad()
		}
		if owner, ok := seenTools[job.Tool]; ok && owner != job.Worker {
			return bad()
		}
		seenTools[job.Tool] = job.Worker
		previous = job.CandidateID
		for name, value := range job.Parameters {
			if contracts.ValidateArtifactName(name) != nil || len(value) > 8192 || !utf8.ValidString(value) {
				return bad()
			}
		}
		for name, ref := range job.Artifacts {
			if contracts.ValidateArtifactName(name) != nil || ref.ValidateExact() != nil || !utf8.ValidString(*ref.Revision) {
				return bad()
			}
		}
		policy := policies[job.Worker]
		switch job.Tool {
		case "scan_sqlmap":
			if job.Request == nil || job.Request.SchemaVersion != 1 || !reflect.DeepEqual(job.Request.TestParameters, policy.TestParameters) || len(job.Parameters) != 0 || len(job.Artifacts) != 0 {
				return bad()
			}
			request := contracts.PreparedHTTPRequest{Method: job.Request.Method, URL: job.Request.URL, Headers: job.Request.Headers, Body: job.Request.Body}
			if _, code := prepareSQLMapRequest(request, job.Request.TestParameters); code != "" {
				return bad()
			}
		case "scan_naabu":
			if job.Request != nil || len(job.Parameters) != 1 || len(job.Artifacts) != 0 || !scanHost(job.Parameters["target"]) || job.Parameters["target"] != strings.ToLower(job.Parameters["target"]) {
				return bad()
			}
		case "scan_nuclei", "scan_ffuf":
			if job.Request != nil || len(job.Parameters) != 1 {
				return bad()
			}
			u, code := scanURL(job.Parameters["target"])
			if code != "" {
				return bad()
			}
			if job.Tool == "scan_ffuf" {
				if len(job.Artifacts) != 1 || job.Artifacts["wordlist"].Revision == nil || strings.Contains(job.Parameters["target"], "FFUFHASH") || strings.Contains(u.Host, "FUZZ") || !strings.Contains(u.EscapedPath()+u.RawQuery, "FUZZ") {
					return bad()
				}
			} else if len(job.Artifacts) != 0 {
				return bad()
			}
		}
		semantic := job
		semantic.ID = ""
		semantic.CandidateID = ""
		if candidate.ID != semanticID("candidate", semantic) {
			return bad()
		}
		counts[job.Worker]++
		seconds[job.Worker] += job.TimeoutSeconds
		totalSeconds += job.TimeoutSeconds
		if counts[job.Worker] > policy.MaxJobs || seconds[job.Worker] > policy.MaxTotalSeconds || totalSeconds > p.Policy.MaxTotalSeconds {
			return bad()
		}
	}
	if p.ID != planIdentity(p) {
		return bad()
	}
	return nil
}

func validPlanPreparation(p ScanPlan) bool {
	if p.Source.MediaType == TargetListMediaType {
		return p.PreparationCoverage == nil && len(p.PreparationGaps) == 0
	}
	if p.Source.MediaType != contracts.HTTPRequestSetMediaType || p.PreparationCoverage == nil {
		return false
	}
	coverage := *p.PreparationCoverage
	if coverage.Operations < 0 || coverage.Operations > 1000 || coverage.Prepared < 0 || coverage.Prepared > 1000 || coverage.Skipped < 0 || coverage.Skipped > 1000 || coverage.Prepared+coverage.Skipped != coverage.Operations || coverage.Complete != (coverage.Skipped == 0 && len(p.PreparationGaps) == 0) {
		return false
	}
	pointer, code := "", ""
	for _, gap := range p.PreparationGaps {
		if !utf8.ValidString(gap.Pointer) || len(gap.Pointer) > 8192 || !planPointerPattern.MatchString(gap.Pointer) || len(gap.Code) > 64 || !planCodePattern.MatchString(gap.Code) || gap.Pointer < pointer || gap.Pointer == pointer && gap.Code <= code {
			return false
		}
		pointer, code = gap.Pointer, gap.Code
	}
	return true
}

func validPlanSourceID(source, mediaType string) bool {
	if mediaType == contracts.HTTPRequestSetMediaType {
		return planRequestIDPattern.MatchString(source)
	}
	if !planLineIDPattern.MatchString(source) {
		return false
	}
	line, err := strconv.Atoi(strings.TrimPrefix(source, "line-"))
	return err == nil && line > 0 && line <= MaxSourceBytes
}

func validPlanToolPolicy(policy contracts.ScanToolPolicy, tool string) bool {
	if (tool == "scan_sqlmap") != (len(policy.TestParameters) > 0) {
		return false
	}
	return (tool == "scan_ffuf") == (policy.WordlistArtifact != "")
}

func validPlanExecution(execution contracts.ToolExecutionConfig) bool {
	dynamic := map[string]contracts.ToolArgumentBinding{}
	switch execution.Tool {
	case "scan_nuclei", "scan_ffuf":
		dynamic["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
	case "scan_naabu":
		dynamic["host"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
	case "scan_sqlmap":
		dynamic["request_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "request"}
		if _, exists := execution.Arguments["url"]; exists {
			return false
		}
	default:
		return false
	}
	if execution.Tool == "scan_ffuf" {
		dynamic["wordlist_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "wordlist"}
	}
	for name, expected := range dynamic {
		actual, exists := execution.Arguments[name]
		if !exists || actual.Source != expected.Source || actual.Name != expected.Name || actual.Value != nil {
			return false
		}
	}
	for name, actual := range execution.Arguments {
		if _, exists := dynamic[name]; !exists && actual.Source != "literal" {
			return false
		}
	}
	return true
}

func knownScanner(tool string) bool {
	switch tool {
	case "scan_nuclei", "scan_naabu", "scan_sqlmap", "scan_ffuf":
		return true
	}
	return false
}

// Reject omitted required fields, explicit nulls and case-insensitive aliases
// before encoding/json can normalize them. Optional fields may only be absent.
func planJSONShape(data json.RawMessage, kind reflect.Type) bool {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return false
	}
	if kind.Kind() == reflect.Pointer {
		return planJSONShape(data, kind.Elem())
	}
	switch kind.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if json.Unmarshal(data, &fields) != nil || fields == nil {
			return false
		}
		for i := 0; i < kind.NumField(); i++ {
			field := kind.Field(i)
			tag := strings.Split(field.Tag.Get("json"), ",")
			name := tag[0]
			if name == "-" {
				continue
			}
			if name == "" {
				name = field.Name
			}
			value, exists := fields[name]
			optional := len(tag) > 1 && tag[1] == "omitempty"
			if !exists {
				if optional {
					continue
				}
				return false
			}
			if !planJSONShape(value, field.Type) {
				return false
			}
			delete(fields, name)
		}
		return len(fields) == 0
	case reflect.Map:
		var fields map[string]json.RawMessage
		if json.Unmarshal(data, &fields) != nil || fields == nil {
			return false
		}
		for _, value := range fields {
			if !planJSONShape(value, kind.Elem()) {
				return false
			}
		}
	case reflect.Slice:
		var values []json.RawMessage
		if json.Unmarshal(data, &values) != nil || values == nil {
			return false
		}
		for _, value := range values {
			if !planJSONShape(value, kind.Elem()) {
				return false
			}
		}
	}
	return true
}
