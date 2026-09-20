package scanplan

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type planSourceInput struct {
	id, target string
	request    *contracts.PreparedHTTPRequest
}
type plannedCandidate struct {
	candidate ScanCandidate
	job       ScanJob
}

// BuildPlan is pure: it performs no Artifact access, Worker calls or network I/O.
// Binding templates, exact refs and source bytes must come from the Run snapshot.
func BuildPlan(input PlanInput, policy contracts.ScanPlanPolicy, bindings map[string]ToolBinding, wordlists map[string]contracts.ArtifactRef) (ScanPlan, error) {
	empty := ScanPlan{}
	if input.Artifact.ValidateExact() != nil || policy.Validate() != nil {
		return empty, failure("invalid_scan_plan_input")
	}
	// Detach all caller-owned maps, slices, revision pointers and literal values.
	raw, err := json.Marshal(struct {
		Policy    contracts.ScanPlanPolicy
		Bindings  map[string]ToolBinding
		Wordlists map[string]contracts.ArtifactRef
	}{policy, bindings, wordlists})
	if err != nil || len(raw) > MaxPlanBytes {
		return empty, failure("invalid_scan_plan_bindings")
	}
	var detached struct {
		Policy    contracts.ScanPlanPolicy
		Bindings  map[string]ToolBinding
		Wordlists map[string]contracts.ArtifactRef
	}
	if json.Unmarshal(raw, &detached) != nil {
		return empty, failure("invalid_scan_plan_bindings")
	}
	policy, bindings, wordlists = detached.Policy, detached.Bindings, detached.Wordlists
	sort.Slice(policy.Tools, func(i, j int) bool { return policy.Tools[i].Worker < policy.Tools[j].Worker })
	for i := range policy.Tools {
		sort.Strings(policy.Tools[i].TestParameters)
	}
	if err := validatePlanBindings(policy, bindings, wordlists); err != nil {
		return empty, err
	}
	revision := *input.Artifact.Revision
	input.Artifact.Revision = &revision
	plan := ScanPlan{SchemaVersion: 1, Source: PlanSource{Artifact: input.Artifact, MediaType: input.MediaType, ContentDigest: digest(input.Data)}, Policy: policy, PreparationGaps: []contracts.PreparationGap{}, Candidates: []ScanCandidate{}, Jobs: []ScanJob{}}
	sources, err := parsePlanSources(input, &plan)
	if err != nil {
		return empty, err
	}
	candidates := map[string]*plannedCandidate{}
	for index, source := range sources {
		for _, toolPolicy := range policy.Tools {
			binding := bindings[toolPolicy.Worker]
			job, code := planJob(source, toolPolicy, binding, wordlists)
			identity := job
			// Unsupported candidates still get an exact stable identity for accounting.
			var basis any = identity
			if code != "" {
				basis = struct{ Worker, Tool, SourceID, Code string }{toolPolicy.Worker, binding.Template.Execution.Tool, source.id, code}
			}
			id := semanticID("candidate", basis)
			if index >= policy.MaxInputs {
				code = "input_budget_exceeded"
				id = semanticID("candidate", struct{ Worker, SourceID, Code string }{toolPolicy.Worker, source.id, code})
			}
			if existing, ok := candidates[id]; ok {
				existing.candidate.SourceIDs = append(existing.candidate.SourceIDs, source.id)
				continue
			}
			selection := "selected"
			if code != "" {
				selection = "skipped"
			}
			candidates[id] = &plannedCandidate{candidate: ScanCandidate{ID: id, Worker: toolPolicy.Worker, Tool: binding.Template.Execution.Tool, SourceIDs: []string{source.id}, Selection: selection, Code: code}, job: job}
		}
	}
	if len(candidates) > MaxPlanCandidates {
		return empty, failure("scan_candidate_limit_exceeded")
	}
	policies := map[string]contracts.ScanToolPolicy{}
	for _, p := range policy.Tools {
		policies[p.Worker] = p
	}
	counts, seconds := map[string]int{}, map[string]int{}
	totalSeconds := 0
	for _, id := range keys(candidates) {
		candidate := candidates[id]
		c := &candidate.candidate
		sort.Strings(c.SourceIDs)
		if c.Selection == "selected" {
			p := policies[c.Worker]
			timeout := candidate.job.TimeoutSeconds
			switch {
			case len(plan.Jobs) >= policy.MaxJobs:
				c.Code = "job_budget_exceeded"
			case counts[c.Worker] >= p.MaxJobs:
				c.Code = "tool_job_budget_exceeded"
			case totalSeconds+timeout > policy.MaxTotalSeconds:
				c.Code = "time_budget_exceeded"
			case seconds[c.Worker]+timeout > p.MaxTotalSeconds:
				c.Code = "tool_time_budget_exceeded"
			}
			if c.Code != "" {
				c.Selection = "skipped"
			} else {
				candidate.job.CandidateID = c.ID
				plan.Jobs = append(plan.Jobs, candidate.job)
				totalSeconds += timeout
				counts[c.Worker]++
				seconds[c.Worker] += timeout
			}
		}
		plan.Candidates = append(plan.Candidates, *c)
	}
	plan.ID = planIdentity(plan)
	for i := range plan.Jobs {
		plan.Jobs[i].ID = jobIdentity(plan.ID, plan.Jobs[i].CandidateID)
	}
	if _, err := MarshalPlan(plan); err != nil {
		return empty, err
	}
	return plan, nil
}

func validatePlanBindings(policy contracts.ScanPlanPolicy, bindings map[string]ToolBinding, wordlists map[string]contracts.ArtifactRef) error {
	if len(bindings) != len(policy.Tools) {
		return failure("invalid_scan_plan_bindings")
	}
	tools := map[string]bool{}
	for _, p := range policy.Tools {
		b, ok := bindings[p.Worker]
		if !ok || b.Template.Validate() != nil || !b.Template.IsToolWorker() || b.Template.Execution == nil || contracts.ValidateArtifactName(b.Namespace) != nil {
			return failure("invalid_scan_plan_bindings")
		}
		e := b.Template.Execution
		if e.ResultArtifact != "report" || !validPlanExecution(*e) || len(b.Template.Toolsets) != 1 || b.Template.Toolsets[0].Ref != (contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}) {
			return failure("invalid_scan_tool_binding")
		}
		if tools[e.Tool] {
			return failure("duplicate_scan_tool")
		}
		tools[e.Tool] = true
		dynamic := map[string]string{}
		switch e.Tool {
		case "scan_nuclei":
			dynamic["url"] = "parameter"
		case "scan_naabu":
			dynamic["host"] = "parameter"
		case "scan_sqlmap":
			dynamic["request_ref"] = "artifact"
		case "scan_ffuf":
			dynamic["url"] = "parameter"
			dynamic["wordlist_ref"] = "artifact"
		default:
			return failure("unsupported_scan_tool")
		}
		for name, source := range dynamic {
			arg, ok := e.Arguments[name]
			if !ok || arg.Source != source {
				return failure("invalid_scan_tool_binding")
			}
		}
		for name, arg := range e.Arguments {
			if _, ok := dynamic[name]; !ok && arg.Source != "literal" {
				return failure("invalid_scan_tool_binding")
			}
		}
		if e.Tool == "scan_sqlmap" {
			if len(p.TestParameters) == 0 {
				return failure("missing_scan_test_parameters")
			}
		} else if len(p.TestParameters) != 0 {
			return failure("unexpected_scan_test_parameters")
		}
		if e.Tool == "scan_ffuf" {
			ref, ok := wordlists[p.WordlistArtifact]
			if p.WordlistArtifact == "" || !ok || ref.ValidateExact() != nil {
				return failure("missing_scan_wordlist")
			}
		} else if p.WordlistArtifact != "" {
			return failure("unexpected_scan_wordlist")
		}
	}
	return nil
}

func parsePlanSources(input PlanInput, plan *ScanPlan) ([]planSourceInput, error) {
	result := []planSourceInput{}
	switch input.MediaType {
	case contracts.HTTPRequestSetMediaType:
		set, err := contracts.DecodeHTTPRequestSet(input.Data)
		if err != nil {
			return nil, failure("invalid_request_set")
		}
		plan.PreparationGaps = set.Gaps
		plan.PreparationCoverage = &set.Coverage
		for _, entry := range set.Requests {
			request := entry.Request
			result = append(result, planSourceInput{id: entry.ID, target: request.URL, request: &request})
		}
	case TargetListMediaType:
		if len(input.Data) > MaxSourceBytes || !utf8.Valid(input.Data) || bytes.HasPrefix(input.Data, []byte{0xef, 0xbb, 0xbf}) {
			return nil, failure("invalid_target_list")
		}
		text := strings.ReplaceAll(string(input.Data), "\r\n", "\n")
		if strings.Contains(text, "\r") {
			return nil, failure("invalid_target_list")
		}
		for i, target := range strings.Split(text, "\n") {
			if target == "" {
				continue
			}
			if len(result) >= 1000 {
				return nil, failure("target_list_limit_exceeded")
			}
			result = append(result, planSourceInput{id: fmt.Sprintf("line-%04d", i+1), target: target})
		}
		if len(result) == 0 {
			return nil, failure("empty_target_list")
		}
	default:
		return nil, failure("unsupported_scan_input_media_type")
	}
	return result, nil
}

func planJob(source planSourceInput, p contracts.ScanToolPolicy, b ToolBinding, wordlists map[string]contracts.ArtifactRef) (ScanJob, string) {
	e := *b.Template.Execution
	job := ScanJob{Worker: p.Worker, Tool: e.Tool, Namespace: b.Namespace, TemplateRef: b.Template.Ref.TemplateID + "@" + b.Template.Ref.Version, TemplateDigest: b.Template.Ref.Digest, Execution: e, TimeoutSeconds: e.TimeoutSeconds, Parameters: map[string]string{}, Artifacts: map[string]contracts.ArtifactRef{}}
	switch e.Tool {
	case "scan_naabu":
		host := source.target
		if strings.Contains(host, "://") {
			u, code := scanURL(host)
			if code != "" {
				return job, code
			}
			host = u.Hostname()
		}
		if !scanHost(host) {
			return job, "unsupported_host"
		}
		job.Parameters[e.Arguments["host"].Name] = strings.ToLower(host)
	case "scan_nuclei", "scan_ffuf":
		u, code := scanURL(source.target)
		if code != "" {
			return job, code
		}
		if source.request != nil && (source.request.Method != "GET" || source.request.Body != "" || len(source.request.Headers) > 0) {
			return job, "request_not_target_only"
		}
		if e.Tool == "scan_ffuf" {
			if strings.Contains(source.target, "FFUFHASH") || strings.Contains(u.Host, "FUZZ") || !strings.Contains(u.EscapedPath()+u.RawQuery, "FUZZ") {
				return job, "missing_fuzz_marker"
			}
			job.Artifacts[e.Arguments["wordlist_ref"].Name] = wordlists[p.WordlistArtifact]
		}
		job.Parameters[e.Arguments["url"].Name] = source.target
	case "scan_sqlmap":
		if source.request == nil {
			return job, "request_set_required"
		}
		request, code := prepareSQLMapRequest(*source.request, p.TestParameters)
		if code != "" {
			return job, code
		}
		job.Request = &request
	}
	return job, ""
}

func semanticID(prefix string, value any) string {
	raw, err := contracts.MarshalPrivateCanonical(value)
	if err != nil {
		return ""
	}
	return prefix + "-" + strings.TrimPrefix(digest(raw), "sha256:")
}
func planIdentity(plan ScanPlan) string {
	plan.ID = ""
	plan.Jobs = append([]ScanJob(nil), plan.Jobs...)
	if plan.Jobs == nil {
		plan.Jobs = []ScanJob{}
	}
	for i := range plan.Jobs {
		plan.Jobs[i].ID = ""
	}
	return semanticID("plan", plan)
}
func jobIdentity(planID, candidateID string) string {
	return semanticID("job", struct {
		PlanID      string `json:"planId"`
		CandidateID string `json:"candidateId"`
	}{planID, candidateID})
}
