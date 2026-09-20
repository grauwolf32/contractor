package scanplan_test

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/scanplan"
	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
)

func planFixture(tools ...string) (contracts.ScanPlanPolicy, map[string]scanplan.ToolBinding) {
	policy := contracts.ScanPlanPolicy{InputArtifact: "input", MaxInputs: 1000, MaxJobs: 100, MaxTotalSeconds: 3600}
	bindings := map[string]scanplan.ToolBinding{}
	for _, tool := range tools {
		worker := strings.TrimPrefix(tool, "scan_")
		arguments := map[string]contracts.ToolArgumentBinding{}
		switch tool {
		case "scan_naabu":
			arguments["host"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
		case "scan_sqlmap":
			arguments["request_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "request"}
		default:
			arguments["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "target"}
		}
		toolPolicy := contracts.ScanToolPolicy{Worker: worker, MaxJobs: 100, MaxTotalSeconds: 3600}
		if tool == "scan_sqlmap" {
			toolPolicy.TestParameters = []string{"id"}
		}
		if tool == "scan_ffuf" {
			toolPolicy.WordlistArtifact = "words"
			arguments["wordlist_ref"] = contracts.ToolArgumentBinding{Source: "artifact", Name: "wordlist"}
		}
		policy.Tools = append(policy.Tools, toolPolicy)
		bindings[worker] = scanplan.ToolBinding{Namespace: worker, Template: contracts.ResolvedAgentTemplate{
			Ref:         contracts.AgentTemplateRef{TemplateID: worker + "-scan", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
			Description: "Fixed local test Worker", Runtime: contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "1"},
			SandboxProfile: contracts.SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"},
			Toolsets:       []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}, Tools: []string{tool}}},
			Execution:      &contracts.ToolExecutionConfig{Tool: tool, Arguments: arguments, ResultArtifact: "report", TimeoutSeconds: 10},
		}}
	}
	return policy, bindings
}

func planInput(targets string) scanplan.PlanInput {
	revision := "targets-revision-1"
	return scanplan.PlanInput{Artifact: contracts.ArtifactRef{Namespace: "inputs", Name: "targets", Revision: &revision}, MediaType: scanplan.TargetListMediaType, Data: []byte(targets)}
}

func buildPlan(t *testing.T, input scanplan.PlanInput, policy contracts.ScanPlanPolicy, bindings map[string]scanplan.ToolBinding, wordlists map[string]contracts.ArtifactRef) scanplan.ScanPlan {
	t.Helper()
	plan, err := scanplan.BuildPlan(input, policy, bindings, wordlists)
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func marshalPlan(t *testing.T, plan scanplan.ScanPlan) []byte {
	t.Helper()
	data, err := scanplan.MarshalPlan(plan)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestBuildPlanDeterminismAndDeduplicationPreserveEverySource(t *testing.T) {
	policy, bindings := planFixture("scan_nuclei", "scan_naabu")
	input := planInput("https://example.test/a\nhttps://example.test/a\n\nhttps://example.test/b\nhttps://other.test/a\n")
	plan := buildPlan(t, input, policy, bindings, nil)
	if len(plan.Jobs) != 5 || len(plan.Candidates) != 5 {
		t.Fatalf("expected three distinct URL and two host jobs, got %d jobs / %d candidates", len(plan.Jobs), len(plan.Candidates))
	}
	var mergedNuclei, mergedNaabu bool
	previous := ""
	for _, candidate := range plan.Candidates {
		if candidate.ID <= previous || !sort.StringsAreSorted(candidate.SourceIDs) || candidate.Selection != "selected" || candidate.Code != "" {
			t.Fatalf("candidate is not canonical: %#v", candidate)
		}
		previous = candidate.ID
		if candidate.Tool == "scan_nuclei" && reflect.DeepEqual(candidate.SourceIDs, []string{"line-0001", "line-0002"}) {
			mergedNuclei = true
		}
		if candidate.Tool == "scan_naabu" && reflect.DeepEqual(candidate.SourceIDs, []string{"line-0001", "line-0002", "line-0004"}) {
			mergedNaabu = true
		}
	}
	if !mergedNuclei || !mergedNaabu {
		t.Fatal("deduplication lost source line provenance")
	}
	first := marshalPlan(t, plan)
	if policy.Tools[0].Worker != "nuclei" {
		t.Fatal("builder sorted caller-owned policy")
	}
	policy.Tools[0], policy.Tools[1] = policy.Tools[1], policy.Tools[0]
	for i := 0; i < 20; i++ {
		if next := marshalPlan(t, buildPlan(t, input, policy, bindings, nil)); !bytes.Equal(first, next) {
			t.Fatal("same exact source and normalized policy produced different plan bytes")
		}
	}
	for _, job := range plan.Jobs {
		if !strings.HasPrefix(job.ID, "job-") || job.CandidateID == "" || job.TemplateDigest == "" || len(job.Artifacts) != 0 {
			t.Fatalf("job must bind immutable semantics without generated execution refs: %#v", job)
		}
	}
}

func TestBuildPlanRejectsInvalidBindingsEvenWhenAllInputsAreSkipped(t *testing.T) {
	for _, name := range []string{"dynamic slot", "result slot", "extra worker"} {
		t.Run(name, func(t *testing.T) {
			policy, bindings := planFixture("scan_nuclei")
			switch name {
			case "dynamic slot":
				bindings["nuclei"].Template.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "other"}
			case "result slot":
				bindings["nuclei"].Template.Execution.ResultArtifact = "other"
			case "extra worker":
				bindings["unused"] = bindings["nuclei"]
			}
			if _, err := scanplan.BuildPlan(planInput("not-an-http-target\n"), policy, bindings, nil); err == nil {
				t.Fatal("invalid fixed Worker configuration survived an all-skipped plan")
			}
		})
	}
}

func TestBuildPlanBudgetsAccountForAllCandidates(t *testing.T) {
	cases := []struct {
		name     string
		change   func(*contracts.ScanPlanPolicy)
		selected int
		code     string
	}{
		{"inputs", func(p *contracts.ScanPlanPolicy) { p.MaxInputs = 1 }, 1, "input_budget_exceeded"},
		{"global jobs", func(p *contracts.ScanPlanPolicy) { p.MaxJobs = 2 }, 2, "job_budget_exceeded"},
		{"tool jobs", func(p *contracts.ScanPlanPolicy) { p.Tools[0].MaxJobs = 1 }, 1, "tool_job_budget_exceeded"},
		{"global seconds", func(p *contracts.ScanPlanPolicy) { p.MaxTotalSeconds = 15 }, 1, "time_budget_exceeded"},
		{"tool seconds", func(p *contracts.ScanPlanPolicy) { p.Tools[0].MaxTotalSeconds = 15 }, 1, "tool_time_budget_exceeded"},
		{"timeout does not fit", func(p *contracts.ScanPlanPolicy) { p.MaxTotalSeconds = 9 }, 0, "time_budget_exceeded"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			policy, bindings := planFixture("scan_nuclei")
			tc.change(&policy)
			plan := buildPlan(t, planInput("https://one.test/\nhttps://two.test/\nhttps://three.test/\nhttps://four.test/\n"), policy, bindings, nil)
			if len(plan.Candidates) != 4 || len(plan.Jobs) != tc.selected {
				t.Fatalf("incorrect accounting: candidates %d jobs %d", len(plan.Candidates), len(plan.Jobs))
			}
			skipped := 0
			for _, candidate := range plan.Candidates {
				if candidate.Selection == "skipped" {
					skipped++
					if candidate.Code != tc.code {
						t.Fatalf("wrong skip reason: %s", candidate.Code)
					}
				}
			}
			if skipped != 4-tc.selected {
				t.Fatal("not all rejected candidates have an explicit reason")
			}
		})
	}
}

func requestSetPlanInput(t *testing.T, request contracts.PreparedHTTPRequest) scanplan.PlanInput {
	t.Helper()
	contentDigest, err := contracts.RequestContentDigest(request)
	if err != nil {
		t.Fatal(err)
	}
	set := contracts.HTTPRequestSet{
		SchemaVersion:     1,
		Source:            contracts.RequestSetSource{Artifact: sourceRef(), ContentDigest: "sha256:" + strings.Repeat("b", 64)},
		PreparationDigest: "sha256:" + strings.Repeat("c", 64),
		Requests:          []contracts.RequestSetEntry{{ID: "request-" + strings.TrimPrefix(contentDigest, "sha256:"), ContentDigest: contentDigest, Request: request, Origins: []contracts.RequestOrigin{{Pointer: "#/paths/~1items/post"}, {Pointer: "#/paths/~1other/post"}}}},
		Gaps:              []contracts.PreparationGap{{Pointer: "#/paths/~1missing/get", Code: "missing_parameter_value"}},
		Coverage:          contracts.RequestSetCoverage{Operations: 3, Prepared: 2, Skipped: 1, Complete: false},
	}
	data, err := contracts.MarshalHTTPRequestSet(set)
	if err != nil {
		t.Fatal(err)
	}
	input := planInput("")
	input.MediaType, input.Data = contracts.HTTPRequestSetMediaType, data
	return input
}

func TestBuildPlanRequestSetUsesExplicitSQLMapParametersAndRetainsPreparationGaps(t *testing.T) {
	policy, bindings := planFixture("scan_sqlmap", "scan_nuclei", "scan_naabu")
	request := contracts.PreparedHTTPRequest{Method: "POST", URL: "https://api.example.test/items", Headers: []contracts.HTTPRequestHeader{{Name: "content-type", Value: "application/json"}}, Body: `{"id":7}`}
	input := requestSetPlanInput(t, request)
	plan := buildPlan(t, input, policy, bindings, nil)
	if len(plan.Jobs) != 2 || len(plan.Candidates) != 3 || plan.PreparationCoverage == nil || plan.PreparationCoverage.Operations != 3 || len(plan.PreparationGaps) != 1 {
		t.Fatalf("missing prepared-input coverage or tool eligibility accounting: %#v", plan)
	}
	var sqlmap *scanplan.ScanJob
	for i := range plan.Jobs {
		if plan.Jobs[i].Tool == "scan_sqlmap" {
			sqlmap = &plan.Jobs[i]
		}
	}
	if sqlmap == nil || sqlmap.Request == nil || sqlmap.Request.Body != request.Body || sqlmap.Request.URL != request.URL || !reflect.DeepEqual(sqlmap.Request.Headers, request.Headers) || !reflect.DeepEqual(sqlmap.Request.TestParameters, []string{"id"}) || len(sqlmap.Artifacts) != 0 {
		t.Fatalf("SQLMap input was not retained or generated refs leaked into the plan: %#v", sqlmap)
	}
	for _, candidate := range plan.Candidates {
		if len(candidate.SourceIDs) != 1 || !strings.HasPrefix(candidate.SourceIDs[0], "request-") {
			t.Fatal("RequestSet provenance was lost")
		}
		if candidate.Tool == "scan_nuclei" && (candidate.Selection != "skipped" || candidate.Code != "request_not_target_only") {
			t.Fatal("target-only tool silently discarded request body")
		}
	}
	withoutParameters, withoutBindings := planFixture("scan_sqlmap")
	withoutParameters.Tools[0].TestParameters = nil
	if _, err := scanplan.BuildPlan(input, withoutParameters, withoutBindings, nil); err == nil {
		t.Fatal("SQLMap guessed test parameters")
	}
}

func TestBuildPlanTargetOnlyToolsAndFFUFWordlistBindings(t *testing.T) {
	policy, bindings := planFixture("scan_ffuf", "scan_sqlmap")
	revision := "wordlist-revision-7"
	wordlists := map[string]contracts.ArtifactRef{"words": {Namespace: "inputs", Name: "wordlist", Revision: &revision}}
	plan := buildPlan(t, planInput("https://example.test/FUZZ\nhttps://example.test/ordinary\n"), policy, bindings, wordlists)
	if len(plan.Jobs) != 1 || plan.Jobs[0].Tool != "scan_ffuf" || !reflect.DeepEqual(plan.Jobs[0].Artifacts["wordlist"], wordlists["words"]) {
		t.Fatalf("FFUF did not retain exact wordlist binding: %#v", plan.Jobs)
	}
	codes := map[string]int{}
	for _, candidate := range plan.Candidates {
		codes[candidate.Code]++
	}
	if codes["missing_fuzz_marker"] != 1 || codes["request_set_required"] != 2 {
		t.Fatalf("unexpected tool eligibility accounting: %#v", codes)
	}
	if _, err := scanplan.BuildPlan(planInput("https://example.test/FUZZ\n"), policy, bindings, nil); err == nil {
		t.Fatal("FFUF accepted absent wordlist artifact")
	}
}

func TestBuildPlanDetachesInputsAndIdentityTracksSemanticChanges(t *testing.T) {
	policy, bindings := planFixture("scan_ffuf")
	revision := "wordlist-revision-1"
	wordlists := map[string]contracts.ArtifactRef{"words": {Namespace: "inputs", Name: "words", Revision: &revision}}
	input := planInput("https://example.test/FUZZ\n")
	plan := buildPlan(t, input, policy, bindings, wordlists)
	before := marshalPlan(t, plan)
	*input.Artifact.Revision = "source-revision-changed"
	revision = "wordlist-revision-changed"
	bindings["ffuf"].Template.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "changed"}
	policy.Tools[0].MaxJobs = 1
	input.Data[0] = 'X'
	if after := marshalPlan(t, plan); !bytes.Equal(before, after) {
		t.Fatal("returned plan aliases caller-owned inputs")
	}

	for _, change := range []string{"source bytes", "source revision", "policy", "template digest", "wordlist revision"} {
		t.Run(change, func(t *testing.T) {
			p, b := planFixture("scan_ffuf")
			in := planInput("https://example.test/FUZZ\n")
			rev := "wordlist-revision-1"
			w := map[string]contracts.ArtifactRef{"words": {Namespace: "inputs", Name: "words", Revision: &rev}}
			switch change {
			case "source bytes":
				in.Data = []byte("https://example.test/FUZZ?x=1\n")
			case "source revision":
				*in.Artifact.Revision = "targets-revision-2"
			case "policy":
				p.MaxInputs--
			case "template digest":
				binding := b["ffuf"]
				binding.Template.Ref.Digest = "sha256:" + strings.Repeat("d", 64)
				b["ffuf"] = binding
			case "wordlist revision":
				rev = "wordlist-revision-2"
			}
			changed := buildPlan(t, in, p, b, w)
			if changed.ID == plan.ID || changed.Jobs[0].ID == plan.Jobs[0].ID {
				t.Fatal("semantic change did not invalidate stable plan/job identity")
			}
		})
	}
}

func TestBuildPlanRejectsMalformedOrUnboundedWholeInput(t *testing.T) {
	for name, change := range map[string]func(*scanplan.PlanInput, *contracts.ScanPlanPolicy, map[string]scanplan.ToolBinding){
		"non-exact source": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Artifact.Revision = nil
		},
		"unknown media type": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.MediaType = "text/plain"
		},
		"invalid UTF8": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = []byte{0xff}
		},
		"BOM": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = append([]byte{0xef, 0xbb, 0xbf}, in.Data...)
		},
		"bare CR": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = []byte("https://example.test/\r")
		},
		"empty list": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = []byte("\n\n")
		},
		"excess source bytes": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = bytes.Repeat([]byte("x"), scanplan.MaxSourceBytes+1)
		},
		"excess source lines": func(in *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			in.Data = bytes.Repeat([]byte("https://example.test/\n"), 1001)
		},
		"zero input budget": func(_ *scanplan.PlanInput, p *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			p.MaxInputs = 0
		},
		"unbounded jobs": func(_ *scanplan.PlanInput, p *contracts.ScanPlanPolicy, _ map[string]scanplan.ToolBinding) {
			p.MaxJobs = 101
		},
		"missing fixed Worker": func(_ *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, b map[string]scanplan.ToolBinding) {
			delete(b, "nuclei")
		},
		"literal target": func(_ *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, b map[string]scanplan.ToolBinding) {
			b["nuclei"].Template.Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "literal", Value: "https://unplanned.test/"}
		},
		"extra dynamic argument": func(_ *scanplan.PlanInput, _ *contracts.ScanPlanPolicy, b map[string]scanplan.ToolBinding) {
			b["nuclei"].Template.Execution.Arguments["templates"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "unplanned"}
		},
	} {
		t.Run(name, func(t *testing.T) {
			input := planInput("https://example.test/\n")
			policy, bindings := planFixture("scan_nuclei")
			change(&input, &policy, bindings)
			if _, err := scanplan.BuildPlan(input, policy, bindings, nil); err == nil {
				t.Fatal("accepted malformed or unbounded plan input")
			}
		})
	}
}

func TestScanPlanStrictCodecAndSchema(t *testing.T) {
	policy, bindings := planFixture("scan_nuclei")
	plan := buildPlan(t, planInput("https://example.test/\n"), policy, bindings, nil)
	canonical := marshalPlan(t, plan)
	decoded, err := scanplan.DecodePlan(canonical)
	if err != nil || !reflect.DeepEqual(decoded, plan) {
		t.Fatalf("canonical round trip differs: %v", err)
	}
	path, err := filepath.Abs("../../api/scan/v1/scan-plan.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := jsonschema.NewCompiler().Compile(path)
	if err != nil {
		t.Fatal(err)
	}
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(canonical))
	if err != nil || schema.Validate(value) != nil {
		t.Fatalf("normative schema rejected canonical plan: %v", err)
	}
	mutations := map[string]func(map[string]any){
		"missing candidates":      func(v map[string]any) { delete(v, "candidates") },
		"null jobs":               func(v map[string]any) { v["jobs"] = nil },
		"unknown root field":      func(v map[string]any) { v["allocationId"] = "physical-worker" },
		"unknown job field":       func(v map[string]any) { v["jobs"].([]any)[0].(map[string]any)["allocationId"] = "physical-worker" },
		"missing source revision": func(v map[string]any) { delete(v["source"].(map[string]any)["artifact"].(map[string]any), "revision") },
		"forged plan ID":          func(v map[string]any) { v["id"] = "plan-" + strings.Repeat("0", 64) },
		"forged job ID":           func(v map[string]any) { v["jobs"].([]any)[0].(map[string]any)["id"] = "job-" + strings.Repeat("0", 64) },
		"changed job target": func(v map[string]any) {
			v["jobs"].([]any)[0].(map[string]any)["parameters"].(map[string]any)["target"] = "https://different.test/"
		},
	}
	for name, change := range mutations {
		t.Run(name, func(t *testing.T) {
			var v map[string]any
			if err := json.Unmarshal(canonical, &v); err != nil {
				t.Fatal(err)
			}
			change(v)
			data, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := scanplan.DecodePlan(data); err == nil {
				t.Fatal("strict codec accepted malformed or forged plan")
			}
		})
	}
	for name, data := range map[string][]byte{
		"duplicate key":     bytes.Replace(canonical, []byte(`"schemaVersion":1`), []byte(`"schemaVersion":1,"schemaVersion":1`), 1),
		"trailing document": append(append([]byte(nil), canonical...), []byte(` {}`)...),
		"excess bytes":      bytes.Repeat([]byte(" "), scanplan.MaxPlanBytes+1),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := scanplan.DecodePlan(data); err == nil {
				t.Fatal("strict codec accepted malformed input bytes")
			}
		})
	}
}

// Recompute all public identities independently so semantic rejections cannot
// pass merely because an earlier content hash stopped matching changed bytes.
func reidentifyPlan(t *testing.T, plan scanplan.ScanPlan) []byte {
	t.Helper()
	for i := range plan.Jobs {
		job := &plan.Jobs[i]
		oldCandidate := job.CandidateID
		semantic := *job
		semantic.ID, semantic.CandidateID = "", ""
		raw, err := contracts.MarshalPrivateCanonical(semantic)
		if err != nil {
			t.Fatal(err)
		}
		job.CandidateID = "candidate-" + strings.TrimPrefix(digest(raw), "sha256:")
		for j := range plan.Candidates {
			if plan.Candidates[j].ID == oldCandidate {
				plan.Candidates[j].ID = job.CandidateID
			}
		}
		job.ID = ""
	}
	sort.Slice(plan.Candidates, func(i, j int) bool { return plan.Candidates[i].ID < plan.Candidates[j].ID })
	sort.Slice(plan.Jobs, func(i, j int) bool { return plan.Jobs[i].CandidateID < plan.Jobs[j].CandidateID })
	plan.ID = ""
	raw, err := contracts.MarshalPrivateCanonical(plan)
	if err != nil {
		t.Fatal(err)
	}
	plan.ID = "plan-" + strings.TrimPrefix(digest(raw), "sha256:")
	for i := range plan.Jobs {
		raw, err = contracts.MarshalPrivateCanonical(map[string]string{"planId": plan.ID, "candidateId": plan.Jobs[i].CandidateID})
		if err != nil {
			t.Fatal(err)
		}
		plan.Jobs[i].ID = "job-" + strings.TrimPrefix(digest(raw), "sha256:")
	}
	raw, err = contracts.MarshalPrivateCanonical(plan)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestScanPlanCodecValidatesSemanticsBeyondHashes(t *testing.T) {
	for name, change := range map[string]func(*scanplan.ScanPlan){
		"target list has preparation coverage": func(p *scanplan.ScanPlan) {
			p.PreparationCoverage = &contracts.RequestSetCoverage{Operations: 1, Prepared: 1, Complete: true}
		},
		"target list has preparation gaps": func(p *scanplan.ScanPlan) {
			p.PreparationGaps = []contracts.PreparationGap{{Pointer: "#", Code: "warning"}}
		},
		"invalid source ID":                   func(p *scanplan.ScanPlan) { p.Candidates[0].SourceIDs[0] = "untracked-source" },
		"zero source line":                    func(p *scanplan.ScanPlan) { p.Candidates[0].SourceIDs[0] = "line-0000" },
		"unexpected request provenance":       func(p *scanplan.ScanPlan) { p.Candidates[0].SourceIDs[0] = "request-" + strings.Repeat("a", 64) },
		"source represented twice for Worker": func(p *scanplan.ScanPlan) { p.Candidates[1].SourceIDs[0] = p.Candidates[0].SourceIDs[0] },
		"selected input budget exceeded":      func(p *scanplan.ScanPlan) { p.Policy.MaxInputs = 1 },
		"non-SQLMap test parameters":          func(p *scanplan.ScanPlan) { p.Policy.Tools[0].TestParameters = []string{"id"} },
		"invalid template selector":           func(p *scanplan.ScanPlan) { p.Jobs[0].TemplateRef = "bad template@1" },
		"wrong dynamic slot": func(p *scanplan.ScanPlan) {
			p.Jobs[0].Execution.Arguments["url"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "elsewhere"}
			p.Jobs[0].Parameters = map[string]string{"elsewhere": "https://example.test/"}
		},
		"unexpected target-only artifact": func(p *scanplan.ScanPlan) { p.Jobs[0].Artifacts["unplanned"] = sourceRef() },
		"extra dynamic argument": func(p *scanplan.ScanPlan) {
			p.Jobs[0].Execution.Arguments["templates"] = contracts.ToolArgumentBinding{Source: "parameter", Name: "unplanned"}
		},
	} {
		t.Run(name, func(t *testing.T) {
			policy, bindings := planFixture("scan_nuclei")
			plan := buildPlan(t, planInput("https://example.test/\nhttps://other.test/\n"), policy, bindings, nil)
			change(&plan)
			if _, err := scanplan.DecodePlan(reidentifyPlan(t, plan)); err == nil {
				t.Fatal("codec accepted rehashed but semantically invalid plan")
			}
		})
	}
	for name, change := range map[string]func(*scanplan.ScanPlan){
		"missing preparation coverage":    func(p *scanplan.ScanPlan) { p.PreparationCoverage = nil },
		"inconsistent preparation counts": func(p *scanplan.ScanPlan) { p.PreparationCoverage.Prepared = 3 },
		"false complete coverage":         func(p *scanplan.ScanPlan) { p.PreparationCoverage.Complete = true },
		"negative preparation count":      func(p *scanplan.ScanPlan) { p.PreparationCoverage.Prepared = -1; p.PreparationCoverage.Skipped = 4 },
		"invalid gap pointer":             func(p *scanplan.ScanPlan) { p.PreparationGaps[0].Pointer = "not-a-pointer" },
		"invalid gap code":                func(p *scanplan.ScanPlan) { p.PreparationGaps[0].Code = "private message with data" },
		"duplicate preparation gap":       func(p *scanplan.ScanPlan) { p.PreparationGaps = append(p.PreparationGaps, p.PreparationGaps[0]) },
	} {
		t.Run(name, func(t *testing.T) {
			policy, bindings := planFixture("scan_sqlmap")
			plan := buildPlan(t, requestSetPlanInput(t, contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/?id=7", Headers: []contracts.HTTPRequestHeader{}}), policy, bindings, nil)
			change(&plan)
			if _, err := scanplan.DecodePlan(reidentifyPlan(t, plan)); err == nil {
				t.Fatal("codec accepted invalid preparation provenance despite matching hashes")
			}
		})
	}
	for _, name := range []string{"missing wordlist", "missing FUZZ marker"} {
		t.Run(name, func(t *testing.T) {
			policy, bindings := planFixture("scan_ffuf")
			plan := buildPlan(t, planInput("https://example.test/FUZZ\n"), policy, bindings, map[string]contracts.ArtifactRef{"words": sourceRef()})
			if name == "missing wordlist" {
				delete(plan.Jobs[0].Artifacts, "wordlist")
			} else {
				plan.Jobs[0].Parameters["target"] = "https://example.test/no-marker"
			}
			if _, err := scanplan.DecodePlan(reidentifyPlan(t, plan)); err == nil {
				t.Fatal("codec accepted an unexecutable rehashed FFUF job")
			}
		})
	}
}

func TestBuildPlanSQLMapParameterLocationsAndRepresentation(t *testing.T) {
	cases := []struct {
		name      string
		request   contracts.PreparedHTTPRequest
		parameter string
		code      string
	}{
		{"query", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/?id=7"}, "id", ""},
		{"form", contracts.PreparedHTTPRequest{Method: "POST", URL: "https://example.test/", Headers: []contracts.HTTPRequestHeader{{Name: "content-type", Value: "application/x-www-form-urlencoded"}}, Body: "id=7"}, "id", ""},
		{"JSON", contracts.PreparedHTTPRequest{Method: "PATCH", URL: "https://example.test/", Headers: []contracts.HTTPRequestHeader{{Name: "content-type", Value: "application/json"}}, Body: `{"id":7}`}, "id", ""},
		{"cookie", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/", Headers: []contracts.HTTPRequestHeader{{Name: "cookie", Value: "id=7"}}}, "id", ""},
		{"header", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/", Headers: []contracts.HTTPRequestHeader{{Name: "x-item-id", Value: "7"}}}, "X-Item-ID", ""},
		{"parameter absent", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/?other=7"}, "id", "test_parameter_unavailable"},
		{"request-file marker", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/?id=*"}, "id", "unsupported_request_representation"},
		{"unsupported framing", contracts.PreparedHTTPRequest{Method: "GET", URL: "https://example.test/?id=7", Headers: []contracts.HTTPRequestHeader{{Name: "connection", Value: "keep-alive"}}}, "id", "unsupported_request_representation"},
		{"body trailing newline", contracts.PreparedHTTPRequest{Method: "POST", URL: "https://example.test/?id=7", Body: "body\n"}, "id", "unsupported_request_representation"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.request.Headers == nil {
				tc.request.Headers = []contracts.HTTPRequestHeader{}
			}
			policy, bindings := planFixture("scan_sqlmap")
			policy.Tools[0].TestParameters = []string{tc.parameter}
			plan := buildPlan(t, requestSetPlanInput(t, tc.request), policy, bindings, nil)
			if len(plan.Candidates) != 1 || plan.Candidates[0].Code != tc.code {
				t.Fatalf("unexpected selection: %#v", plan.Candidates)
			}
			if (len(plan.Jobs) == 1) != (tc.code == "") {
				t.Fatal("unsupported representation was selected or valid parameter was skipped")
			}
		})
	}
}

func TestScanPlanSchemaAcceptsEverySupportedToolAndMethod(t *testing.T) {
	path, err := filepath.Abs("../../api/scan/v1/scan-plan.schema.json")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := jsonschema.NewCompiler().Compile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, tool := range []string{"scan_nuclei", "scan_naabu", "scan_ffuf", "scan_sqlmap"} {
		methods := []string{"GET"}
		if tool == "scan_sqlmap" {
			methods = []string{"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
		}
		for _, method := range methods {
			t.Run(tool+"/"+method, func(t *testing.T) {
				policy, bindings := planFixture(tool)
				input := planInput("https://example.test/FUZZ?id=7\n")
				var wordlists map[string]contracts.ArtifactRef
				if tool == "scan_ffuf" {
					wordlists = map[string]contracts.ArtifactRef{"words": sourceRef()}
				}
				if tool == "scan_sqlmap" {
					input = requestSetPlanInput(t, contracts.PreparedHTTPRequest{Method: method, URL: "https://example.test/?id=7", Headers: []contracts.HTTPRequestHeader{}})
				}
				plan := buildPlan(t, input, policy, bindings, wordlists)
				if len(plan.Jobs) != 1 {
					t.Fatal("supported input did not produce one job")
				}
				value, err := jsonschema.UnmarshalJSON(bytes.NewReader(marshalPlan(t, plan)))
				if err != nil {
					t.Fatal(err)
				}
				if err := schema.Validate(value); err != nil {
					t.Fatalf("schema rejected supported scanner plan: %v", err)
				}
			})
		}
	}
}

func TestBuildPlanPreservesLargeOriginalLineNumbersAndCRLF(t *testing.T) {
	policy, bindings := planFixture("scan_naabu")
	plan := buildPlan(t, planInput(strings.Repeat("\r\n", 1234)+"EXAMPLE.TEST\r\n"), policy, bindings, nil)
	if len(plan.Jobs) != 1 || plan.Jobs[0].Parameters["target"] != "example.test" || !reflect.DeepEqual(plan.Candidates[0].SourceIDs, []string{"line-1235"}) {
		t.Fatalf("blank line accounting or CRLF host input changed: %#v", plan)
	}
}
