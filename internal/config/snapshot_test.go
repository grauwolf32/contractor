package config

import (
	"cmp"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configtest"
)

// detachedOverrideProfile selects every override field, including a cleared
// credential, so the profile clone path carries a Gateway body and both
// credential shapes.
const detachedOverrideProfile = `apiVersion: contractor/v1alpha1
kind: ExecutionConfig
metadata: {name: detached-override, version: "1"}
spec:
  planner:
    llmGateway: local-litellm@1
    credential: development-planner
  agents:
    builder:
      modelPolicy: worker@1
      llmGateway: local-litellm@1
      credential: development-worker
    validator:
      credential: null
`

// TestSnapshotAccessorResultsShareNoMemoryWithSnapshot checks the Snapshot
// contract mechanically: every pointer, slice and map in an accessor result
// must be an allocation the Snapshot does not hold, so no caller mutation, at
// any depth, reaches the published configuration. The catalog declares Gateway
// failure signatures, Planner and Worker selections with credentials, an inline
// escalation override, an ExecutionConfig profile with every field, scan plans
// and AuditProfiles. A new exported accessor fails the test until it is
// exercised here.
func TestSnapshotAccessorResultsShareNoMemoryWithSnapshot(t *testing.T) {
	t.Parallel()

	root := configtest.CopyWithEscalation(t, coreFixtureRoot)
	for _, topic := range []string{
		"audit-profile-catalog", "audit-completion-catalog", "audit-scan-catalog", "workflow-examples",
	} {
		if err := os.CopyFS(root, os.DirFS(filepath.Join("testdata", topic))); err != nil {
			t.Fatal(err)
		}
	}
	writeFile(t, filepath.Join(root, "llm-gateways/local_litellm.yaml"), []byte(signedGatewayManifest))
	writeFile(t, filepath.Join(root, "execution-configs/detached_override.yaml"), []byte(detachedOverrideProfile))
	replaceFile(t, filepath.Join(root, "workflows/openapi_from_workspace_v5.yaml"), "ref: strong-oas-review@1",
		"agents: {validator: {llmGateway: local-litellm@1, credential: development-worker}}")
	snapshot := mustLoad(t, root)
	owned := newSpanIndex(mutableSpans(reflect.ValueOf(snapshot).Elem(), "Snapshot"))

	type accessorResult struct {
		call  string
		value any
	}
	var results []accessorResult
	covered := make(map[string]bool)
	record := func(method, argument string, value any, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("Snapshot.%s(%s): %v", method, argument, err)
		}
		covered[method] = true
		results = append(results, accessorResult{call: method + "(" + argument + ")", value: value})
	}

	record("Counts", "", snapshot.Counts(), nil)
	selectors := make(map[ConfigurationKind][]string)
	for _, kind := range []ConfigurationKind{
		ConfigurationAgentTemplates, ConfigurationExecutionConfigs,
		ConfigurationModelPolicies, ConfigurationLLMGateways,
	} {
		discovery, err := snapshot.ConfigurationDiscovery(kind)
		record("ConfigurationDiscovery", string(kind), discovery, err)
		for _, resource := range discovery.Resources {
			selector := resource.Ref.Name + "@" + resource.Ref.Version
			selectors[kind] = append(selectors[kind], selector)
			value, err := snapshot.Configuration(kind, selector)
			record("Configuration", string(kind)+", "+selector, value, err)
		}
	}
	instructionRefs := make(map[string]struct{})
	for _, selector := range selectors[ConfigurationAgentTemplates] {
		template, err := snapshot.AgentTemplate(selector)
		record("AgentTemplate", selector, template, err)
		if !template.IsToolWorker() {
			instructionRefs[template.Instructions.Ref] = struct{}{}
			instructions, err := snapshot.AgentInstructions(selector)
			record("AgentInstructions", selector, instructions, err)
		}
		// The index keeps its sorted bindings unexported; callers read them
		// only through Page, which must return a detached page.
		bindings, err := snapshot.AgentTemplateWorkflowBindings(selector)
		record("AgentTemplateWorkflowBindings", selector, bindings.Page(nil, math.MaxInt), err)
	}
	for _, selector := range selectors[ConfigurationModelPolicies] {
		policy, err := snapshot.ModelPolicy(selector)
		record("ModelPolicy", selector, policy, err)
	}
	for _, selector := range selectors[ConfigurationLLMGateways] {
		gateway, err := snapshot.LLMGateway(selector)
		record("LLMGateway", selector, gateway, err)
	}
	// The Workflow discovery view keeps its sorted Workflows unexported;
	// callers read them only through Page, which must return detached copies.
	workflowDiscovery, err := snapshot.WorkflowDiscovery()
	record("WorkflowDiscovery", "", workflowDiscovery.Page("", math.MaxInt,
		func(resolved.WorkflowRef, resolved.WorkflowPresentation) bool { return true }), err)
	credentials := developmentCredentialLookup(t, snapshot)
	workflows := snapshot.Workflows()
	record("Workflows", "", workflows, nil)
	for _, listed := range workflows {
		selector := listed.Ref.Name + "@" + listed.Ref.Version
		workflow, err := snapshot.Workflow(selector)
		record("Workflow", selector, workflow, err)
		runWorkflow, err := snapshot.ResolveRunWorkflow(t.Context(), selector, resolved.ExecutionConfigPatch{}, credentials)
		record("ResolveRunWorkflow", selector, runWorkflow, err)
		pinnedWorkflow, err := snapshot.ResolvePinnedRunWorkflow(t.Context(), runWorkflow, credentials)
		record("ResolvePinnedRunWorkflow", selector, pinnedWorkflow, err)
		for _, stage := range workflow.Stages {
			instructionRefs[stage.Instructions.Ref] = struct{}{}
		}
	}
	patched, err := snapshot.ResolveRunWorkflow(t.Context(), "artifact-copy@1", decodeExecutionConfigPatch(t,
		`{"workers":{"llmGateway":"local-litellm@1","credential":"development-worker"}}`), credentials)
	record("ResolveRunWorkflow", "artifact-copy@1 with a Worker patch", patched, err)
	for _, ref := range slices.Sorted(maps.Keys(instructionRefs)) {
		instructions, err := snapshot.Instructions(ref)
		record("Instructions", ref, instructions, err)
	}
	profiles := snapshot.AuditProfiles()
	record("AuditProfiles", "", profiles, nil)
	for _, listed := range profiles {
		selector := listed.Ref.Name + "@" + listed.Ref.Version
		profile, err := snapshot.AuditProfile(selector)
		record("AuditProfile", selector, profile, err)
		profile, err = snapshot.ResolveAuditProfile(t.Context(), selector)
		record("ResolveAuditProfile", selector, profile, err)
		profile, err = snapshot.ResolvePinnedAuditProfile(t.Context(), profile)
		record("ResolvePinnedAuditProfile", selector, profile, err)
	}

	if len(profiles) == 0 || len(selectors[ConfigurationExecutionConfigs]) < 2 {
		t.Fatalf("catalog lost its AuditProfiles or ExecutionConfig profiles: %d, %v", len(profiles), selectors)
	}
	snapshotType := reflect.TypeOf(snapshot)
	for index := range snapshotType.NumMethod() {
		if name := snapshotType.Method(index).Name; !covered[name] {
			t.Errorf("Snapshot.%s is not exercised by the deep-copy check", name)
		}
	}
	for _, result := range results {
		for _, span := range mutableSpans(reflect.ValueOf(result.value), result.call) {
			if shared, ok := owned.overlap(span); ok {
				t.Errorf("%s shares memory with %s", span.path, shared.path)
			}
		}
	}
}

// memorySpan is the address range of one allocation a caller could write
// through: a pointer target, a slice backing array or a map.
type memorySpan struct {
	start, end uintptr
	path       string
}

type spanKey struct {
	address uintptr
	typ     reflect.Type
}

type spanWalker struct {
	spans []memorySpan
	seen  map[spanKey]bool
}

// mutableSpans returns every allocation reachable from value, including
// through unexported fields. Strings and scalars boxed in interfaces cannot be
// written through, and zero-size targets share one runtime address, so neither
// is recorded.
func mutableSpans(value reflect.Value, path string) []memorySpan {
	walker := spanWalker{seen: make(map[spanKey]bool)}
	walker.walk(value, path)
	return walker.spans
}

func (w *spanWalker) enter(value reflect.Value, size uintptr, path string) bool {
	key := spanKey{address: value.Pointer(), typ: value.Type()}
	if w.seen[key] {
		return false
	}
	w.seen[key] = true
	if size > 0 {
		w.spans = append(w.spans, memorySpan{start: key.address, end: key.address + size, path: path})
	}
	return true
}

func (w *spanWalker) walk(value reflect.Value, path string) {
	switch value.Kind() {
	case reflect.Pointer:
		if !value.IsNil() && w.enter(value, value.Type().Elem().Size(), path) {
			w.walk(value.Elem(), path)
		}
	case reflect.Slice:
		size := uintptr(value.Cap()) * value.Type().Elem().Size()
		if value.Cap() > 0 && w.enter(value, size, path) {
			for index := range value.Len() {
				w.walk(value.Index(index), fmt.Sprintf("%s[%d]", path, index))
			}
		}
	case reflect.Map:
		if !value.IsNil() && w.enter(value, 1, path) {
			for entry := value.MapRange(); entry.Next(); {
				key := fmt.Sprint(entry.Key())
				w.walk(entry.Key(), path+"[key "+key+"]")
				w.walk(entry.Value(), path+"["+key+"]")
			}
		}
	case reflect.Struct:
		for index := range value.NumField() {
			w.walk(value.Field(index), path+"."+value.Type().Field(index).Name)
		}
	case reflect.Array:
		for index := range value.Len() {
			w.walk(value.Index(index), fmt.Sprintf("%s[%d]", path, index))
		}
	case reflect.Interface:
		if !value.IsNil() {
			w.walk(value.Elem(), path)
		}
	}
}

// spanIndex finds overlapping spans in O(log n): spans are sorted by start and
// maxEnd[i] is the largest end among the first i+1 spans.
type spanIndex struct {
	spans  []memorySpan
	maxEnd []uintptr
}

func newSpanIndex(spans []memorySpan) spanIndex {
	sorted := slices.SortedFunc(slices.Values(spans), func(left, right memorySpan) int {
		return cmp.Compare(left.start, right.start)
	})
	maxEnd := make([]uintptr, len(sorted))
	for index, span := range sorted {
		maxEnd[index] = span.end
		if index > 0 {
			maxEnd[index] = max(maxEnd[index], maxEnd[index-1])
		}
	}
	return spanIndex{spans: sorted, maxEnd: maxEnd}
}

func (x spanIndex) overlap(span memorySpan) (memorySpan, bool) {
	candidates := sort.Search(len(x.spans), func(index int) bool { return x.spans[index].start >= span.end })
	if candidates == 0 || x.maxEnd[candidates-1] <= span.start {
		return memorySpan{}, false
	}
	for index := candidates - 1; index >= 0; index-- {
		if x.spans[index].end > span.start {
			return x.spans[index], true
		}
	}
	return memorySpan{}, false
}

func TestSnapshotAccessorsReturnDeepCopies(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, copyCoreFixture(t))

	policy, _ := snapshot.ModelPolicy("test-worker@1")
	*policy.Temperature = 99
	policyAgain, _ := snapshot.ModelPolicy("test-worker@1")
	if policyAgain.Temperature == nil || *policyAgain.Temperature != 0.1 {
		t.Fatalf("ModelPolicy mutation leaked into Snapshot: %+v", policyAgain)
	}

	gateway, _ := snapshot.LLMGateway("local-litellm@1")
	gateway.CredentialManager.ManagementURL = "https://corrupted.invalid"
	gatewayAgain, _ := snapshot.LLMGateway("local-litellm@1")
	if gatewayAgain.CredentialManager == nil ||
		gatewayAgain.CredentialManager.ManagementURL != "http://127.0.0.1:4000" {
		t.Fatalf("LLMGatewayConfig mutation leaked into Snapshot: %+v", gatewayAgain)
	}
	template, _ := snapshot.AgentTemplate("artifact_builder@1")
	template.Toolsets[0].Tools[0] = "corrupted"
	*template.ModelPolicy.Temperature = 99
	templateAgain, _ := snapshot.AgentTemplate("artifact_builder@1")
	if templateAgain.Toolsets[0].Tools[0] != "list_artifacts" || *templateAgain.ModelPolicy.Temperature != 0.1 {
		t.Fatalf("AgentTemplate mutation leaked into Snapshot: %+v", templateAgain)
	}

	workflow, _ := snapshot.Workflow("artifact-copy@1")
	workflow.Inputs["source"] = resolved.ArtifactSlot{Required: false, MediaTypes: []string{"corrupted/type"}}
	stage := workflow.Stages["copy"]
	stage.Result.Artifacts["copied"] = resolved.ArtifactSlot{MediaTypes: []string{"corrupted/type"}}
	binding := stage.Agents["builder"]
	binding.Template.Toolsets[0].Tools[0] = "corrupted"
	stage.Agents["builder"] = binding
	stage.WorkflowOutputs["result"] = "corrupted"
	workflow.Stages["copy"] = stage

	workflowAgain, _ := snapshot.Workflow("artifact-copy@1")
	stageAgain := workflowAgain.Stages["copy"]
	if !workflowAgain.Inputs["source"].Required || workflowAgain.Inputs["source"].MediaTypes[0] != "text/plain" {
		t.Fatalf("Workflow input mutation leaked into Snapshot: %+v", workflowAgain.Inputs)
	}
	if stageAgain.Result.Artifacts["copied"].MediaTypes[0] != "text/plain" ||
		stageAgain.Agents["builder"].Template.Toolsets[0].Tools[0] != "list_artifacts" ||
		stageAgain.WorkflowOutputs["result"] != "copied" {
		t.Fatalf("nested Workflow mutation leaked into Snapshot: %+v", stageAgain)
	}
}
