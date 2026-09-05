package config

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

const openAPIValidationEscalationYAML = `        failed:
          escalate:
            maxAttempts: 1
            executionConfig:
              ref: strong-oas-review@1
            then:
              fail: {}
`

func TestExecutionConfigProfilePinsResolvedStageVariant(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	profile, err := snapshot.ExecutionConfig("strong-oas-review@1")
	if err != nil {
		t.Fatal(err)
	}
	assertDigest(t, profile.Ref.Digest)
	validatorOverride, ok := profile.Override.Agents["validator"]
	if !ok || validatorOverride.ModelPolicy == nil ||
		validatorOverride.ModelPolicy.Ref.PolicyID != "strong_domain_worker" {
		t.Fatalf("resolved profile = %+v", profile)
	}

	workflow, err := snapshot.Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	assertPinnedStrongValidationVariant(t, workflow, profile.Ref.Digest)

	patch := decodeExecutionConfigPatch(t, `{
  "stages": {
    "openapi_validate": {
      "agents": {"validator": {"modelPolicy": "worker@1"}}
    }
  }
}`)
	runWorkflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", patch, developmentCredentialLookup(t, snapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	stage := runWorkflow.Stages["openapi_validate"]
	if got := stage.ExecutionConfig.Agents["validator"].ModelPolicy.Ref.PolicyID; got != "worker" {
		t.Fatalf("Run base validator policy = %q, want worker", got)
	}
	if got := stage.On.Failed.Escalate.ExecutionConfig.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID; got != "strong_domain_worker" {
		t.Fatalf("Run escalation policy = %q, want strong_domain_worker", got)
	}
	if got := stage.On.Failed.Escalate.ExecutionConfig.Effective.Agents["validator"].Origins.ModelPolicy; got != "executionConfig.strong-oas-review@1.agents.validator" {
		t.Fatalf("Run escalation origin = %q", got)
	}

	// Accessors return deep copies, including dependency bodies.
	profile.Override.Agents["validator"] = ResolvedExecutionSelectionOverride{}
	again, err := snapshot.ExecutionConfig("strong-oas-review@1")
	if err != nil || again.Override.Agents["validator"].ModelPolicy == nil {
		t.Fatalf("profile accessor leaked caller mutation: (%+v, %v)", again, err)
	}
}

func TestExecutionConfigProfileAndRunSnapshotSurviveCatalogChange(t *testing.T) {
	root := copyConfigTree(t)
	first := mustLoad(t, root, MVPDescriptors())
	stored, err := first.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", ExecutionConfigPatch{}, developmentCredentialLookup(t, first),
	)
	if err != nil {
		t.Fatal(err)
	}
	oldDigest := stored.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig.Ref.Digest

	profilePath := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
	replaceFile(t, profilePath, "modelPolicy: strong_domain_worker@1", "modelPolicy: domain_worker@1")
	second := mustLoad(t, root, MVPDescriptors())
	newWorkflow, err := second.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", ExecutionConfigPatch{}, developmentCredentialLookup(t, second),
	)
	if err != nil {
		t.Fatal(err)
	}
	newVariant := newWorkflow.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig
	if oldDigest == newVariant.Ref.Digest {
		t.Fatal("semantic profile change did not alter its digest")
	}
	if got := newVariant.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID; got != "domain_worker" {
		t.Fatalf("new variant policy = %q", got)
	}

	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var durable ResolvedWorkflow
	if err := json.Unmarshal(encoded, &durable); err != nil {
		t.Fatal(err)
	}
	assertPinnedStrongValidationVariant(t, durable, oldDigest)
}

func TestExecutionConfigDigestNormalizesPresentation(t *testing.T) {
	baselineRoot := copyConfigTree(t)
	baseline := mustLoad(t, baselineRoot, MVPDescriptors())
	baselineProfile, _ := baseline.ExecutionConfig("strong-oas-review@1")

	variantRoot := copyConfigTree(t)
	writeFile(t, filepath.Join(variantRoot, "execution-configs/strong_oas_review.yaml"), []byte(`kind: ExecutionConfig
apiVersion: contractor/v1alpha1
metadata: {version: '1', name: strong-oas-review}
spec:
  agents:
    validator: {modelPolicy: 'strong_domain_worker@1'}
`))
	variant := mustLoad(t, variantRoot, MVPDescriptors())
	variantProfile, _ := variant.ExecutionConfig("strong-oas-review@1")
	if baselineProfile.Ref.Digest != variantProfile.Ref.Digest {
		t.Fatalf("presentation changed digest: %s != %s", baselineProfile.Ref.Digest, variantProfile.Ref.Digest)
	}
}

func TestExecutionConfigProfileSupportsCredentialClear(t *testing.T) {
	root := copyConfigTree(t)
	profilePath := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
	replaceFile(
		t, profilePath,
		"      modelPolicy: strong_domain_worker@1",
		"      modelPolicy: strong_domain_worker@1\n      credential: null",
	)
	snapshot := mustLoad(t, root, MVPDescriptors())
	profile, _ := snapshot.ExecutionConfig("strong-oas-review@1")
	credential := profile.Override.Agents["validator"].Credential
	if credential == nil || !credential.Clear || credential.Ref != nil {
		t.Fatalf("credential clear override = %+v", credential)
	}
}

func TestInlineEscalationResolvesWithoutProfileRef(t *testing.T) {
	root := copyConfigTree(t)
	inline := strings.Replace(
		openAPIValidationEscalationYAML,
		"            executionConfig:\n              ref: strong-oas-review@1",
		"            executionConfig:\n              agents:\n                validator: {modelPolicy: worker@1}", 1,
	)
	replaceEscalation(t, root, inline)
	snapshot := mustLoad(t, root, MVPDescriptors())
	workflow, err := snapshot.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", ExecutionConfigPatch{}, developmentCredentialLookup(t, snapshot),
	)
	if err != nil {
		t.Fatal(err)
	}
	variant := workflow.Stages["openapi_validate"].On.Failed.Escalate.ExecutionConfig
	if variant.Ref != nil || variant.Effective.Agents["validator"].ModelPolicy.Ref.PolicyID != "worker" {
		t.Fatalf("inline variant = %+v", variant)
	}
	if got := variant.Effective.Agents["validator"].Origins.ModelPolicy; got !=
		"workflow.stages.openapi_validate.on.failed.escalate.executionConfig.agents.validator" {
		t.Fatalf("inline origin = %q", got)
	}
}

func TestRunCreationValidatesCredentialsInEveryEscalationVariant(t *testing.T) {
	root := copyConfigTree(t)
	profilePath := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
	replaceFile(
		t, profilePath,
		"      modelPolicy: strong_domain_worker@1",
		"      modelPolicy: strong_domain_worker@1\n      credential: escalation-credential",
	)
	snapshot := mustLoad(t, root, MVPDescriptors())
	lookup := developmentCredentialLookup(t, snapshot)
	_, err := snapshot.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", ExecutionConfigPatch{}, lookup,
	)
	if err == nil || !strings.Contains(err.Error(), "failed escalation") ||
		!strings.Contains(err.Error(), "credential is unavailable") {
		t.Fatalf("missing escalation credential error = %v", err)
	}
	localGateway, _ := snapshot.LLMGateway("local-litellm@1")
	lookup["escalation-credential"] = credentialMetadata("escalation-credential", localGateway.Ref)
	if _, err := snapshot.ResolveRunWorkflow(
		t.Context(), "openapi-from-workspace@5", ExecutionConfigPatch{}, lookup,
	); err != nil {
		t.Fatalf("matching escalation credential was rejected: %v", err)
	}
}

func TestExecutionConfigEscalationRejectsInvalidShapesAndConsumers(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, string)
		want   string
	}{
		{
			name: "ref plus inline",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML,
					"              ref: strong-oas-review@1",
					"              ref: strong-oas-review@1\n              agents:\n                validator: {modelPolicy: domain_worker@1}", 1,
				))
			},
			want: "exactly ref or inline planner/agents",
		},
		{
			name: "empty inline",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML,
					"            executionConfig:\n              ref: strong-oas-review@1",
					"            executionConfig: {}", 1,
				))
			},
			want: "exactly ref or inline planner/agents",
		},
		{
			name: "null inline field beside ref",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML,
					"              ref: strong-oas-review@1",
					"              ref: strong-oas-review@1\n              agents: null", 1,
				))
			},
			want: "exactly ref or inline planner/agents",
		},
		{
			name: "unknown profile",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML, "strong-oas-review@1", "absent@1", 1,
				))
			},
			want: "unknown ExecutionConfig",
		},
		{
			name: "unknown logical binding",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
				replaceFile(t, path, "    validator:", "    ghost:")
			},
			want: "unknown logical Agent",
		},
		{
			name: "empty profile agents",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
				writeFile(t, path, []byte(`apiVersion: contractor/v1alpha1
kind: ExecutionConfig
metadata: {name: strong-oas-review, version: "1"}
spec:
  agents: {}
`))
			},
			want: "agents must be a non-empty mapping",
		},
		{
			name: "empty selection leaf",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
				replaceFile(t, path, "      modelPolicy: strong_domain_worker@1", "      {}")
			},
			want: "must select at least one field",
		},
		{
			name: "unknown model policy",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
				replaceFile(t, path, "strong_domain_worker@1", "absent@1")
			},
			want: "unknown ModelPolicy",
		},
		{
			name: "incompatible Worker policy",
			mutate: func(t *testing.T, root string) {
				path := filepath.Join(root, "execution-configs/strong_oas_review.yaml")
				replaceFile(t, path, "strong_domain_worker@1", "planner@1")
			},
			want: "tool-using Worker modelPolicy requires maxToolCalls",
		},
		{
			name: "passthrough planner override",
			mutate: func(t *testing.T, root string) {
				inline := strings.Replace(
					openAPIValidationEscalationYAML,
					"            executionConfig:\n              ref: strong-oas-review@1",
					"            executionConfig:\n              planner: {modelPolicy: planner@1}", 1,
				)
				replaceEscalation(t, root, inline)
			},
			want: "passthrough@1 does not accept Planner",
		},
		{
			name: "zero maxAttempts",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML, "maxAttempts: 1", "maxAttempts: 0", 1,
				))
			},
			want: "maxAttempts must be at least 1",
		},
		{
			name: "invalid then action",
			mutate: func(t *testing.T, root string) {
				replaceEscalation(t, root, strings.Replace(
					openAPIValidationEscalationYAML, "fail: {}", "succeed: {}", 1,
				))
			},
			want: `action "succeed" is not allowed`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			root := copyConfigTree(t)
			test.mutate(t, root)
			snapshot, err := Load(root, MVPDescriptors())
			if err == nil || snapshot != nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Load() = (%v, %v), want error containing %q", snapshot, err, test.want)
			}
		})
	}
}

func TestValidateWorkflowGraphRejectsTamperedEscalationVariant(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["openapi_validate"]
	stage.On.Failed.Escalate.ExecutionConfig.Effective.Agents["validator"] =
		stage.ExecutionConfig.Agents["validator"]
	workflow.Stages["openapi_validate"] = stage
	if err := ValidateWorkflowGraph(workflow); err == nil ||
		!strings.Contains(err.Error(), "does not match its pinned override") {
		t.Fatalf("ValidateWorkflowGraph() error = %v", err)
	}
}

func replaceEscalation(t *testing.T, root, replacement string) {
	t.Helper()
	path := filepath.Join(root, "workflows/openapi_from_workspace_v5.yaml")
	replaceFile(t, path, openAPIValidationEscalationYAML, replacement)
}

func assertPinnedStrongValidationVariant(t *testing.T, workflow ResolvedWorkflow, digest string) {
	t.Helper()
	action := workflow.Stages["openapi_validate"].On.Failed
	if action.Kind != TransitionEscalate || action.Escalate == nil {
		t.Fatalf("validation failure action = %+v", action)
	}
	variant := action.Escalate.ExecutionConfig
	if variant.Ref == nil || variant.Ref.ConfigID != "strong-oas-review" ||
		variant.Ref.Version != "1" || variant.Ref.Digest != digest {
		t.Fatalf("pinned ExecutionConfig ref = %+v", variant.Ref)
	}
	selection := variant.Effective.Agents["validator"]
	if selection.ModelPolicy.Ref.PolicyID != "strong_domain_worker" ||
		selection.ModelPolicy.Model != "worker-strong-model" {
		t.Fatalf("pinned effective validator = %+v", selection)
	}
}
