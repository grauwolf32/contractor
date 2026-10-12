package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/persistence/migrations"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// snapshotJSONRead is one JSON path that SQL reads from a document the Server
// encodes from a Go type and stores in a jsonb column. Go SQL builds these
// paths from the constants beside each type; migrations (indexes and
// triggers) cannot, so sql lists the text each migration must contain.
type snapshotJSONRead struct {
	name     string
	document func(*testing.T) any
	// path selects members; "*" is any member or element, "**" any depth.
	path []string
	// want, when set, is the text SQL compares every selected value with.
	want string
	sql  []string
}

// TestSQLReadsTheJSONMembersItsGoTypesEncode is the regression for SQL that
// read JSON members of Go-encoded snapshots by name: renaming a json tag left
// credential deletion checks, Audit preparation, completion and finding
// intake reading members that no longer exist, and only database tests could
// notice. Each read path must resolve in a value encoded from its Go type, and
// each migration must read the same members.
func TestSQLReadsTheJSONMembersItsGoTypesEncode(t *testing.T) {
	migrationSQL := allMigrationSQL(t)
	for _, read := range snapshotJSONReads() {
		t.Run(read.name, func(t *testing.T) {
			encoded, err := json.Marshal(read.document(t))
			if err != nil {
				t.Fatal(err)
			}
			var document any
			if err := json.Unmarshal(encoded, &document); err != nil {
				t.Fatal(err)
			}
			values := jsonValuesAt(document, read.path)
			if len(values) == 0 {
				t.Fatalf("path %v selects nothing in %s", read.path, encoded)
			}
			for _, value := range values {
				if read.want != "" && jsonText(t, value) != read.want {
					t.Fatalf("path %v selects %s, SQL compares it with %q", read.path, jsonText(t, value), read.want)
				}
			}
			for _, snippet := range read.sql {
				if !strings.Contains(migrationSQL, snippet) {
					t.Errorf("no migration reads %q", snippet)
				}
			}
		})
	}
}

func snapshotJSONReads() []snapshotJSONRead {
	llmCredential, runtimeCredential := "llm-credential", "runtime-credential"
	workerLLM, workerTelemetry, workerProxy, workerCaido, plannerTelemetry :=
		"worker-llm", "worker-telemetry", "worker-proxy", "worker-caido", "planner-telemetry"
	allocationLLMPath := []string{
		runstore.AllocationProvenanceMember, runtimesettings.LLMCredentialMember, llmgateway.CredentialIDMember,
	}
	projectTargetPath := []string{runtimesettings.CredentialMember, runtimesettings.CredentialIDMember}
	prepare := []string{config.AuditProfileWorkflowsMember, "prepare"}
	check := []string{config.AuditProfileWorkflowsMember, "check"}
	member := func(prefix []string, members ...string) []string {
		return append(append([]string{}, prefix...), members...)
	}
	return []snapshotJSONRead{
		// Credential deletion checks and their indexes.
		{
			name: "Workflow snapshot credential", document: workflowSnapshotDocument,
			path: []string{"**", llmgateway.CredentialIDMember}, want: llmCredential,
			sql: []string{"jsonb_path_query_array(workflow_snapshot, '$.**." + llmgateway.CredentialIDMember + ".**', '{}', true)"},
		},
		{
			name: "Run snapshot LLM credentials", document: runSnapshotDocument,
			path: []string{runtimeconfig.LLMCredentialIDsMember, "*"}, want: llmCredential,
			sql: []string{"runtime_config_snapshot -> '" + runtimeconfig.LLMCredentialIDsMember + "'"},
		},
		{
			name: "Run snapshot Runtime credentials", document: runSnapshotDocument,
			path: []string{runtimeconfig.RuntimeCredentialIDsMember, "*"}, want: runtimeCredential,
			sql: []string{"runtime_config_snapshot -> '" + runtimeconfig.RuntimeCredentialIDsMember + "'"},
		},
		{
			name: "Project HTTP target credential", document: projectTargetDocument,
			path: projectTargetPath, want: runtimeCredential,
			sql: []string{
				"project_http_target_snapshot #>> '" + textArray(projectTargetPath) + "'",
				"project_http_target_snapshot ? '" + runtimesettings.CredentialMember + "'",
			},
		},
		{
			name: "allocation LLM credential", document: allocationDocument,
			path: allocationLLMPath, want: llmCredential,
			sql: []string{"runtime_configuration #>> '" + textArray(allocationLLMPath) + "'"},
		},
		{
			name: "allocation Runtime credentials", document: allocationDocument,
			path: []string{
				runstore.AllocationProvenanceMember, runtimesettings.RuntimeCredentialRefsMember, "*",
				runtimesettings.CredentialIDMember,
			},
			want: runtimeCredential,
			sql: []string{"(runtime_configuration -> '" + runstore.AllocationProvenanceMember + "'::text) -> '" +
				runtimesettings.RuntimeCredentialRefsMember + "'::text"},
		},
		{
			name: "Audit baseline LLM credentials", document: baselineDocument,
			path: []string{runtimeconfig.LLMCredentialIDsMember, "*"}, want: llmCredential,
		},
		{
			name: "Audit baseline Runtime credentials", document: baselineDocument,
			path: []string{runtimeconfig.RuntimeCredentialIDsMember, "*"}, want: runtimeCredential,
		},
		{
			name: "RuntimeConfig worker LLM credential", document: runtimeConfigDocument,
			path: textArrayPath(runtimeconfig.WorkerLLMGatewayCredentialPath), want: workerLLM,
		},
		{
			name: "RuntimeConfig worker telemetry credential", document: runtimeConfigDocument,
			path: textArrayPath(runtimeconfig.WorkerTelemetryCredentialPath), want: workerTelemetry,
		},
		{
			name: "RuntimeConfig worker HTTP proxy credential", document: runtimeConfigDocument,
			path: textArrayPath(runtimeconfig.WorkerHTTPProxyCredentialPath), want: workerProxy,
		},
		{
			name: "RuntimeConfig worker Caido credential", document: runtimeConfigDocument,
			path: textArrayPath(runtimeconfig.WorkerCaidoCredentialPath), want: workerCaido,
		},
		{
			name: "RuntimeConfig planner telemetry credential", document: runtimeConfigDocument,
			path: textArrayPath(runtimeconfig.PlannerTelemetryCredentialPath), want: plannerTelemetry,
		},

		// Audit queries and triggers over the retained ResolvedAuditProfile.
		{
			name: "profile role kind", document: auditProfileDocument,
			path: member(prepare, config.AuditWorkflowKindMember), want: string(config.AuditWorkflowPrepare),
			sql: []string{
				"profile_snapshot->'" + config.AuditProfileWorkflowsMember + "'",
				"role.value->>'" + config.AuditWorkflowKindMember + "' = '" + string(config.AuditWorkflowPrepare) + "'",
				"binding->>'" + config.AuditWorkflowKindMember + "' IS DISTINCT FROM '" + string(config.AuditWorkflowPrepare) + "'",
			},
		},
		{
			name: "profile role attempt budget", document: auditProfileDocument,
			path: member(prepare, config.AuditWorkflowMaxRunAttemptsMember), want: "2",
			sql: []string{"(binding->>'" + config.AuditWorkflowMaxRunAttemptsMember + "')::integer"},
		},
		{
			name: "profile role binding", document: auditProfileDocument,
			path: []string{config.AuditProfileWorkflowsMember, "*"},
			sql: []string{
				"profile_snapshot #> ARRAY['" + config.AuditProfileWorkflowsMember + "', e.workflow_role]",
				"profile_snapshot #> ARRAY['" + config.AuditProfileWorkflowsMember + "', NEW.workflow_role]",
			},
		},
		{
			name: "worker completion stage", document: auditProfileDocument,
			path: member(check, "workerCompletion", "stage"), want: "copy",
			sql: []string{"binding -> 'workerCompletion'", "binding #>> '{workerCompletion,stage}'"},
		},
		{
			name: "worker completion agent", document: auditProfileDocument,
			path: member(check, "workerCompletion", "agent"), want: "builder",
			sql: []string{"binding #>> '{workerCompletion,agent}'"},
		},
		{
			name: "worker completion kind", document: auditProfileDocument,
			path: member(check, "workerCompletion", "kind"), want: contracts.AuditCheckResultsV1,
			sql: []string{"binding #>> '{workerCompletion,kind}'"},
		},
		{
			name: "preparation input from the Audit", document: auditProfileDocument,
			path: member(prepare, "inputs", "source", "source"), want: string(config.AuditInputFromAudit),
			sql: []string{
				"(binding->'inputs') ? name", "jsonb_each(binding->'inputs')",
				"mapping.value->>'source' = '" + string(config.AuditInputFromAudit) + "'",
			},
		},
		{
			name: "preparation input name", document: auditProfileDocument,
			path: member(prepare, "inputs", "source", "name"), want: "source",
			sql: []string{"audit.baseline_snapshot #> ARRAY['inputs', mapping.value->>'name']"},
		},
		{
			name: "preparation input from another role", document: auditProfileDocument,
			path: member(prepare, "inputs", "previous", "source"), want: string(config.AuditInputFromPreparation),
			sql: []string{"mapping.value->>'source' = '" + string(config.AuditInputFromPreparation) + "'"},
		},
		{
			name: "preparation input role", document: auditProfileDocument,
			path: member(prepare, "inputs", "previous", "role"), want: "inventory",
			sql: []string{"producer.workflow_role = mapping.value->>'role'"},
		},
		{
			name: "preparation literal parameter", document: auditProfileDocument,
			path: member(prepare, "parameters", "mode", "source"), want: string(config.AuditParameterLiteral),
			sql: []string{
				"(binding->'parameters') ? name", "jsonb_each(binding->'parameters')",
				"mapping.value->>'source' = '" + string(config.AuditParameterLiteral) + "'",
			},
		},
		{
			name: "preparation literal parameter value", document: auditProfileDocument,
			path: member(prepare, "parameters", "mode", "value"), want: "strict",
			sql: []string{"mapping.value->>'value'"},
		},
		{
			name: "preparation scope parameter", document: auditProfileDocument,
			path: member(prepare, "parameters", "objective", "source"), want: string(config.AuditParameterScopeField),
			sql: []string{
				"mapping.value->>'source' = '" + string(config.AuditParameterScopeField) + "'",
				"audit.baseline_snapshot #> ARRAY['scope', mapping.value->>'name']",
			},
		},
		{
			name: "preparation output mapping", document: auditProfileDocument,
			path: member(prepare, "outputs", "inventory"), want: "result",
			sql: []string{"jsonb_each(binding->'outputs')", "binding #>> ARRAY['outputs', output->>'logicalName']"},
		},
		{
			name: "preparation required Workflow input", document: auditProfileDocument,
			path: member(prepare, "workflow", "inputs", "source", "required"), want: "true",
			sql: []string{"binding #>> ARRAY['workflow', 'inputs', mapping.key, 'required'] = 'true'"},
		},
		{
			name: "preparation Workflow output media types", document: auditProfileDocument,
			path: member(prepare, "workflow", "outputs", "result", "mediaTypes", "*"), want: "application/json",
			sql: []string{"binding #> ARRAY['workflow', 'outputs', output->>'workflowOutput', 'mediaTypes']"},
		},
		{
			name: "finding confirmation policy", document: auditProfileDocument,
			path: textArrayPath(config.AuditProfileFindingConfirmationPath), want: string(config.AuditFindingHumanRequired),
		},

		// Audit triggers over the retained baseline snapshot.
		{
			name: "baseline schema", document: baselineDocument,
			path: []string{"schema"}, want: auditbaseline.Schema,
			sql: []string{"baseline_snapshot->>'schema' IS DISTINCT FROM '" + auditbaseline.Schema + "'"},
		},
		{
			name: "baseline inputs", document: baselineDocument,
			path: []string{"inputs", "source", "digest"}, want: "sha256:" + strings.Repeat("a", 64),
			sql: []string{"jsonb_typeof(NEW.baseline_snapshot->'inputs')"},
		},
		{
			name: "baseline inventory", document: baselineDocument,
			path: []string{"inventory", "worklist"},
			sql:  []string{"baseline_snapshot ? 'inventory'"},
		},
		{
			name: "baseline scope field", document: baselineDocument,
			path: []string{"scope", "objective"}, want: "objective text",
		},

		// The completion trigger compares a Run's completion snapshot with the profile.
		{
			name: "Run completion stage", document: auditCompletionDocument,
			path: []string{"stage"}, want: "copy",
			sql: []string{"stored.audit_completion ->> 'stage'"},
		},
		{
			name: "Run completion agent", document: auditCompletionDocument,
			path: []string{"agent"}, want: "builder",
			sql: []string{"stored.audit_completion ->> 'agent'"},
		},
		{
			name: "Run completion contract kind", document: auditCompletionDocument,
			path: []string{"contract", "kind"}, want: contracts.AuditCheckResultsV1,
			sql: []string{"stored.audit_completion #>> '{contract,kind}'"},
		},

		// Eval metrics read the Planner of a stored Stage snapshot.
		{
			name: "Stage snapshot Planner ID", document: stageSnapshotDocument,
			path: textArrayPath(config.StagePlannerIDPath), want: "passthrough",
		},
		{
			name: "Stage snapshot Planner version", document: stageSnapshotDocument,
			path: textArrayPath(config.StagePlannerVersionPath), want: "1",
		},
	}
}

func workflowSnapshotDocument(*testing.T) any {
	return config.ResolvedWorkflow{
		Ref: config.WorkflowRef{Name: "artifact-copy", Version: "1"}, EntryStage: "copy",
		Stages: map[string]config.ResolvedStage{"copy": {
			ExecutionConfig: config.ResolvedStageExecutionConfig{
				Agents: map[string]config.ResolvedConsumerExecutionConfig{"builder": {
					Credential: &llmgateway.LLMCredentialRef{CredentialID: "llm-credential"},
				}},
			},
		}},
	}
}

func stageSnapshotDocument(*testing.T) any {
	return config.ResolvedStage{Planner: config.PlannerRef{PlannerID: "passthrough", Version: "1"}}
}

func runSnapshotDocument(*testing.T) any {
	return runtimeconfig.RunSnapshot{
		LLMCredentialIDs: []string{"llm-credential"}, RuntimeCredentialIDs: []string{"runtime-credential"},
	}
}

func projectTargetDocument(*testing.T) any {
	return runtimesettings.HTTPOriginTargetRef{
		URL: "https://app.example", Credential: &runtimesettings.RuntimeCredentialRef{
			CredentialID: "runtime-credential", Kind: runtimesettings.RuntimeCredentialOriginBasic,
		},
	}
}

func allocationDocument(*testing.T) any {
	return runstore.AllocationRuntimeConfiguration{
		Provenance: runtimesettings.ResolvedRuntimeConfigProvenance{
			LLMCredential: &llmgateway.LLMCredentialRef{CredentialID: "llm-credential"},
			RuntimeCredentialRefs: []runtimesettings.RuntimeCredentialRef{{
				CredentialID: "runtime-credential", Kind: runtimesettings.RuntimeCredentialCaidoBearer,
			}},
		},
	}
}

func baselineDocument(*testing.T) any {
	artifact := auditstore.ExactArtifact{
		Ref:    contracts.ArtifactRef{Namespace: "projects", Name: "source"},
		Digest: "sha256:" + strings.Repeat("a", 64),
	}
	return auditbaseline.BaselineSnapshot{
		Schema: auditbaseline.Schema, Inputs: map[string]auditstore.ExactArtifact{"source": artifact},
		Scope:            auditbaseline.Scope{Objective: "objective text"},
		LLMCredentialIDs: []string{"llm-credential"}, RuntimeCredentialIDs: []string{"runtime-credential"},
		Inventory: &auditbaseline.BaselineInventory{Worklist: artifact},
	}
}

func auditCompletionDocument(*testing.T) any {
	return runstore.AuditCompletionSnapshot{
		Stage: "copy", Agent: "builder",
		Contract: contracts.WorkerCompletionContract{Kind: contracts.AuditCheckResultsV1},
	}
}

func auditProfileDocument(*testing.T) any {
	return config.ResolvedAuditProfile{
		Workflows: map[string]config.ResolvedAuditWorkflowBinding{
			"prepare": {
				Kind: config.AuditWorkflowPrepare, MaxRunAttempts: 2,
				Workflow: config.ResolvedWorkflow{
					Inputs:  map[string]config.ArtifactSlot{"source": {Required: true}},
					Outputs: map[string]config.ArtifactSlot{"result": {MediaTypes: []string{"application/json"}}},
				},
				Inputs: map[string]config.AuditWorkflowInputMapping{
					"source":   {Source: config.AuditInputFromAudit, Name: "source"},
					"previous": {Source: config.AuditInputFromPreparation, Role: "inventory", Name: "result"},
				},
				Parameters: map[string]config.AuditWorkflowParameterMapping{
					"mode":      {Source: config.AuditParameterLiteral, Value: "strict"},
					"objective": {Source: config.AuditParameterScopeField, Name: "objective"},
				},
				Outputs: map[string]string{"inventory": "result"},
			},
			"check": {
				Kind: config.AuditWorkflowCheck,
				WorkerCompletion: &config.AuditWorkerCompletion{
					Kind: contracts.AuditCheckResultsV1, Stage: "copy", Agent: "builder",
				},
			},
		},
		Interaction: config.AuditInteractionPolicy{FindingConfirmation: config.AuditFindingHumanRequired},
	}
}

// runtimeConfigDocument is the canonical document a published RuntimeConfig
// version stores, with every credential member set.
func runtimeConfigDocument(t *testing.T) any {
	const document = `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig",
"metadata":{"name":"credential-paths","version":"1"},
"spec":{"worker":{
  "llmGateway":{"gateway":"local-litellm@1","credential":"worker-llm"},
  "telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"worker-telemetry"},
  "httpProxy":{"adapter":"http-proxy@1","proxyUrl":"http://proxy.example:8080","targets":["llm-gateway"],"credential":"worker-proxy"},
  "caido":{"adapter":"caido-graphql@1","endpoint":"https://caido.example","credential":"worker-caido"}},
 "planner":{"telemetry":{"adapter":"otlp-http@1","endpoint":"https://otel.example/v1/traces","credential":"planner-telemetry"}}}}`
	prepared, err := runtimeconfig.PreparePublication([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	version, err := prepared.Resolve(context.Background(), runtimeconfig.GatewayResolverFunc(
		func(_ context.Context, selector string) (llmgateway.ResolvedLLMGatewayConfig, error) {
			id, gatewayVersion, _ := strings.Cut(selector, "@")
			return llmgateway.ResolvedLLMGatewayConfig{
				Ref: llmgateway.LLMGatewayConfigRef{
					GatewayID: id, Version: gatewayVersion, Digest: "sha256:" + strings.Repeat("b", 64),
				},
				Protocol: llmgateway.OpenAICompatibleProtocol, URL: "http://127.0.0.1:4000/v1",
			}, nil
		},
	))
	if err != nil {
		t.Fatal(err)
	}
	return json.RawMessage(version.CanonicalDocument)
}

// jsonValuesAt returns every value that path selects in a decoded document.
func jsonValuesAt(value any, path []string) []any {
	if len(path) == 0 {
		return []any{value}
	}
	head, rest := path[0], path[1:]
	var children []any
	switch typed := value.(type) {
	case map[string]any:
		if head == "*" || head == "**" {
			for _, child := range typed {
				children = append(children, child)
			}
		} else if child, ok := typed[head]; ok {
			children = append(children, child)
		}
	case []any:
		if head == "*" || head == "**" {
			children = append(children, typed...)
		}
	}
	var result []any
	if head == "**" {
		result = append(result, jsonValuesAt(value, rest)...)
		for _, child := range children {
			result = append(result, jsonValuesAt(child, path)...)
		}
		return result
	}
	for _, child := range children {
		result = append(result, jsonValuesAt(child, rest)...)
	}
	return result
}

func jsonText(t *testing.T, value any) string {
	t.Helper()
	if text, ok := value.(string); ok {
		return text
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// textArrayPath splits a PostgreSQL text-array path such as {a,b}.
func textArrayPath(path string) []string {
	return strings.Split(strings.TrimSuffix(strings.TrimPrefix(path, "{"), "}"), ",")
}

func textArray(members []string) string {
	return "{" + strings.Join(members, ",") + "}"
}

func allMigrationSQL(t *testing.T) string {
	t.Helper()
	entries, err := fs.ReadDir(migrations.Files, ".")
	if err != nil {
		t.Fatal(err)
	}
	var all strings.Builder
	for _, entry := range entries {
		data, err := fs.ReadFile(migrations.Files, entry.Name())
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&all, "-- %s\n%s\n", entry.Name(), data)
	}
	return all.String()
}
