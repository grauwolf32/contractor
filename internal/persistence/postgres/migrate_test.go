package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"testing"
)

func TestEmbeddedMigrationsAreOrderedAndExcludeRuntimeLiveness(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(items) == 0 {
		t.Fatal("no migrations loaded")
	}
	previous := int64(0)
	for _, item := range items {
		if item.version <= previous {
			t.Fatalf("migration versions are not increasing: %d after %d", item.version, previous)
		}
		previous = item.version
		lower := strings.ToLower(string(item.contents))
		for _, forbidden := range []string{"runtime_agents", "heartbeat_lease", "create type"} {
			if strings.Contains(lower, forbidden) {
				t.Fatalf("migration %s contains forbidden durable construct %q", item.name, forbidden)
			}
		}
	}
}

func TestRuntimeConfigurationMigrationContainsExactIdempotentBootstrap(t *testing.T) {
	t.Parallel()

	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var runtimeConfiguration migration
	for _, item := range items {
		if item.name == "000015_runtime_configuration.sql" {
			runtimeConfiguration = item
			break
		}
	}
	if runtimeConfiguration.name == "" {
		t.Fatal("RuntimeConfig migration is not embedded")
	}
	canonical := `{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}`
	sum := sha256.Sum256([]byte(canonical))
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if digest != "sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f" {
		t.Fatalf("test fixture digest = %s", digest)
	}
	contents := string(runtimeConfiguration.contents)
	for _, required := range []string{
		canonical,
		digest,
		"'default'",
		"ON CONFLICT (name, version) DO NOTHING",
		"ON CONFLICT (label) DO NOTHING",
		"revision numeric(20, 0)",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("RuntimeConfig migration does not contain %q", required)
		}
	}
}

func TestRuntimeCredentialMigrationKeepsOnlyEncryptedReplayState(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var runtimeCredentials migration
	for _, item := range items {
		if item.name == "000016_runtime_credentials.sql" {
			runtimeCredentials = item
			break
		}
	}
	if runtimeCredentials.name == "" {
		t.Fatal("Runtime credential migration is not embedded")
	}
	contents := string(runtimeCredentials.contents)
	for _, required := range []string{
		"encryption_schema_version", "request_mac bytea", "octet_length(request_mac) = 32",
		"runtime_credential_tombstones", "contractor_protect_runtime_credential_immutable",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Runtime credential migration does not contain %q", required)
		}
	}
	for _, forbidden := range []string{"plaintext", "request_hash", "secret json", "token text", "password text"} {
		if strings.Contains(strings.ToLower(contents), forbidden) {
			t.Fatalf("Runtime credential migration contains unsafe construct %q", forbidden)
		}
	}
}

func TestCaidoRuntimeCredentialMigrationExtendsBothClosedKinds(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var caido migration
	for _, item := range items {
		if item.name == "000023_caido_runtime_credentials.sql" {
			caido = item
			break
		}
	}
	if caido.name == "" {
		t.Fatal("Caido Runtime credential migration is not embedded")
	}
	contents := string(caido.contents)
	if strings.Count(contents, "'caido-bearer@1'") != 2 ||
		!strings.Contains(contents, "runtime_credentials_credential_kind_check") ||
		!strings.Contains(contents, "runtime_credential_creations_credential_kind_check") {
		t.Fatalf("Caido Runtime credential migration is incomplete: %s", contents)
	}
}

func TestWorkflowRunMetadataLabelMigrationIsBoundedIndexedAndImmutable(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var labels migration
	for _, item := range items {
		if item.name == "000024_workflow_run_metadata_labels.sql" {
			labels = item
			break
		}
	}
	if labels.name == "" {
		t.Fatal("WorkflowRun metadata-label migration is not embedded")
	}
	contents := string(labels.contents)
	for _, required := range []string{
		"PRIMARY KEY (run_id, label_key)",
		"UNIQUE (run_id, ordinal)",
		"ordinal BETWEEN 1 AND 32",
		"workflow_run_metadata_labels_exact_idx",
		"contractor_protect_workflow_run_metadata_label_immutable",
		"ON DELETE CASCADE",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("WorkflowRun metadata-label migration does not contain %q", required)
		}
	}
}

func TestProjectMigrationKeepsIdentityImmutableAndOwnerIndexed(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var projects migration
	for _, item := range items {
		if item.name == "000025_projects.sql" {
			projects = item
			break
		}
	}
	if projects.name == "" {
		t.Fatal("Project migration is not embedded")
	}
	contents := string(projects.contents)
	for _, required := range []string{
		"kind IN ('project', 'evaluation')",
		"UNIQUE (owner_id, request_idempotency_key)",
		"projects_owner_kind_created_idx",
		"contractor_protect_project",
		"NEW.revision <> OLD.revision + 1",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Project migration does not contain %q", required)
		}
	}
}

func TestProjectArtifactScopeMigrationAddsOnlyPublicProjectScope(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var scope migration
	for _, item := range items {
		if item.name == "000026_project_artifact_scope.sql" {
			scope = item
			break
		}
	}
	if scope.name == "" {
		t.Fatal("Project Artifact scope migration is not embedded")
	}
	contents := string(scope.contents)
	for _, required := range []string{
		"scope_kind IN ('user', 'project', 'run')",
		"CASE WHEN scope_kind = 'project' THEN scope_id ELSE NULL END",
		"FOREIGN KEY (project_id) REFERENCES projects(project_id)",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Project Artifact scope migration does not contain %q", required)
		}
	}
}

func TestProjectOutputPublicationMigrationIsExactBoundedAndImmutable(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var publication migration
	for _, item := range items {
		if item.name == "000028_project_output_publications.sql" {
			publication = item
			break
		}
	}
	if publication.name == "" {
		t.Fatal("Project output-publication migration is not embedded")
	}
	contents := string(publication.contents)
	for _, required := range []string{
		"status IN ('published', 'already_present', 'failed')",
		"source_scope_kind text GENERATED ALWAYS AS ('run'::text)",
		"target_scope_kind text GENERATED ALWAYS AS ('project'::text)",
		"source_name = output_name AND target_name = output_name",
		"workflow_run_output_publications_immutable",
		"project_output_publish",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Project output-publication migration does not contain %q", required)
		}
	}
}

func TestTerminalRunPurgeMigrationTracksPinOwnershipAndKeepsBypassScoped(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var purge migration
	for _, item := range items {
		if item.name == "000032_terminal_run_purge.sql" {
			purge = item
			break
		}
	}
	if purge.name == "" {
		t.Fatal("terminal Run purge migration is not embedded")
	}
	contents := string(purge.contents)
	for _, required := range []string{
		"ADD COLUMN run_id text",
		"ALTER COLUMN run_id SET NOT NULL",
		"REFERENCES workflow_runs(run_id) ON DELETE CASCADE",
		"artifact_pins_run_idx",
		"contractor.lifecycle_purge",
		"IN ('run', 'project')",
		"TG_OP = 'DELETE' AND contractor_lifecycle_purge_enabled()",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("terminal Run purge migration does not contain %q", required)
		}
	}
}

func TestProjectDeletionMigrationIsDurableClaimedAndFenced(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var deletion migration
	for _, item := range items {
		if item.name == "000033_project_deletion.sql" {
			deletion = item
			break
		}
	}
	if deletion.name == "" {
		t.Fatal("Project deletion migration is not embedded")
	}
	contents := strings.ToLower(string(deletion.contents))
	for _, required := range []string{
		"lifecycle_state text not null default 'active'",
		"deletion_phase text",
		"deletion_claim_expires_at timestamptz",
		"projects_deletion_claim_idx",
		"contractor_require_active_project",
		"for share",
		"workflow_runs_project_admission",
		"artifact_scopes_project_admission",
		"artifact_bindings_project_admission",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Project deletion migration does not contain %q", required)
		}
	}
}

func TestAuditStateMigrationPinsAttemptsReceiptsClaimsAndProjectFence(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var auditState migration
	for _, item := range items {
		if item.name == "000034_audit_state.sql" {
			auditState = item
			break
		}
	}
	if auditState.name == "" {
		t.Fatal("Audit state migration is not embedded")
	}
	contents := strings.ToLower(string(auditState.contents))
	for _, required := range []string{
		"create table audits",
		"create table audit_rounds",
		"create table audit_items",
		"create table audit_executions",
		"create table audit_execution_items",
		"create table audit_collection_receipts",
		"create table audit_controller_claims",
		"audit_executions_run_unique",
		"audit_executions_role_attempt_unique",
		"audit_execution_items_item_attempt unique",
		"unique (execution_id)",
		"outstanding_run_count",
		"contractor_require_active_audit_project",
		"for share",
		"audit_collection_receipts_protect_immutable",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Audit state migration does not contain %q", required)
		}
	}
	if strings.Contains(contents, "max_active_runs") {
		t.Fatal("Audit state migration retains removed Audit-specific Run concurrency")
	}
}

func TestWorkflowRunQueueMigrationAddsOnlyAPartialReadIndex(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var queue migration
	for _, item := range items {
		if item.name == "000029_workflow_run_queue.sql" {
			queue = item
			break
		}
	}
	if queue.name == "" {
		t.Fatal("WorkflowRun Queue migration is not embedded")
	}
	contents := string(queue.contents)
	for _, required := range []string{
		"workflow_runs_owner_nonterminal_queue_idx",
		"ON workflow_runs (owner_id, created_at, run_id)",
		"WHERE state IN ('initializing', 'running', 'cancelling')",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("WorkflowRun Queue migration does not contain %q", required)
		}
	}
	if strings.Contains(strings.ToUpper(contents), "CREATE TABLE") {
		t.Fatalf("WorkflowRun Queue migration created a second queue table: %s", contents)
	}
}

func TestOwnerQueueControlMigrationIsDurableAndRevisionProtected(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var control migration
	for _, item := range items {
		if item.name == "000031_owner_queue_controls.sql" {
			control = item
			break
		}
	}
	if control.name == "" {
		t.Fatal("owner Queue control migration is not embedded")
	}
	contents := strings.ToLower(string(control.contents))
	for _, required := range []string{
		"create table owner_queue_controls",
		"owner_id text primary key",
		"paused boolean not null default false",
		"revision bigint not null default 1",
		"owner_queue_controls_protect_mutation",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("owner Queue control migration does not contain %q", required)
		}
	}
	for _, forbidden := range []string{"run_id", "stage_execution_id", "priority", "position"} {
		if strings.Contains(contents, forbidden) {
			t.Fatalf("owner Queue control migration contains execution field %q", forbidden)
		}
	}
}

func TestRuntimeAgentPrincipalMigrationStoresConfigurationButNoLiveness(t *testing.T) {
	t.Parallel()
	items, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	var principals migration
	for _, item := range items {
		if item.name == "000017_runtime_agent_principals.sql" {
			principals = item
			break
		}
	}
	if principals.name == "" {
		t.Fatal("Runtime Agent principal migration is not embedded")
	}
	contents := strings.ToLower(string(principals.contents))
	for _, required := range []string{
		"runtime_agent_principals", "runtime_agent_id", "label_revision numeric(20, 0)",
		"contractor_valid_runtime_agent_labels", "contractor_protect_runtime_agent_principal",
	} {
		if !strings.Contains(contents, required) {
			t.Fatalf("Runtime Agent principal migration does not contain %q", required)
		}
	}
	for _, forbidden := range []string{"heartbeat", "lease_expires", "observed_state", "allocation_id", "control_url"} {
		if strings.Contains(contents, forbidden) {
			t.Fatalf("Runtime Agent principal migration contains liveness field %q", forbidden)
		}
	}
}

func TestInvalidDatabaseConfigurationIsRedacted(t *testing.T) {
	t.Parallel()

	const secret = "not-a-real-secret"
	pool, err := OpenPool(context.Background(), "://"+secret, PoolOptions{})
	if pool != nil {
		pool.Close()
		t.Fatal("OpenPool returned a pool for an invalid URL")
	}
	if !errors.Is(err, ErrInvalidDatabaseConfiguration) {
		t.Fatalf("OpenPool error = %v", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("OpenPool error exposed database configuration: %v", err)
	}
}
