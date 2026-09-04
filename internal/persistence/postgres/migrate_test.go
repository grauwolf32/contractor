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
