package postgres

import (
	"context"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/explaintest"
	"github.com/jackc/pgx/v5"
)

// TestRowTriggerGuardsFireOnTheColumnsTheyCompare checks the column-scoped
// guards of stage_allocations, audits, artifact_bindings and audit_executions
// (PER-R22): each reads through NEW and OLD only the columns it lists in
// UPDATE OF and the declared qualifiers, and no BEFORE UPDATE trigger of the
// table assigns a column of NEW.
func TestRowTriggerGuardsFireOnTheColumnsTheyCompare(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	for _, table := range []struct {
		name       string
		qualifiers explaintest.Qualifiers
		guards     int
		why        string
	}{
		{name: "stage_allocations", guards: 4,
			why: "the identity, performance policy, placement label and completion guards"},
		// The preparation guard selects the Audit's executions by audit_id.
		{name: "audits", guards: 1, why: "the preparation guard",
			qualifiers: explaintest.Qualifiers{"audits_protect_preparation_baseline": {"audit_id"}}},
		{name: "artifact_bindings", guards: 1, why: "the identity guard; Project and Audit admission depend on other tables"},
		// The preparation check reads the Run and the receipt's source only
		// to validate the receipt being assigned.
		{name: "audit_executions", guards: 2, why: "the collection guard and the preparation check",
			qualifiers: explaintest.Qualifiers{"audit_executions_validate_preparation": {
				"collection_disposition", "collection_retained_refs", "collection_source_output_digest",
				"collection_source_output_ref", "run_id",
			}}},
	} {
		if guards := explaintest.RequireColumnScopedGuards(t, ctx, pool, table.name, table.qualifiers); guards != table.guards {
			t.Errorf("found %d column-scoped guards of %s, want %d: %s (PER-R22)", guards, table.name, table.guards, table.why)
		}
	}
}

// rowTrigger is the migrated definition of one trigger, as pg_get_triggerdef
// prints it without the schema, and the reason it keeps firing on every
// UPDATE or DELETE of its table, when it does.
type rowTrigger struct {
	definition, retained string
}

// rowTriggerInventory is every trigger of the five tables whose trigger work
// V409-006 reviewed. A trigger that fires on UPDATE without a column list
// and without a WHEN condition comparing OLD with NEW, or on DELETE without a
// WHEN condition, states why it is retained.
var rowTriggerInventory = map[string]map[string]rowTrigger{
	"stage_allocations": {
		"stage_allocations_protect_identity":           {definition: "CREATE TRIGGER stage_allocations_protect_identity BEFORE UPDATE OF allocation_id, stage_execution_id, logical_agent_name, namespace, agent_template_ref, worker_runtime_ref, runtime_agent_instance_id, runtime_agent_id, runtime_agent_label_revision, runtime_configuration_schema_version, runtime_configuration, created_at ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_identity()"},
		"stage_allocations_protect_performance_policy": {definition: "CREATE TRIGGER stage_allocations_protect_performance_policy BEFORE UPDATE OF performance_collection_policy ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_protect_stage_allocation_performance_policy()"},
		"stage_allocations_protect_placement_labels":   {definition: "CREATE TRIGGER stage_allocations_protect_placement_labels BEFORE UPDATE OF runtime_placement_labels ON stage_allocations FOR EACH ROW WHEN ((new.runtime_placement_labels IS DISTINCT FROM old.runtime_placement_labels)) EXECUTE FUNCTION contractor_protect_placement_labels()"},
		"stage_allocations_require_completion":         {definition: "CREATE TRIGGER stage_allocations_require_completion BEFORE INSERT OR UPDATE OF completion_contract ON stage_allocations FOR EACH ROW EXECUTE FUNCTION contractor_require_allocation_completion()"},
	},
	"audits": {
		"audits_discard_report_pages":     {definition: "CREATE TRIGGER audits_discard_report_pages AFTER UPDATE OF state ON audits FOR EACH ROW WHEN (((old.state = 'finalizing'::text) AND (new.state <> 'finalizing'::text))) EXECUTE FUNCTION contractor_discard_audit_report_pages()"},
		"audits_hold_artifacts":           {definition: "CREATE TRIGGER audits_hold_artifacts AFTER INSERT ON audits FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts()"},
		"audits_hold_artifacts_on_update": {definition: "CREATE TRIGGER audits_hold_artifacts_on_update AFTER UPDATE OF input_selection, baseline_snapshot ON audits FOR EACH ROW WHEN (((old.input_selection IS DISTINCT FROM new.input_selection) OR (old.baseline_snapshot IS DISTINCT FROM new.baseline_snapshot))) EXECUTE FUNCTION contractor_hold_audit_artifacts()"},
		"audits_release_artifacts": {definition: "CREATE TRIGGER audits_release_artifacts AFTER DELETE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_hold_audit_artifacts()",
			retained: "deleting an Audit releases the exact Project inputs and Skill sources it holds (DEL-R63); an Audit without any changes no revision"},
		"audits_eval_tombstone": {definition: "CREATE TRIGGER audits_eval_tombstone BEFORE DELETE ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_retain_deleted_execution()",
			retained: "every column it matches an audit-create suboperation on is NOT NULL, so any Audit delete can match"},
		"audits_project_admission":            {definition: "CREATE TRIGGER audits_project_admission BEFORE INSERT ON audits FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_admission()"},
		"audits_protect_preparation_baseline": {definition: "CREATE TRIGGER audits_protect_preparation_baseline BEFORE UPDATE OF phase, baseline_snapshot, profile_snapshot, input_selection, profile_name, profile_version, profile_digest, owner_id, project_id ON audits FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_preparation_baseline()"},
		"eval_audits_projection": {definition: "CREATE TRIGGER eval_audits_projection AFTER DELETE OR UPDATE OF state, finished_at, dispatch_state ON audits FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed()",
			retained: "deleting an Audit removes the execution its Eval members project"},
	},
	"artifact_bindings": {
		"artifact_bindings_audit_admission": {definition: "CREATE TRIGGER artifact_bindings_audit_admission BEFORE INSERT OR UPDATE ON artifact_bindings FOR EACH ROW WHEN ((new.scope_kind = 'audit'::text)) EXECUTE FUNCTION contractor_guard_audit_artifact_admission()",
			retained: "every UPDATE of an Audit binding is fenced against Audit deletion; the decision reads the owning Audit and Project, not the binding's mutable columns"},
		"artifact_bindings_project_admission": {definition: "CREATE TRIGGER artifact_bindings_project_admission BEFORE INSERT OR UPDATE ON artifact_bindings FOR EACH ROW WHEN ((new.scope_kind = 'project'::text)) EXECUTE FUNCTION contractor_guard_project_artifact_binding_admission()",
			retained: "every UPDATE of a Project binding is an artifact write that a deleting Project rejects at the write (DEL-R40); the decision reads projects and audits, not the binding's columns"},
		"artifact_bindings_protect_identity": {definition: "CREATE TRIGGER artifact_bindings_protect_identity BEFORE UPDATE OF scope_kind, scope_id, namespace, name, created_at, current_revision, frozen ON artifact_bindings FOR EACH ROW EXECUTE FUNCTION contractor_protect_artifact_binding()"},
		"eval_outputs_projection":            {definition: "CREATE TRIGGER eval_outputs_projection AFTER INSERT ON artifact_bindings FOR EACH ROW WHEN (((new.scope_kind = 'run'::text) AND (new.namespace = 'outputs'::text))) EXECUTE FUNCTION contractor_eval_output_changed()"},
		"eval_outputs_projection_on_update":  {definition: "CREATE TRIGGER eval_outputs_projection_on_update AFTER UPDATE ON artifact_bindings FOR EACH ROW WHEN (((new.scope_kind = 'run'::text) AND (new.namespace = 'outputs'::text) AND ((old.current_revision IS DISTINCT FROM new.current_revision) OR (old.frozen IS DISTINCT FROM new.frozen)))) EXECUTE FUNCTION contractor_eval_output_changed()"},
		"eval_outputs_projection_on_delete":  {definition: "CREATE TRIGGER eval_outputs_projection_on_delete AFTER DELETE ON artifact_bindings FOR EACH ROW WHEN (((old.scope_kind = 'run'::text) AND (old.namespace = 'outputs'::text))) EXECUTE FUNCTION contractor_eval_output_changed()"},
	},
	"audit_executions": {
		"audit_executions_project_admission":              {definition: "CREATE TRIGGER audit_executions_project_admission BEFORE INSERT ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_guard_audit_child_admission()"},
		"audit_executions_protect_collection":             {definition: "CREATE TRIGGER audit_executions_protect_collection BEFORE UPDATE OF execution_id, audit_id, run_id, terminal_outcome, terminal_run_generation, terminal_run_sequence, collection_receipt_id, collection_disposition, collection_source_output_ref, collection_source_output_digest, collection_retained_refs, collection_error_code, collection_request_digest, collected_at ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_execution_collection()"},
		"audit_executions_require_matching_run":           {definition: "CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run AFTER INSERT ON audit_executions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_matching_audit_run()"},
		"audit_executions_require_matching_run_on_update": {definition: "CREATE CONSTRAINT TRIGGER audit_executions_require_matching_run_on_update AFTER UPDATE ON audit_executions DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN (((old.run_id IS DISTINCT FROM new.run_id) OR (old.run_deleted_at IS DISTINCT FROM new.run_deleted_at) OR (old.execution_id IS DISTINCT FROM new.execution_id) OR (old.submission_key IS DISTINCT FROM new.submission_key))) EXECUTE FUNCTION contractor_require_matching_audit_run()"},
		"audit_executions_validate_preparation":           {definition: "CREATE TRIGGER audit_executions_validate_preparation BEFORE INSERT OR UPDATE OF execution_id, audit_id, round_id, role, workflow_role, role_attempt, manifest_ref, manifest_digest, submission_key, request_digest, preparation_snapshot, preparation_outputs, collection_receipt_id ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_validate_audit_preparation_execution()"},
		"eval_audit_children_projection": {definition: "CREATE TRIGGER eval_audit_children_projection AFTER INSERT OR DELETE OR UPDATE ON audit_executions FOR EACH ROW EXECUTE FUNCTION contractor_eval_audit_child_changed()",
			retained: "every execution write changes the state, Run, terminal outcome, collection or Run deletion that Eval inventory reads, and the function's work is one indexed probe of eval_submissions"},
	},
	"workflow_runs": {
		"workflow_runs_hold_audit_inputs_on_insert":         {definition: "CREATE TRIGGER workflow_runs_hold_audit_inputs_on_insert AFTER INSERT ON workflow_runs FOR EACH ROW WHEN ((new.audit_input_materialization IS NOT NULL)) EXECUTE FUNCTION contractor_hold_audit_run_inputs()"},
		"workflow_runs_hold_audit_inputs_on_update":         {definition: "CREATE TRIGGER workflow_runs_hold_audit_inputs_on_update AFTER UPDATE OF audit_input_materialization ON workflow_runs FOR EACH ROW WHEN (((old.audit_input_materialization)::text IS DISTINCT FROM (new.audit_input_materialization)::text)) EXECUTE FUNCTION contractor_hold_audit_run_inputs()"},
		"workflow_runs_protect_audit_input_materialization": {definition: "CREATE TRIGGER workflow_runs_protect_audit_input_materialization BEFORE UPDATE OF audit_input_materialization ON workflow_runs FOR EACH ROW WHEN (((old.audit_input_materialization)::text IS DISTINCT FROM (new.audit_input_materialization)::text)) EXECUTE FUNCTION contractor_protect_audit_input_materialization()"},
		"workflow_runs_release_audit_inputs":                {definition: "CREATE TRIGGER workflow_runs_release_audit_inputs AFTER DELETE ON workflow_runs FOR EACH ROW WHEN ((old.audit_input_materialization IS NOT NULL)) EXECUTE FUNCTION contractor_hold_audit_run_inputs()"},
		"eval_runs_projection": {definition: "CREATE TRIGGER eval_runs_projection AFTER DELETE OR UPDATE OF state, finished_at ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_eval_execution_changed()",
			retained: "deleting a Run removes the execution its Eval members, or those of its Audit, project"},
		"workflow_runs_capture_lifecycle_event":            {definition: "CREATE TRIGGER workflow_runs_capture_lifecycle_event AFTER INSERT ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event()"},
		"workflow_runs_capture_lifecycle_event_on_update":  {definition: "CREATE TRIGGER workflow_runs_capture_lifecycle_event_on_update AFTER UPDATE ON workflow_runs FOR EACH ROW WHEN ((old.state IS DISTINCT FROM new.state)) EXECUTE FUNCTION contractor_capture_workflow_run_lifecycle_event()"},
		"workflow_runs_eval_tombstone":                     {definition: "CREATE TRIGGER workflow_runs_eval_tombstone BEFORE DELETE ON workflow_runs FOR EACH ROW WHEN (((old.project_id IS NOT NULL) AND (old.request_idempotency_key IS NOT NULL) AND (old.request_digest IS NOT NULL))) EXECUTE FUNCTION contractor_eval_retain_deleted_execution()"},
		"workflow_runs_project_admission":                  {definition: "CREATE TRIGGER workflow_runs_project_admission BEFORE INSERT ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_guard_workflow_run_project_admission()"},
		"workflow_runs_hold_skill_sources":                 {definition: "CREATE TRIGGER workflow_runs_hold_skill_sources AFTER INSERT ON workflow_runs FOR EACH ROW WHEN ((new.skill_snapshot <> '[]'::jsonb)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts()"},
		"workflow_runs_hold_skill_sources_on_update":       {definition: "CREATE TRIGGER workflow_runs_hold_skill_sources_on_update AFTER UPDATE OF skill_snapshot ON workflow_runs FOR EACH ROW WHEN ((old.skill_snapshot IS DISTINCT FROM new.skill_snapshot)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts()"},
		"workflow_runs_release_skill_sources":              {definition: "CREATE TRIGGER workflow_runs_release_skill_sources AFTER DELETE ON workflow_runs FOR EACH ROW WHEN ((old.skill_snapshot <> '[]'::jsonb)) EXECUTE FUNCTION contractor_hold_run_skill_artifacts()"},
		"workflow_runs_protect_audit_completion":           {definition: "CREATE TRIGGER workflow_runs_protect_audit_completion BEFORE UPDATE OF audit_completion ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_audit_completion()"},
		"workflow_runs_protect_immutable":                  {definition: "CREATE TRIGGER workflow_runs_protect_immutable BEFORE UPDATE OF run_id, owner_id, project_id, workflow_name, workflow_version, workflow_schema_version, workflow_snapshot, parameters, metadata_labels, request_idempotency_key, request_digest, run_event_generation, runtime_labels, runtime_config_snapshot, project_http_target_snapshot, publication_mode, audit_execution_id, audit_submission_key, created_at, cancellation_schema_version, run_cancellation ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_immutable()"},
		"workflow_runs_protect_placement_labels":           {definition: "CREATE TRIGGER workflow_runs_protect_placement_labels BEFORE UPDATE OF required_placement_labels ON workflow_runs FOR EACH ROW WHEN ((new.required_placement_labels IS DISTINCT FROM old.required_placement_labels)) EXECUTE FUNCTION contractor_protect_placement_labels()"},
		"workflow_runs_protect_skill_snapshot":             {definition: "CREATE TRIGGER workflow_runs_protect_skill_snapshot BEFORE UPDATE OF skill_snapshot ON workflow_runs FOR EACH ROW EXECUTE FUNCTION contractor_protect_workflow_run_skill_snapshot()"},
		"workflow_runs_require_audit_completion":           {definition: "CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion AFTER INSERT ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION contractor_require_audit_completion()"},
		"workflow_runs_require_audit_completion_on_update": {definition: "CREATE CONSTRAINT TRIGGER workflow_runs_require_audit_completion_on_update AFTER UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN ((old.audit_completion IS DISTINCT FROM new.audit_completion)) EXECUTE FUNCTION contractor_require_audit_completion()"},
		"workflow_runs_require_complete_audit_binding": {definition: "CREATE CONSTRAINT TRIGGER workflow_runs_require_complete_audit_binding AFTER INSERT OR UPDATE ON workflow_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW WHEN ((new.publication_mode = 'audit-managed'::text)) EXECUTE FUNCTION contractor_require_complete_audit_run_binding()",
			retained: "the binding of an audit-managed Run is checked at commit against its AuditExecution, whichever Run column the UPDATE changes (PER-R22)"},
	},
}

// TestRowTriggerInventory compares every trigger of the reviewed tables with
// rowTriggerInventory (V409-006, PER-R22): an added, dropped or changed
// trigger fails until the inventory states its event and column conditions,
// and every trigger that still fires on each UPDATE or DELETE of its table
// carries the reason it does. The store tests check that each condition
// keeps the trigger's effects; this test keeps the conditions in view.
func TestRowTriggerInventory(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, storageReviewDatabaseURL(t))
	if _, err := ApplyMigrations(ctx, pool, MigrationBudgets{}); err != nil {
		t.Fatal(err)
	}
	rows, err := pool.Query(ctx, rowTriggerInventorySQL, slices.Sorted(maps.Keys(rowTriggerInventory)))
	if err != nil {
		t.Fatal(err)
	}
	type migrated struct {
		table, name, definition string
		update, delete, columns bool
		when                    string
	}
	triggers, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (migrated, error) {
		var trigger migrated
		err := row.Scan(&trigger.table, &trigger.name, &trigger.definition, &trigger.update, &trigger.delete,
			&trigger.columns, &trigger.when)
		return trigger, err
	})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]map[string]bool{}
	for _, trigger := range triggers {
		if seen[trigger.table] == nil {
			seen[trigger.table] = map[string]bool{}
		}
		seen[trigger.table][trigger.name] = true
		want, ok := rowTriggerInventory[trigger.table][trigger.name]
		if !ok {
			t.Errorf("%s has the trigger %s, which the inventory does not state: %s", trigger.table, trigger.name, trigger.definition)
			continue
		}
		if trigger.definition != want.definition {
			t.Errorf("%s.%s is\n  %s\nwant\n  %s", trigger.table, trigger.name, trigger.definition, want.definition)
		}
		broad := trigger.update && !trigger.columns && !strings.Contains(trigger.when, "old.") ||
			trigger.delete && trigger.when == ""
		switch {
		case broad && want.retained == "":
			t.Errorf("%s.%s fires on every UPDATE or DELETE of %s without a stated reason", trigger.table, trigger.name, trigger.table)
		case !broad && want.retained != "":
			t.Errorf("%s.%s states a reason for firing on every change, but it is scoped", trigger.table, trigger.name)
		}
	}
	for _, table := range slices.Sorted(maps.Keys(rowTriggerInventory)) {
		for _, name := range slices.Sorted(maps.Keys(rowTriggerInventory[table])) {
			if !seen[table][name] {
				t.Errorf("%s has no trigger %s", table, name)
			}
		}
	}
}

// rowTriggerInventorySQL lists the triggers of the tables named in $1 other
// than the internal triggers of foreign keys: the definition without the
// schema, whether it fires on UPDATE and on DELETE, whether it has an UPDATE
// OF column list, and its WHEN condition, or ”.
const rowTriggerInventorySQL = `
SELECT class.relname::text, trigger.tgname::text,
       replace(pg_get_triggerdef(trigger.oid), quote_ident(namespace.nspname) || '.', ''),
       -- UPDATE (16) and DELETE (8) of pg_trigger.tgtype
       trigger.tgtype & 16 <> 0, trigger.tgtype & 8 <> 0, trigger.tgattr <> ''::int2vector,
       COALESCE(substring(pg_get_triggerdef(trigger.oid) FROM ' WHEN (.*) EXECUTE '), '')
  FROM pg_trigger AS trigger
  JOIN pg_class AS class ON class.oid = trigger.tgrelid
  JOIN pg_namespace AS namespace ON namespace.oid = class.relnamespace
 WHERE NOT trigger.tgisinternal AND namespace.nspname = current_schema()
   AND class.relname = ANY($1::text[])
 ORDER BY 1, 2`
