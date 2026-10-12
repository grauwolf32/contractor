package postgres

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestPostgresMutationReceiptsFoldMigration upgrades a schema with a receipt
// in each of the three folded tables and a held Eval claim. Each receipt must
// replay afterwards through the Go key its family is looked up by now, the
// folded tables must be gone and the Eval claim released with its epoch.
func TestPostgresMutationReceiptsFoldMigration(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	installMigrationPrefix(t, ctx, pool, 12)
	if _, err := pool.Exec(ctx, `
INSERT INTO projects (project_id, owner_id, kind, name, request_idempotency_key, request_digest)
VALUES ('upgrade-project', 'owner', 'project', 'Upgrade', 'upgrade-project', 'sha256:'||repeat('a', 64)),
       ('upgrade-evaluation', 'owner', 'evaluation', 'Upgrade evaluation', 'upgrade-evaluation', 'sha256:'||repeat('b', 64));
INSERT INTO audits (
    audit_id, owner_id, project_id, profile_name, profile_version, profile_digest,
    profile_snapshot, input_selection, max_rounds, batch_size, max_items_per_round,
    max_items_total, max_submitted_runs, max_item_run_attempts, max_evidence_bytes
) VALUES ('upgrade-audit', 'owner', 'upgrade-project', 'profile', '1', 'sha256:'||repeat('a', 64),
          '{}', '{}', 1, 1, 10, 10, 10, 1, 1024);
INSERT INTO audit_idempotency (
    owner_id, operation, idempotency_key, request_digest, audit_id, resource_id, response_snapshot
) VALUES ('owner', 'audit.create', 'create-key', 'sha256:'||repeat('1', 64), 'upgrade-audit', 'upgrade-audit',
          '{"auditId": "upgrade-audit"}');
INSERT INTO eval_mutation_receipts (
    owner_id, project_id, resource_id, operation, operation_key, request_sha256, expected_revision, response
) VALUES ('owner', 'upgrade-evaluation', 'experiments', 'experiment-create', 'eval-key',
          'sha256:'||repeat('2', 64), NULL, convert_to('{"experimentId":"e","revision":1}', 'UTF8'));
INSERT INTO runtime_management_operations (
    idempotency_key_digest, request_digest, operation_kind, resource_id, result, actor_id, performed_at
) VALUES ('sha256:'||repeat('c', 64), 'sha256:'||repeat('3', 64), 'runtime-label.delete', 'label',
          '{"deleted": true}', 'operator', clock_timestamp());
INSERT INTO eval_experiments (
    experiment_id, owner_id, project_id, portable_id, control_mode, name, state, max_in_flight, wall_ms
) VALUES ('upgrade-experiment', 'owner', 'upgrade-evaluation', 'upgrade-experiment', 'external', 'Upgrade', 'running', 1, 1000);
INSERT INTO eval_controller_claims (experiment_id, epoch, holder_id, expires_at)
VALUES ('upgrade-experiment', 4, 'old-holder', clock_timestamp() + interval '1 minute')`); err != nil {
		t.Fatal(err)
	}

	result, err := ApplyMigrations(ctx, pool, MigrationBudgets{})
	if err != nil || !slices.Contains(result.AppliedVersions, 13) || !slices.Contains(result.AppliedVersions, 14) {
		t.Fatalf("fold receipts and align claims = %+v, %v", result, err)
	}
	for _, folded := range []struct {
		name     string
		key      ReceiptKey
		digest   string
		want     Receipt
		jsonBody bool
	}{
		{"audit", AuditReceiptKey("owner", "audit.create", "create-key"), "sha256:" + repeat('1'), Receipt{
			Operation: "audit.create", ResourceID: "upgrade-audit", Response: []byte(`{"auditId": "upgrade-audit"}`),
			OwnerID: "owner", AuditID: "upgrade-audit",
		}, true},
		{"eval", EvalReceiptKey("owner", "upgrade-evaluation", "experiments", "experiment-create", "eval-key"), "sha256:" + repeat('2'), Receipt{
			Operation: "experiment-create", ResourceID: "experiments", Response: []byte(`{"experimentId":"e","revision":1}`),
			OwnerID: "owner", ProjectID: "upgrade-evaluation",
		}, false},
		{"runtime-management", RuntimeManagementReceiptKey("sha256:" + repeat('c')), "sha256:" + repeat('3'), Receipt{
			Operation: "runtime-label.delete", ResourceID: "label", Response: []byte(`{"deleted": true}`),
		}, true},
	} {
		receipt, found, err := FindReceipt(ctx, pool, folded.key, folded.digest)
		if err != nil || !found {
			t.Fatalf("%s receipt after the fold = (%t, %v)", folded.name, found, err)
		}
		folded.want.RequestDigest = folded.digest
		got, want := receipt.Response, folded.want.Response
		if folded.jsonBody {
			if !sameJSON(t, got, want) {
				t.Fatalf("%s response = %s, want %s", folded.name, got, want)
			}
			receipt.Response, folded.want.Response = nil, nil
		} else if !bytes.Equal(got, want) {
			t.Fatalf("%s response = %q, want the stored bytes %q", folded.name, got, want)
		}
		if !reflect.DeepEqual(receipt, folded.want) {
			t.Fatalf("%s receipt = %+v, want %+v", folded.name, receipt, folded.want)
		}
	}
	var auditGone, evalGone, runtimeGone, tombstoneReadsReceipts bool
	if err := pool.QueryRow(ctx, `
SELECT to_regclass('audit_idempotency') IS NULL, to_regclass('eval_mutation_receipts') IS NULL,
       to_regclass('runtime_management_operations') IS NULL,
       (SELECT prosrc LIKE '%FROM mutation_receipts i%' FROM pg_proc
         WHERE proname = 'contractor_eval_retain_deleted_execution')`).Scan(
		&auditGone, &evalGone, &runtimeGone, &tombstoneReadsReceipts,
	); err != nil {
		t.Fatal(err)
	}
	if !auditGone || !evalGone || !runtimeGone || !tombstoneReadsReceipts {
		t.Fatalf("folded tables gone = (%t, %t, %t), tombstone trigger reads receipts = %t",
			auditGone, evalGone, runtimeGone, tombstoneReadsReceipts)
	}
	var released bool
	var epoch int64
	if err := pool.QueryRow(ctx, `
SELECT holder_id IS NULL AND claimed_at IS NULL AND expires_at IS NULL, epoch
  FROM eval_controller_claims WHERE experiment_id = 'upgrade-experiment'`).Scan(&released, &epoch); err != nil {
		t.Fatal(err)
	}
	if !released || epoch != 4 {
		t.Fatalf("Eval claim after the migration = released %t at epoch %d, want released at 4", released, epoch)
	}
}

func repeat(character byte) string { return string(bytes.Repeat([]byte{character}, 64)) }

func sameJSON(t *testing.T, left, right []byte) bool {
	t.Helper()
	var leftValue, rightValue any
	if err := json.Unmarshal(left, &leftValue); err != nil {
		t.Fatalf("decode %q: %v", left, err)
	}
	if err := json.Unmarshal(right, &rightValue); err != nil {
		t.Fatalf("decode %q: %v", right, err)
	}
	return reflect.DeepEqual(leftValue, rightValue)
}

// installMigrationPrefix applies the embedded migrations up to and including
// version last, as an older Server would have, so a migration test can seed
// the schema it upgrades.
func installMigrationPrefix(t *testing.T, ctx context.Context, pool *pgxpool.Pool, last int64) {
	t.Helper()
	available, err := loadMigrations()
	if err != nil {
		t.Fatal(err)
	}
	err = InTx(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `CREATE TABLE contractor_schema_migrations (
version bigint PRIMARY KEY CHECK (version > 0), name text NOT NULL CHECK (btrim(name) <> ''),
checksum bytea NOT NULL CHECK (octet_length(checksum) = 32),
applied_at timestamptz NOT NULL DEFAULT clock_timestamp())`); err != nil {
			return err
		}
		for _, item := range available {
			if item.version > last {
				break
			}
			if _, err := tx.Exec(ctx, string(item.contents)); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO contractor_schema_migrations (version, name, checksum) VALUES ($1, $2, $3)`,
				item.version, item.name, item.checksum[:]); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("install the schema through version %d: %v", last, err)
	}
}
