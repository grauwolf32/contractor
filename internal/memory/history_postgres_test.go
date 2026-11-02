package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

	artifactpostgres "github.com/grauwolf32/contractor/internal/artifacts/postgres"
)

// A Planner Memory note write runs in a transaction Memory owns, so history
// admission cannot commit maintenance inside it. A pending answer rolls the
// note back and then commits the binding's maintenance pass on the pool; the
// model sees the retryable memory_unavailable, never memory_namespace_full,
// and the same write succeeds after the passes (MEM-R99, ART-R127). The
// removed revisions never return to the Run's lifetime revision counter.
func TestPostgresPlannerMemoryCommitsHistoryMaintenanceAfterRollback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	pool := isolatedMemoryPool(t, ctx)
	createRunningMemoryStage(t, ctx, pool, "history-run", "history-stage")
	store, err := NewPostgresStore(pool)
	if err != nil {
		t.Fatal(err)
	}
	namespace, err := NewNamespace(store, Binding{RunID: "history-run", StageExecutionID: "history-stage", Namespace: "builder"})
	if err != nil {
		t.Fatal(err)
	}
	for index := range 130 {
		if _, err := namespace.WriteMemory(ctx, "journal", fmt.Sprintf("entry %d", index), "", nil); err != nil {
			t.Fatal(err)
		}
	}
	counters := func() (retained, inserted int64) {
		t.Helper()
		if err := pool.QueryRow(ctx, `SELECT binding.retained_revisions, scope.revision_count
FROM artifact_bindings AS binding
JOIN artifact_scopes AS scope ON scope.scope_kind = binding.scope_kind AND scope.scope_id = binding.scope_id
WHERE binding.scope_kind = 'run' AND binding.scope_id = 'history-run'
  AND binding.namespace = 'builder' AND binding.name = 'memory.journal'`).Scan(&retained, &inserted); err != nil {
			t.Fatal(err)
		}
		return retained, inserted
	}
	if retained, inserted := counters(); retained != 130 || inserted != 130 {
		t.Fatalf("before retention: %d retained, %d inserted", retained, inserted)
	}
	policy, err := artifactpostgres.NewHistoryPolicy(2, artifactpostgres.DefaultHistoryMaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	limited := artifactpostgres.WithHistoryPolicy(ctx, policy)
	// 130 retained revisions need 129 removals for a limit of 2: two pending
	// answers, each after a committed pass of 64, then the write.
	for pass, want := range []int64{130 - 64, 130 - 128} {
		if _, err := namespace.WriteMemory(limited, "journal", "next", "", nil); memoryErrorCode(err) != CodeUnavailable {
			t.Fatalf("pass %d answered %v, want memory_unavailable", pass, err)
		}
		if retained, inserted := counters(); retained != want || inserted != 130 {
			t.Fatalf("after pass %d: %d retained, %d inserted; want %d and 130", pass, retained, inserted, want)
		}
	}
	note, err := namespace.WriteMemory(limited, "journal", "next", "", nil)
	if err != nil || note.Content != "next" {
		t.Fatalf("write after maintenance = %+v, %v", note, err)
	}
	if retained, inserted := counters(); retained != 2 || inserted != 131 {
		t.Fatalf("after the write: %d retained, %d inserted; want 2 and 131", retained, inserted)
	}
}
