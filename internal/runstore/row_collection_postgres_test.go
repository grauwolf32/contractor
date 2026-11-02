package runstore

import "testing"

func TestPostgresStageCollectionPreservesEmptySliceShapes(t *testing.T) {
	ctx := t.Context()
	store := NewPostgresStore(isolatedRunStorePool(t, ctx))
	for _, test := range []struct {
		name    string
		wantNil bool
		read    func() ([]StageExecution, error)
	}{
		{"stage history", true, func() ([]StageExecution, error) { return store.ListStageExecutions(ctx, "missing-run") }},
		{"unreleased allocations", true, func() ([]StageExecution, error) {
			return store.ListStageExecutionsWithUnreleasedAllocations(ctx, "missing-run")
		}},
		{"terminal recovery", false, func() ([]StageExecution, error) {
			return store.ListTerminalStageExecutionsWithAllocations(ctx, nil)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			values, err := test.read()
			if err != nil || len(values) != 0 || (values == nil) != test.wantNil {
				t.Fatalf("empty stages: nil=%t len=%d err=%v, want nil=%t", values == nil, len(values), err, test.wantNil)
			}
		})
	}
}
