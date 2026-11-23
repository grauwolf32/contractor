package agentprincipal

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

func TestPrincipalDatabaseFailureIdentity(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPrincipalPool(t, ctx)
	repository := NewRepository(pool)
	id := strings.Repeat("a", 64)
	at := time.Now().UTC()
	if _, err := repository.Insert(ctx, Principal{RuntimeAgentID: id, Labels: []string{}, LabelRevision: 1,
		CreatedBy: "operator", UpdatedBy: "operator", CreatedAt: at, UpdatedAt: at}); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `CREATE FUNCTION principal_write_failure() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'private SQL payload' USING ERRCODE = 'serialization_failure';
END $$`); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `CREATE TRIGGER aaa_principal_failure BEFORE UPDATE ON runtime_agent_principals
FOR EACH ROW EXECUTE FUNCTION principal_write_failure()`); err != nil {
		t.Fatal(err)
	}
	_, err := repository.ReplaceLabels(ctx, id, 1, []string{"alpha"}, "operator", at)
	if !persistencepostgres.IsTransactionConflict(err) || strings.Contains(err.Error(), "private SQL payload") {
		t.Fatalf("principal write failure lost identity or safe message: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	for name, read := range map[string]func() error{
		"list":       func() error { _, err := repository.List(cancelled, "", 10); return err },
		"get":        func() error { _, err := repository.Get(cancelled, id); return err },
		"allocation": func() error { _, err := repository.HasLiveAllocation(cancelled, id); return err },
		"label":      func() error { _, err := repository.ListByLabel(cancelled, "alpha"); return err },
	} {
		if err := read(); !errors.Is(err, context.Canceled) {
			t.Errorf("%s lost cancellation: %v", name, err)
		}
	}
}
