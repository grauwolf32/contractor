package pgstore

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func TestSharedPostgresCostSurvivesAuditPurge(t *testing.T) {
	f := newCollectingAuditFixture(t, "cost-purge", 1<<20)
	collectPurgeFinding(t, f, "cost-collection")
	c := reporting.CostContribution{ReportID: "cost", ReceivedAt: time.Date(2026, 11, 8, 0, 0, 0, 0, time.UTC), Complete: true, ModelCalls: 1, Cost: &reporting.LLMCost{Currency: "USD", Calls: 1, Groups: []reporting.LLMCostGroup{{Model: "model", Phase: "main", Source: "provider", Calls: 1, Amount: "0.25"}}}}
	total := reporting.NewLLMCostTotals(false)
	if err := total.ReplaceCostContribution(nil, &c, "worker", true, c.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	missing := reporting.CostContribution{ReceivedAt: c.ReceivedAt}
	if err := total.ReplaceCostContribution(nil, &missing, "planner", true, c.ReceivedAt); err != nil {
		t.Fatal(err)
	}
	if err := persistencepostgres.InTx(f.ctx, f.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if err := NewRunCostsTx(tx).Apply(f.ctx, f.execution.ExecutionID, *f.execution.RunID, "worker", nil, &c, total); err != nil {
			return err
		}
		if err := projectstore.NewPostgresStore(tx).ApplyLLMCost(f.ctx, f.audit.OwnerID, f.project.ProjectID, "worker", true, nil, &c); err != nil {
			return err
		}
		if err := NewRunCostsTx(tx).Apply(f.ctx, f.execution.ExecutionID, *f.execution.RunID, "planner", nil, &missing, total); err != nil {
			return err
		}
		return projectstore.NewPostgresStore(tx).ApplyLLMCost(f.ctx, f.audit.OwnerID, f.project.ProjectID, "planner", true, nil, &missing)
	}); err != nil {
		t.Fatal(err)
	}
	claim := drainDeletedAuditForPurge(t, f)
	if err := purgeFixtureUntilAbsent(t, f, claim); err != nil {
		t.Fatal(err)
	}
	got, _, _, err := projectstore.NewPostgresStore(f.pool).ReadLLMCost(f.ctx, f.audit.OwnerID, f.project.ProjectID)
	if err != nil || got.Amount != "0.25" || got.ChecksAmount != "0.25" || got.UnknownReports != "1" || got.Reports != "2" {
		t.Fatalf("Audit purge lost Project spending: %+v %v", got, err)
	}
}
