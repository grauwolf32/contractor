package evalservice

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func TestPostgresEvalSettlingEnforcesBudgetUntilAcceptedWorkDrains(t *testing.T) {
	for _, kind := range []string{"workflow", "audit"} {
		for _, budget := range []string{"deadline", "tokens"} {
			t.Run(kind+"/"+budget, func(t *testing.T) {
				h := newHarness(t)
				experiment := h.create(t, kind)
				var draft evaldomain.Draft
				if err := json.Unmarshal(experiment.Draft.Bytes(), &draft); err != nil {
					t.Fatal(err)
				}
				draft.Budgets.MaxInFlight = 8
				if budget == "tokens" {
					limit := int64(100)
					draft.Budgets.MaxObservedTotalTokens = &limit
				}
				update := frozen(t, "DraftUpdate", evaldomain.DraftUpdate{Name: experiment.Name, Draft: draft})
				if err := h.service.tx(t.Context(), func(store *evalstore.Store) error {
					_, err := store.UpdateDraft(t.Context(), h.scope, experiment.ID, update, identity(t, "budget", experiment.Revision, update))
					return err
				}); err != nil {
					t.Fatal(err)
				}
				h.command(t, h.get(t, experiment.ID), "prepare")
				coordinator := h.coordinator(t, "budget-controller")
				tick(t, coordinator)
				h.command(t, h.get(t, experiment.ID), "start")
				for attempt := 0; attempt < 60; attempt++ {
					tick(t, coordinator)
					experiment = h.get(t, experiment.ID)
					if experiment.State == "settling" && experiment.Outstanding == 8 {
						var accepted int
						if err := h.pool.QueryRow(t.Context(), `SELECT count(*) FROM eval_submissions WHERE experiment_id=$1 AND state='accepted'`, experiment.ID).Scan(&accepted); err != nil {
							t.Fatal(err)
						}
						if accepted == 8 {
							break
						}
					}
				}
				if experiment.State != "settling" || experiment.Outstanding != 8 {
					t.Fatalf("fixture did not reach settling with eight live members: %s/%d", experiment.State, experiment.Outstanding)
				}

				if budget == "deadline" {
					h.service.now = func() time.Time { return experiment.DeadlineAt.Add(time.Second) }
				} else {
					store := evalstore.NewPostgresStore(h.pool)
					claims, err := store.Claim(t.Context(), "budget-observer", time.Minute, 1)
					if err != nil || len(claims) != 1 {
						t.Fatalf("claim: %v %v", claims, err)
					}
					members, err := store.Outstanding(t.Context(), experiment.OwnerID, experiment.ID, 100)
					if err != nil || len(members) != 8 {
						t.Fatalf("outstanding: %v %v", members, err)
					}
					if err := h.service.tx(t.Context(), func(store *evalstore.Store) error {
						return store.ObserveTokens(t.Context(), h.scope, experiment.ID, members[0], claims[0], 100)
					}); err != nil {
						t.Fatal(err)
					}
					if err := store.ReleaseClaim(t.Context(), claims[0]); err != nil {
						t.Fatal(err)
					}
				}

				tick(t, coordinator)
				if state := h.get(t, experiment.ID).State; state != "cancelling" {
					t.Fatalf("budget stopped applying to active accepted work in settling: %s", state)
				}
				for attempt := 0; attempt < 4; attempt++ {
					tick(t, coordinator)
					if attempt == 0 {
						table := "workflow_runs"
						if kind == "audit" {
							table = "audits"
						}
						var cancelled int
						if err := h.pool.QueryRow(t.Context(), "SELECT count(*) FROM "+table+" WHERE state='cancelling'").Scan(&cancelled); err != nil {
							t.Fatal(err)
						}
						if cancelled != 8 {
							t.Fatalf("only %d of eight active executions received cancellation", cancelled)
						}
					}
					if kind == "workflow" {
						h.finishRuns(t)
					} else {
						h.finishAudits(t)
					}
				}
				experiment = h.get(t, experiment.ID)
				if experiment.State != "cancelled" || experiment.Outstanding != 0 {
					t.Fatalf("accepted work did not drain: %s/%d", experiment.State, experiment.Outstanding)
				}
			})
		}
	}
}
