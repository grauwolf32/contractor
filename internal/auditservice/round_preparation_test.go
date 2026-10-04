package auditservice

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

func TestPrepareNextRoundReportsInconsistentSnapshots(t *testing.T) {
	roundID := "round-1"
	audit := auditstore.Audit{AuditID: "audit", CurrentRoundID: &roundID, ProfileSnapshot: json.RawMessage(`{}`)}
	claim := auditstore.ControllerClaim{AuditID: "audit"}
	for name, snapshot := range map[string]auditstore.ReconcileSnapshot{
		"missing Round":       {Audit: audit},
		"open Round":          {Audit: audit, Round: &auditstore.Round{RoundID: roundID, State: auditstore.RoundExecuting}},
		"undecodable profile": {Audit: audit, Round: &auditstore.Round{RoundID: roundID, State: auditstore.RoundClosed}},
	} {
		t.Run(name, func(t *testing.T) {
			params, reason, err := (&Service{}).PrepareNextRound(t.Context(), claim, snapshot)
			var inconsistent *RoundPreparationError
			if params.RoundID != "" || reason != nil || !errors.Is(err, ErrRoundPreparationInconsistent) ||
				!errors.As(err, &inconsistent) || inconsistent.Diagnostic == "" ||
				!strings.Contains(err.Error(), inconsistent.Diagnostic) {
				t.Fatalf("PrepareNextRound = (%+v, %+v, %v)", params, reason, err)
			}
		})
	}
}

func TestRoundPreparationErrorUnwrapsOnlyItsClass(t *testing.T) {
	err := inconsistentRound("the pinned Audit baseline is invalid", ErrInvalid)
	if !errors.Is(err, ErrRoundPreparationInconsistent) || errors.Is(err, ErrInvalid) ||
		err.Error() != ErrRoundPreparationInconsistent.Error()+": the pinned Audit baseline is invalid: "+ErrInvalid.Error() {
		t.Fatalf("inconsistent Round error = %v", err)
	}
}
