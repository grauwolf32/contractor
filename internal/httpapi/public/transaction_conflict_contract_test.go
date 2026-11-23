package public

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/grauwolf32/contractor/internal/findingintake"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

type conflictingSchedulerSettings struct{ err error }

func (s conflictingSchedulerSettings) GetSchedulerSettings(context.Context) (settingsstore.SchedulerSettings, error) {
	return settingsstore.SchedulerSettings{}, s.err
}

func (s conflictingSchedulerSettings) UpdateSchedulerSettings(
	context.Context, settingsstore.UpdateSchedulerSettingsParams,
) (settingsstore.SchedulerSettings, error) {
	return settingsstore.SchedulerSettings{}, s.err
}

// conflictingFindingProposals fails an owner import as finding intake does when
// a transaction conflict, such as Project correlation fence contention,
// outlasts its finite transaction retries.
type conflictingFindingProposals struct {
	missingFindingProposals
	err error
}

func (p conflictingFindingProposals) ImportIntoAudit(
	context.Context, findingintake.ImportRequest,
) (findingintake.AuditHold, bool, error) {
	return findingintake.AuditHold{}, false, p.err
}

func TestTransactionConflictResponsesConformToOpenAPI(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	for _, code := range []string{persistencepostgres.SQLStateSerializationFailure, persistencepostgres.SQLStateDeadlockDetected} {
		conflict := &pgconn.PgError{Code: code}
		fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
			newTestAuthentication(t), mustTestOrigins(t), false, nil,
			func(dependencies *testServices) {
				dependencies.SchedulerSettings = conflictingSchedulerSettings{err: conflict}
				dependencies.Audits = &fakeAuditManagement{err: conflict}
				dependencies.FindingProposals = conflictingFindingProposals{
					err: fmt.Errorf("lock Project correlation before finding retention: %w", conflict),
				}
			},
		)
		requests := []*http.Request{
			newPublicContractRequest(http.MethodGet, "/v1/operations/settings/scheduler", nil),
			newPublicContractRequest(http.MethodGet, "/v1/audit-standards", nil),
			jsonContractRequest(http.MethodPost, "/v1/audits/audit-one/finding-proposal-imports",
				`{"runId":"run-one","proposal":{"namespace":"finding-proposals","name":"proposal-one","revision":"proposal-r1"}}`),
		}
		for _, request := range requests {
			method, path := request.Method, request.URL.Path
			response := serveAndValidatePublicContract(t, router, fixture.handler, request, true)
			if response.Code != http.StatusServiceUnavailable ||
				!strings.Contains(response.Body.String(), `"code":"storage_transaction_conflict"`) ||
				!strings.Contains(response.Body.String(), `"retryable":true`) {
				t.Fatalf("SQLSTATE %s on %s %s = %d: %s", code, method, path, response.Code, response.Body.String())
			}
		}
	}
}
