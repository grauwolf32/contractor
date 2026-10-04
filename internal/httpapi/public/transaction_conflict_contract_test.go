package public

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/routers/gorillamux"
	"github.com/jackc/pgx/v5/pgconn"

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

func TestTransactionConflictResponsesConformToOpenAPI(t *testing.T) {
	router, err := gorillamux.NewRouter(loadPublicOpenAPI(t))
	if err != nil {
		t.Fatalf("build contract router: %v", err)
	}
	for _, code := range []string{"40001", "40P01"} {
		conflict := &pgconn.PgError{Code: code}
		fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid",
			newTestAuthentication(t), mustTestOrigins(t), false, nil,
			func(dependencies *Dependencies) {
				dependencies.SchedulerSettings = conflictingSchedulerSettings{err: conflict}
				dependencies.Audits = &fakeAuditManagement{err: conflict}
			},
		)
		for _, path := range []string{"/v1/operations/settings/scheduler", "/v1/audit-standards"} {
			response := serveAndValidatePublicContract(t, router, fixture.handler,
				newPublicContractRequest(http.MethodGet, path, nil), true)
			if response.Code != http.StatusServiceUnavailable ||
				!strings.Contains(response.Body.String(), `"code":"storage_transaction_conflict"`) {
				t.Fatalf("SQLSTATE %s on GET %s = %d: %s", code, path, response.Code, response.Body.String())
			}
		}
	}
}
