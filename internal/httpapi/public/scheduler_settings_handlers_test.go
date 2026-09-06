package public

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func TestSchedulerSettingsGetReplaceAndNoOp(t *testing.T) {
	var logs bytes.Buffer
	fixture := newHandlerFixtureWithAuth(
		t,
		"../../config/testdata/valid",
		newTestAuthentication(t),
		mustTestOrigins(t),
		false,
		slog.New(slog.NewJSONHandler(&logs, nil)),
	)

	get := authenticatedRequest(http.MethodGet, "/v1/operations/settings/scheduler", bytes.NewReader(nil))
	got := httptest.NewRecorder()
	fixture.handler.ServeHTTP(got, get)
	if got.Code != http.StatusOK || got.Header().Get("ETag") != `"1"` ||
		got.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("GET Scheduler settings = %d headers=%v body=%s", got.Code, got.Header(), got.Body.String())
	}
	var initial schedulerSettingsResponse
	if err := json.Unmarshal(got.Body.Bytes(), &initial); err != nil ||
		initial.MaxConcurrentRuns != 1 || initial.Revision != "1" || initial.UpdatedAt.IsZero() {
		t.Fatalf("initial Scheduler settings = (%+v, %v)", initial, err)
	}

	replace := schedulerSettingsRequest(http.MethodPut, `{"maxConcurrentRuns":4}`, `"1"`)
	replaced := httptest.NewRecorder()
	fixture.handler.ServeHTTP(replaced, replace)
	if replaced.Code != http.StatusOK || replaced.Header().Get("ETag") != `"2"` ||
		replaced.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("PUT Scheduler settings = %d headers=%v body=%s", replaced.Code, replaced.Header(), replaced.Body.String())
	}
	var changed schedulerSettingsResponse
	if err := json.Unmarshal(replaced.Body.Bytes(), &changed); err != nil ||
		changed.MaxConcurrentRuns != 4 || changed.Revision != "2" ||
		!changed.UpdatedAt.After(initial.UpdatedAt) {
		t.Fatalf("changed Scheduler settings = (%+v, %v)", changed, err)
	}
	if fixture.notifier.calls != 1 || fixture.settings.updates != 1 {
		t.Fatalf("changed side effects = wake:%d updates:%d", fixture.notifier.calls, fixture.settings.updates)
	}
	if len(fixture.operations.changes) != 1 ||
		fixture.operations.changes[0].Resource != controlplane.OperationsSchedulerSettings ||
		fixture.operations.changes[0].ResourceID != "" {
		t.Fatalf("Scheduler settings invalidations = %+v", fixture.operations.changes)
	}
	logAfterChange := logs.String()
	for _, expected := range []string{
		`"audit_action":"scheduler_settings.replace"`,
		`"actor_id":"user-1"`,
		`"max_concurrent_runs":4`,
		`"revision":2`,
	} {
		if !strings.Contains(logAfterChange, expected) {
			t.Fatalf("Scheduler settings audit log %q lacks %q", logAfterChange, expected)
		}
	}
	if strings.Contains(logAfterChange, "maxConcurrentRuns") {
		t.Fatalf("Scheduler settings audit copied request JSON: %q", logAfterChange)
	}

	noOp := schedulerSettingsRequest(http.MethodPut, `{"maxConcurrentRuns":4}`, `"2"`)
	unchanged := httptest.NewRecorder()
	fixture.handler.ServeHTTP(unchanged, noOp)
	if unchanged.Code != http.StatusOK || unchanged.Header().Get("ETag") != `"2"` ||
		unchanged.Body.String() != replaced.Body.String() || fixture.notifier.calls != 1 ||
		fixture.settings.updates != 1 || len(fixture.operations.changes) != 1 ||
		logs.String() != logAfterChange {
		t.Fatalf(
			"no-op replacement = status:%d headers:%v body:%s wake:%d updates:%d invalidations:%d logs:%q",
			unchanged.Code, unchanged.Header(), unchanged.Body.String(), fixture.notifier.calls,
			fixture.settings.updates, len(fixture.operations.changes), logs.String(),
		)
	}

	readBack := authenticatedRequest(http.MethodGet, "/v1/operations/settings/scheduler", bytes.NewReader(nil))
	readBackResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(readBackResponse, readBack)
	if readBackResponse.Code != http.StatusOK || readBackResponse.Body.String() != replaced.Body.String() {
		t.Fatalf("read-back Scheduler settings = %d %s", readBackResponse.Code, readBackResponse.Body.String())
	}
}

func TestSchedulerSettingsReplaceRejectsInvalidRequestsWithoutMutation(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		contentType string
		ifMatch     []string
		target      string
		wantStatus  int
	}{
		{name: "missing content type", body: `{"maxConcurrentRuns":2}`, ifMatch: []string{`"1"`}, wantStatus: http.StatusUnsupportedMediaType},
		{name: "wrong content type", body: `{"maxConcurrentRuns":2}`, contentType: "text/plain", ifMatch: []string{`"1"`}, wantStatus: http.StatusUnsupportedMediaType},
		{name: "content type parameters", body: `{"maxConcurrentRuns":2}`, contentType: "application/json; charset=utf-8", ifMatch: []string{`"1"`}, wantStatus: http.StatusUnsupportedMediaType},
		{name: "missing precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", wantStatus: http.StatusBadRequest},
		{name: "repeated precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`"1"`, `"1"`}, wantStatus: http.StatusBadRequest},
		{name: "weak precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`W/"1"`}, wantStatus: http.StatusBadRequest},
		{name: "comma precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`"1","2"`}, wantStatus: http.StatusBadRequest},
		{name: "unquoted precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{"1"}, wantStatus: http.StatusBadRequest},
		{name: "zero-leading precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`"01"`}, wantStatus: http.StatusBadRequest},
		{name: "stale precondition", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`"2"`}, wantStatus: http.StatusPreconditionFailed},
		{name: "unknown field", body: `{"maxConcurrentRuns":2,"extra":true}`, contentType: "application/json", ifMatch: []string{`"1"`}, wantStatus: http.StatusBadRequest},
		{name: "fraction", body: `{"maxConcurrentRuns":2.5}`, contentType: "application/json", ifMatch: []string{`"1"`}, wantStatus: http.StatusBadRequest},
		{name: "null", body: `{"maxConcurrentRuns":null}`, contentType: "application/json", ifMatch: []string{`"1"`}, wantStatus: http.StatusBadRequest},
		{name: "too small", body: `{"maxConcurrentRuns":0}`, contentType: "application/json", ifMatch: []string{`"1"`}, wantStatus: http.StatusBadRequest},
		{name: "too large", body: `{"maxConcurrentRuns":33}`, contentType: "application/json", ifMatch: []string{`"1"`}, wantStatus: http.StatusBadRequest},
		{name: "unexpected query", body: `{"maxConcurrentRuns":2}`, contentType: "application/json", ifMatch: []string{`"1"`}, target: "?force=true", wantStatus: http.StatusBadRequest},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newHandlerFixture(t)
			request := authenticatedRequest(
				http.MethodPut, "/v1/operations/settings/scheduler"+test.target,
				bytes.NewReader([]byte(test.body)),
			)
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			for _, value := range test.ifMatch {
				request.Header.Add("If-Match", value)
			}
			response := httptest.NewRecorder()
			fixture.handler.ServeHTTP(response, request)
			if response.Code != test.wantStatus || fixture.settings.updates != 0 ||
				fixture.notifier.calls != 0 || len(fixture.operations.changes) != 0 {
				t.Fatalf(
					"invalid replacement = status:%d want:%d body:%s updates:%d wakes:%d changes:%d",
					response.Code, test.wantStatus, response.Body.String(), fixture.settings.updates,
					fixture.notifier.calls, len(fixture.operations.changes),
				)
			}
		})
	}
}

func TestSchedulerSettingsRequireOperationsCapability(t *testing.T) {
	current := &handler{dependencies: Dependencies{SchedulerSettings: newFakeSchedulerSettings()}}
	tests := []struct {
		name       string
		principal  *auth.Principal
		wantStatus int
	}{
		{name: "unauthenticated", wantStatus: http.StatusUnauthorized},
		{name: "user only", principal: &auth.Principal{UserID: "user", Username: "user", Capabilities: []string{auth.CapabilityUser}}, wantStatus: http.StatusForbidden},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/v1/operations/settings/scheduler", nil)
			if test.principal != nil {
				request = request.WithContext(auth.WithPrincipal(request.Context(), *test.principal))
			}
			response := httptest.NewRecorder()
			current.getSchedulerSettings(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("capability response = %d, want %d: %s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}
}

func TestSchedulerSettingsBrowserReplaceRequiresCSRF(t *testing.T) {
	fixture := newHandlerFixture(t)
	login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, testBrowserOrigin)
	if login.response.Code != http.StatusOK {
		t.Fatal(login.response.Body.String())
	}
	request := browserRequest(
		http.MethodPut, "/v1/operations/settings/scheduler", []byte(`{"maxConcurrentRuns":2}`),
		login.cookie, testBrowserOrigin, "",
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", `"1"`)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusForbidden || fixture.settings.updates != 0 ||
		fixture.notifier.calls != 0 || len(fixture.operations.changes) != 0 {
		t.Fatalf("missing-CSRF replacement = %d body:%s", response.Code, response.Body.String())
	}
}

func schedulerSettingsRequest(method, body, etag string) *http.Request {
	request := authenticatedRequest(
		method, "/v1/operations/settings/scheduler", bytes.NewReader([]byte(body)),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("If-Match", etag)
	return request
}
