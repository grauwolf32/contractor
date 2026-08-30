package public

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestBrowserLoginSessionLogoutAndSecureCookie(t *testing.T) {
	fixture := newHandlerFixture(t)
	login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, testBrowserOrigin)
	if login.response.Code != http.StatusOK || login.session.Principal.UserID != "user-1" ||
		login.session.Principal.Username != "admin" || len(login.session.CSRFToken) != 43 ||
		strings.Contains(login.response.Body.String(), login.cookie.Value) ||
		strings.Contains(login.response.Body.String(), testAuthPassword) {
		t.Fatalf("login = %d headers=%v body=%s", login.response.Code, login.response.Header(), login.response.Body.String())
	}
	setCookie := login.response.Header().Get("Set-Cookie")
	for _, required := range []string{
		"__Host-contractor_session=", "Path=/", "HttpOnly", "Secure", "SameSite=Lax",
	} {
		if !strings.Contains(setCookie, required) {
			t.Fatalf("secure session cookie %q lacks %q", setCookie, required)
		}
	}
	if strings.Contains(strings.ToLower(setCookie), "domain=") ||
		login.response.Header().Get("Cache-Control") != "no-store" ||
		login.response.Header().Get("Access-Control-Allow-Origin") != testBrowserOrigin ||
		login.response.Header().Get("Access-Control-Allow-Credentials") != "true" {
		t.Fatalf("unsafe login response headers: %v", login.response.Header())
	}

	recoverRequest := httptest.NewRequest(http.MethodGet, "/v1/auth/session", nil)
	recoverRequest.AddCookie(login.cookie)
	recoverRequest.Header.Set("Origin", testBrowserOrigin)
	recovered := httptest.NewRecorder()
	fixture.handler.ServeHTTP(recovered, recoverRequest)
	var recoveredSession authSessionResponse
	if recovered.Code != http.StatusOK || json.Unmarshal(recovered.Body.Bytes(), &recoveredSession) != nil ||
		recoveredSession.CSRFToken != login.session.CSRFToken ||
		recovered.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("session recovery = %d %v %s", recovered.Code, recovered.Header(), recovered.Body.String())
	}

	logoutRequest := browserRequest(http.MethodPost, "/v1/auth/logout", nil, login.cookie, testBrowserOrigin, login.session.CSRFToken)
	logout := httptest.NewRecorder()
	fixture.handler.ServeHTTP(logout, logoutRequest)
	if logout.Code != http.StatusNoContent || !strings.Contains(logout.Header().Get("Set-Cookie"), "Max-Age=0") ||
		logout.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("logout = %d %v %s", logout.Code, logout.Header(), logout.Body.String())
	}

	reused := httptest.NewRecorder()
	fixture.handler.ServeHTTP(reused, recoverRequest.Clone(recoverRequest.Context()))
	if reused.Code != http.StatusUnauthorized {
		t.Fatalf("destroyed cookie reuse = %d: %s", reused.Code, reused.Body.String())
	}
}

func TestLoopbackDevelopmentUsesSeparateInsecureCookieName(t *testing.T) {
	origin := "http://127.0.0.1:5173"
	origins, err := auth.NewOriginPolicy([]string{origin}, true)
	if err != nil {
		t.Fatal(err)
	}
	fixture := newHandlerFixtureWithAuth(
		t,
		"../../config/testdata/valid",
		newTestAuthentication(t),
		origins,
		true,
		nil,
	)
	login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, origin)
	setCookie := login.response.Header().Get("Set-Cookie")
	if login.response.Code != http.StatusOK ||
		!strings.Contains(setCookie, auth.LoopbackCookieName+"=") ||
		strings.Contains(setCookie, auth.SecureCookieName+"=") || strings.Contains(setCookie, "; Secure") {
		t.Fatalf("loopback cookie = %d %q", login.response.Code, setCookie)
	}
}

func TestCookieUnsafeRequestsFailClosedBeforeDomainSideEffects(t *testing.T) {
	fixture := newHandlerFixture(t)
	login := loginBrowser(t, fixture.handler, "admin", testAuthPassword, testBrowserOrigin)
	if login.response.Code != http.StatusOK {
		t.Fatal(login.response.Body.String())
	}

	fixture.runs.runs["run-csrf"] = runstore.WorkflowRun{
		RunID: "run-csrf", OwnerID: "user-1", WorkflowName: "artifact-copy",
		WorkflowVersion: "1", WorkflowSchemaVersion: contracts.APIVersion, State: runstore.RunRunning,
	}
	cancel := browserRequest(
		http.MethodPost, "/v1/runs/run-csrf/cancel", []byte(`{}`), login.cookie, testBrowserOrigin, "",
	)
	cancel.Header.Set("Content-Type", "application/json")
	cancelResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(cancelResponse, cancel)
	if cancelResponse.Code != http.StatusForbidden || fixture.runs.runs["run-csrf"].State != runstore.RunRunning {
		t.Fatalf("missing-CSRF cancel mutated Run: %d %+v", cancelResponse.Code, fixture.runs.runs["run-csrf"])
	}

	publish := browserRequest(
		http.MethodPost,
		"/v1/configurations/model-policies",
		[]byte(`{"name":"csrf-policy","version":"1","modelPolicy":{"model":"model"}}`),
		login.cookie,
		testBrowserOrigin,
		strings.Repeat("x", 43),
	)
	publish.Header.Set("Content-Type", "application/json")
	publish.Header.Set("Idempotency-Key", "csrf-publication")
	publishResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(publishResponse, publish)
	if publishResponse.Code != http.StatusForbidden {
		t.Fatalf("wrong-CSRF publication = %d: %s", publishResponse.Code, publishResponse.Body.String())
	}
	if _, err := fixture.configs.Snapshot().ModelPolicy("csrf-policy@1"); err == nil {
		t.Fatal("wrong-CSRF publication changed configuration")
	}

	gateway, _ := fixture.configs.Snapshot().LLMGateway("local-litellm@1")
	policy, _ := fixture.configs.Snapshot().ModelPolicy("worker@1")
	credentialBody, err := json.Marshal(createCredentialRequest{
		CredentialID: "csrf-credential", LLMGateway: gateway.Ref,
		GatewayPolicy: credentials.GatewayPolicy{ModelPolicies: []contracts.ModelPolicyRef{policy.Ref}},
	})
	if err != nil {
		t.Fatal(err)
	}
	createCredential := browserRequest(
		http.MethodPost, "/v1/operations/credentials", credentialBody,
		login.cookie, "https://attacker.invalid", login.session.CSRFToken,
	)
	createCredential.Header.Set("Content-Type", "application/json")
	createCredential.Header.Set("Idempotency-Key", "csrf-credential")
	credentialResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(credentialResponse, createCredential)
	if credentialResponse.Code != http.StatusForbidden || credentialResponse.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("disallowed-origin credential create = %d %v", credentialResponse.Code, credentialResponse.Header())
	}
	if _, err := fixture.credentials.GetCredential(t.Context(), "csrf-credential"); err == nil {
		t.Fatal("disallowed-origin request created a credential")
	}
}

func TestLoginFailuresAreGenericAndRateLimitedBySocketPeer(t *testing.T) {
	fixture := newHandlerFixture(t)
	unknown := loginBrowser(t, fixture.handler, "unknown", testAuthPassword, testBrowserOrigin)
	badPassword := loginBrowser(t, fixture.handler, "admin", "incorrect password", testBrowserOrigin)
	if unknown.response.Code != http.StatusUnauthorized || badPassword.response.Code != http.StatusUnauthorized ||
		unknown.response.Body.String() != badPassword.response.Body.String() {
		t.Fatalf("credential failures differ: unknown=%d %s bad=%d %s", unknown.response.Code, unknown.response.Body.String(), badPassword.response.Code, badPassword.response.Body.String())
	}
	for range 3 {
		failed := loginBrowser(t, fixture.handler, "admin", "incorrect password", testBrowserOrigin)
		if failed.response.Code != http.StatusUnauthorized {
			t.Fatalf("bounded failed login = %d: %s", failed.response.Code, failed.response.Body.String())
		}
	}
	limitedRequest := newLoginRequest("admin", "incorrect password", testBrowserOrigin)
	limitedRequest.Header.Set("X-Forwarded-For", "203.0.113.200")
	limited := httptest.NewRecorder()
	fixture.handler.ServeHTTP(limited, limitedRequest)
	if limited.Code != http.StatusTooManyRequests || limited.Header().Get("Retry-After") == "" ||
		limited.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("rate limit = %d %v %s", limited.Code, limited.Header(), limited.Body.String())
	}
}

func TestLoginBodyIsBoundedBeforePasswordVerification(t *testing.T) {
	fixture := newHandlerFixture(t)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/auth/login",
		strings.NewReader(strings.Repeat("x", int(maximumLoginBodyBytes+1))),
	)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Origin", testBrowserOrigin)
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("oversized login = %d %v %s", response.Code, response.Header(), response.Body.String())
	}
}

func TestCredentialedCORSPreflightIsExactAndBounded(t *testing.T) {
	fixture := newHandlerFixture(t)
	unauthorized := httptest.NewRequest(http.MethodGet, "/v1/runs", nil)
	unauthorized.Header.Set("Origin", testBrowserOrigin)
	unauthorizedResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(unauthorizedResponse, unauthorized)
	if unauthorizedResponse.Code != http.StatusUnauthorized ||
		unauthorizedResponse.Header().Get("Access-Control-Allow-Origin") != testBrowserOrigin {
		t.Fatalf("CORS granted authority: %d %v", unauthorizedResponse.Code, unauthorizedResponse.Header())
	}
	preflight := httptest.NewRequest(http.MethodOptions, "/v1/runs", nil)
	preflight.Header.Set("Origin", testBrowserOrigin)
	preflight.Header.Set("Access-Control-Request-Method", http.MethodPost)
	preflight.Header.Set("Access-Control-Request-Headers", "content-type, idempotency-key, x-csrf-token")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, preflight)
	if response.Code != http.StatusNoContent || response.Header().Get("Access-Control-Allow-Origin") != testBrowserOrigin ||
		response.Header().Get("Access-Control-Allow-Credentials") != "true" ||
		response.Header().Get(APIVersionHeader) != APIVersion ||
		!strings.Contains(response.Header().Get("Access-Control-Allow-Headers"), "X-CSRF-Token") ||
		!strings.Contains(response.Header().Get("Access-Control-Expose-Headers"), APIVersionHeader) {
		t.Fatalf("preflight = %d %v %s", response.Code, response.Header(), response.Body.String())
	}
	for _, origin := range []string{"null", "*", "https://attacker.invalid"} {
		request := preflight.Clone(preflight.Context())
		request.Header = preflight.Header.Clone()
		request.Header.Set("Origin", origin)
		rejected := httptest.NewRecorder()
		fixture.handler.ServeHTTP(rejected, request)
		if rejected.Code != http.StatusForbidden || rejected.Header().Get("Access-Control-Allow-Origin") != "" {
			t.Errorf("origin %q = %d %v", origin, rejected.Code, rejected.Header())
		}
	}
	unexpectedHeader := preflight.Clone(preflight.Context())
	unexpectedHeader.Header = preflight.Header.Clone()
	unexpectedHeader.Header.Set("Access-Control-Request-Headers", "X-Forwarded-For")
	rejected := httptest.NewRecorder()
	fixture.handler.ServeHTTP(rejected, unexpectedHeader)
	if rejected.Code != http.StatusForbidden {
		t.Fatalf("unexpected preflight header = %d", rejected.Code)
	}
	unsupportedMethod := preflight.Clone(preflight.Context())
	unsupportedMethod.Header = preflight.Header.Clone()
	unsupportedMethod.Header.Set("Access-Control-Request-Method", http.MethodDelete)
	unsupportedMethodResponse := httptest.NewRecorder()
	fixture.handler.ServeHTTP(unsupportedMethodResponse, unsupportedMethod)
	if unsupportedMethodResponse.Code != http.StatusForbidden {
		t.Fatalf("unimplemented route method preflight = %d", unsupportedMethodResponse.Code)
	}
}

func TestAuthenticationFailuresDoNotLogPasswordCookieOrCSRF(t *testing.T) {
	testAuthenticationOnce.Do(func() {
		testAuthenticationHash, testAuthenticationErr = auth.HashPassword([]byte(testAuthPassword))
	})
	if testAuthenticationErr != nil {
		t.Fatal(testAuthenticationErr)
	}
	bootstrap, err := auth.NewBootstrap("user-1", "admin", testAuthenticationHash)
	if err != nil {
		t.Fatal(err)
	}
	authentication, err := auth.NewService(bootstrap, auth.Options{Random: failingReader{}})
	if err != nil {
		t.Fatal(err)
	}
	var logs bytes.Buffer
	fixture := newHandlerFixtureWithAuth(
		t,
		"../../config/testdata/valid",
		authentication,
		mustTestOrigins(t),
		false,
		slog.New(slog.NewJSONHandler(&logs, nil)),
	)
	request := newLoginRequest("admin", testAuthPassword, testBrowserOrigin)
	request.Header.Set("Cookie", "attacker_cookie_canary")
	request.Header.Set("X-CSRF-Token", "csrf_canary_that_must_never_reach_the_application_log")
	response := httptest.NewRecorder()
	fixture.handler.ServeHTTP(response, request)
	if response.Code != http.StatusInternalServerError {
		t.Fatalf("entropy failure = %d: %s", response.Code, response.Body.String())
	}
	for _, canary := range []string{
		testAuthPassword,
		"attacker_cookie_canary",
		"csrf_canary_that_must_never_reach_the_application_log",
	} {
		if strings.Contains(logs.String(), canary) {
			t.Fatalf("authentication log leaked canary: %s", logs.String())
		}
	}
}

type browserLoginResult struct {
	response *httptest.ResponseRecorder
	cookie   *http.Cookie
	session  authSessionResponse
}

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, io.ErrUnexpectedEOF }

func loginBrowser(t *testing.T, handler http.Handler, username, password, origin string) browserLoginResult {
	t.Helper()
	request := newLoginRequest(username, password, origin)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	result := browserLoginResult{response: response}
	if response.Code == http.StatusOK {
		if err := json.Unmarshal(response.Body.Bytes(), &result.session); err != nil {
			t.Fatal(err)
		}
		cookies := response.Result().Cookies()
		if len(cookies) != 1 {
			t.Fatalf("login cookies = %v", cookies)
		}
		result.cookie = cookies[0]
	}
	return result
}

func newLoginRequest(username, password, origin string) *http.Request {
	body, _ := json.Marshal(loginRequest{Username: username, Password: password})
	request := httptest.NewRequest(http.MethodPost, "/v1/auth/login", bytes.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	if origin != "" {
		request.Header.Set("Origin", origin)
	}
	return request
}

func browserRequest(
	method string,
	target string,
	body []byte,
	cookie *http.Cookie,
	origin string,
	csrf string,
) *http.Request {
	request := httptest.NewRequest(method, target, bytes.NewReader(body))
	request.AddCookie(cookie)
	if origin != "" {
		request.Header.Set("Origin", origin)
	}
	if csrf != "" {
		request.Header.Set("X-CSRF-Token", csrf)
	}
	return request
}
