package public

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auth"
)

func TestLoginAcceptsMaximumEscapedPasswords(t *testing.T) {
	username := strings.Repeat("a", 64)
	for _, test := range []struct {
		name         string
		password     string
		encodedBytes int
	}{
		{"ascii", strings.Repeat("a", 1024), 1117},
		{"backslash", strings.Repeat(`\`, 1024), 2141},
		{"quote", strings.Repeat(`"`, 1024), 2141},
		{"control", strings.Repeat("\x01", 1024), 6237},
		{"multibyte", strings.Repeat("💡", 256), 1117},
	} {
		t.Run(test.name, func(t *testing.T) {
			if len(test.password) != auth.MaximumPasswordBytes {
				t.Fatalf("password bytes = %d", len(test.password))
			}
			hash, err := auth.HashPassword([]byte(test.password))
			if err != nil {
				t.Fatal(err)
			}
			bootstrap, err := auth.NewBootstrap("user-1", username, hash)
			if err != nil {
				t.Fatal(err)
			}
			service, err := auth.NewService(bootstrap, auth.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := service.Login(username, []byte(test.password), "127.0.0.1"); err != nil {
				t.Fatalf("direct authentication rejected supported password: %v", err)
			}
			fixture := newHandlerFixtureWithAuth(t, "../../config/testdata/valid", service, mustTestOrigins(t), false, nil)
			server := httptest.NewServer(fixture.handler)
			defer server.Close()
			body := marshalLoginBody(t, username, test.password)
			if len(body) != test.encodedBytes {
				t.Fatalf("encoded body bytes = %d, want %d", len(body), test.encodedBytes)
			}
			response, responseBody := postLoginHTTP(t, server, body, testBrowserOrigin, false)
			if response.StatusCode != http.StatusOK {
				t.Fatalf("direct authentication succeeded but HTTP login with %d body bytes = %d: %s", len(body), response.StatusCode, responseBody)
			}
			var session authSessionResponse
			if err := json.Unmarshal(responseBody, &session); err != nil {
				t.Fatal(err)
			}
			if session.Principal.UserID != bootstrap.Principal.UserID || session.Principal.Username != username ||
				session.CSRFToken == "" || len(response.Cookies()) != 1 {
				t.Fatalf("successful HTTP login did not create the expected session: %+v", session)
			}
		})
	}
}

func TestLoginBodyIsBoundedBeforePasswordVerification(t *testing.T) {
	const bodyLimit = 8 * 1024
	for _, chunked := range []bool{false, true} {
		name := "content-length"
		if chunked {
			name = "chunked"
		}
		t.Run(name, func(t *testing.T) {
			fixture := newHandlerFixture(t)
			server := httptest.NewServer(fixture.handler)
			defer server.Close()
			wrongPassword := marshalLoginBody(t, "admin", "incorrect password")
			atLimit := append(wrongPassword, bytes.Repeat([]byte(" "), bodyLimit-len(wrongPassword))...)
			overLimit := append(bytes.Clone(atLimit), ' ')
			// Valid JSON with trailing whitespace tests the raw-byte cap, not a
			// syntax error. Repeated overflow must not spend failed-login attempts.
			for attempt := range 6 {
				response, body := postLoginHTTP(t, server, overLimit, testBrowserOrigin, chunked)
				if response.StatusCode != http.StatusRequestEntityTooLarge ||
					!bytes.Contains(body, []byte(`"request_too_large"`)) || len(response.Cookies()) != 0 {
					t.Fatalf("oversized attempt %d = %d: %s", attempt, response.StatusCode, body)
				}
			}
			response, body := postLoginHTTP(t, server, atLimit, testBrowserOrigin, chunked)
			if response.StatusCode != http.StatusUnauthorized || len(response.Cookies()) != 0 {
				t.Fatalf("exact-limit wrong password after six oversized attempts = %d: %s", response.StatusCode, body)
			}
			validPassword := marshalLoginBody(t, "admin", testAuthPassword)
			validAtLimit := append(validPassword, bytes.Repeat([]byte(" "), bodyLimit-len(validPassword))...)
			response, body = postLoginHTTP(t, server, validAtLimit, testBrowserOrigin, chunked)
			if response.StatusCode != http.StatusOK || len(response.Cookies()) != 1 {
				t.Fatalf("exact-limit valid password = %d: %s", response.StatusCode, body)
			}
		})
	}
}

func TestLoginBodyRetainsDecodedValidationAndStrictParsing(t *testing.T) {
	validBody := marshalLoginBody(t, "admin", testAuthPassword)
	for _, test := range []struct {
		name   string
		body   []byte
		origin string
		status int
	}{
		{"short-password", marshalLoginBody(t, "admin", strings.Repeat("a", 11)), testBrowserOrigin, http.StatusUnauthorized},
		{"long-password", marshalLoginBody(t, "admin", strings.Repeat("a", 1025)), testBrowserOrigin, http.StatusUnauthorized},
		{"long-multibyte-password", marshalLoginBody(t, "admin", strings.Repeat("💡", 256)+"a"), testBrowserOrigin, http.StatusUnauthorized},
		{"long-escaped-password", marshalLoginBody(t, "admin", strings.Repeat("\x01", 1025)), testBrowserOrigin, http.StatusUnauthorized},
		{"long-username", marshalLoginBody(t, strings.Repeat("a", 65), testAuthPassword), testBrowserOrigin, http.StatusUnauthorized},
		{"invalid-username", marshalLoginBody(t, "admin/user", testAuthPassword), testBrowserOrigin, http.StatusUnauthorized},
		{"unknown-field", append(bytes.Clone(validBody[:len(validBody)-1]), []byte(`,"extra":true}`)...), testBrowserOrigin, http.StatusBadRequest},
		{"multiple-values", append(bytes.Clone(validBody), []byte(` {}`)...), testBrowserOrigin, http.StatusBadRequest},
		{"malformed", bytes.Clone(validBody[:len(validBody)-1]), testBrowserOrigin, http.StatusBadRequest},
		{"wrong-type", []byte(`{"username":"admin","password":true}`), testBrowserOrigin, http.StatusBadRequest},
		{"missing-password", []byte(`{"username":"admin"}`), testBrowserOrigin, http.StatusUnauthorized},
		{"disallowed-origin", validBody, "https://attacker.invalid", http.StatusForbidden},
	} {
		t.Run(test.name, func(t *testing.T) {
			if len(test.body) > 8*1024 {
				t.Fatal("fixture exceeds raw-body limit instead of testing decoded validation")
			}
			fixture := newHandlerFixture(t)
			server := httptest.NewServer(fixture.handler)
			defer server.Close()
			response, body := postLoginHTTP(t, server, test.body, test.origin, false)
			if response.StatusCode != test.status || len(response.Cookies()) != 0 {
				t.Fatalf("login = %d, want %d: %s", response.StatusCode, test.status, body)
			}
			if test.status == http.StatusUnauthorized && response.Header.Get("WWW-Authenticate") != "Session" {
				t.Fatalf("invalid credentials lost their authentication challenge: %v", response.Header)
			}
			if test.status == http.StatusForbidden && response.Header.Get("Access-Control-Allow-Origin") != "" {
				t.Fatal("disallowed login origin received CORS permission")
			}
		})
	}
}

func marshalLoginBody(t *testing.T, username, password string) []byte {
	t.Helper()
	body, err := json.Marshal(loginRequest{Username: username, Password: password})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func postLoginHTTP(t *testing.T, server *httptest.Server, body []byte, origin string, chunked bool) (*http.Response, []byte) {
	t.Helper()
	request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL+"/v1/auth/login", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Origin", origin)
	if chunked {
		request.ContentLength = -1
	}
	response, err := server.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.Header.Get("Cache-Control") != "no-store" {
		t.Fatalf("login response omitted no-store: %v", response.Header)
	}
	return response, responseBody
}
