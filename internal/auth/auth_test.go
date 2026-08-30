package auth

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const testPassword = "correct horse battery staple"

func TestPasswordHashAndBootstrapAreStrictAndRedacted(t *testing.T) {
	hash, err := HashPassword([]byte(testPassword))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(hash, "$argon2id$v=19$m=65536,t=3,p=1$") {
		t.Fatalf("unexpected PHC envelope")
	}
	bootstrap, err := NewBootstrap("local-admin", "Admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	if !bootstrap.hash.verify([]byte(testPassword)) || bootstrap.hash.verify([]byte("incorrect password")) {
		t.Fatal("Argon2id verifier did not distinguish the password")
	}
	if rendered := fmt.Sprintf("%+v %#v", bootstrap, bootstrap); strings.Contains(rendered, hash) {
		t.Fatalf("bootstrap formatting leaked password hash")
	}
	for _, invalid := range []string{
		strings.Replace(hash, "argon2id", "argon2i", 1),
		strings.Replace(hash, "m=65536", "m=32768", 1),
		strings.Replace(hash, "t=3", "t=4", 1),
		strings.Replace(hash, "p=1", "p=2", 1),
	} {
		if _, err := NewBootstrap("local-admin", "Admin", invalid); !errors.Is(err, ErrInvalidBootstrap) {
			t.Fatalf("accepted invalid PHC parameter set")
		}
	}
}

func TestLoadBootstrapRequiresOwnerOnlyStrictSingleDocument(t *testing.T) {
	hash, err := HashPassword([]byte(testPassword))
	if err != nil {
		t.Fatal(err)
	}
	valid, err := BootstrapYAML("local-admin", "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "local-auth.yaml")
	if err := os.WriteFile(path, valid, 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadBootstrap(path)
	if err != nil || loaded.Principal.UserID != "local-admin" || loaded.Principal.Username != "admin" {
		t.Fatalf("LoadBootstrap = (%+v, %v)", loaded.Principal, err)
	}
	if err := os.Chmod(path, 0o640); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadBootstrap(path); !errors.Is(err, ErrInvalidBootstrap) {
		t.Fatal("group-readable bootstrap was accepted")
	}
	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(filepath.Dir(path), "local-auth-link.yaml")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadBootstrap(link); !errors.Is(err, ErrInvalidBootstrap) {
		t.Fatal("symlinked bootstrap was accepted")
	}
	if err := os.WriteFile(path, append(valid, []byte("unknown: true\n")...), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadBootstrap(path); !errors.Is(err, ErrInvalidBootstrap) {
		t.Fatal("unknown bootstrap field was accepted")
	}
}

func TestSessionLifecycleCSRFRevocationExpiryAndRestart(t *testing.T) {
	bootstrap := testBootstrap(t)
	now := time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	clock := func() time.Time { return now }
	service, err := NewService(bootstrap, Options{Now: clock})
	if err != nil {
		t.Fatal(err)
	}
	logins := make([]Login, 0, 9)
	for range 9 {
		result, err := service.Login("admin", []byte(testPassword), "192.0.2.1")
		if err != nil {
			t.Fatal(err)
		}
		if result.CookieValue == "" || result.Session.CSRFToken == "" ||
			result.Session.Principal.UserID != "local-admin" {
			t.Fatalf("incomplete login result")
		}
		if strings.Contains(fmt.Sprintf("%+v %#v", result, result), result.CookieValue) ||
			strings.Contains(fmt.Sprintf("%+v %#v", result, result), result.Session.CSRFToken) {
			t.Fatal("session formatting leaked a secret")
		}
		logins = append(logins, result)
		now = now.Add(time.Second)
	}
	if _, err := service.Lookup(logins[0].CookieValue); !errors.Is(err, ErrInvalidSession) {
		t.Fatal("ninth login did not revoke the oldest session")
	}
	current, err := service.Lookup(logins[8].CookieValue)
	if err != nil {
		t.Fatal(err)
	}
	if service.ValidateCSRF(current, current.CSRFToken) != nil ||
		!errors.Is(service.ValidateCSRF(current, strings.Repeat("x", 43)), ErrInvalidCSRF) {
		t.Fatal("session-bound CSRF validation failed")
	}
	forged := current
	forged.CSRFToken = strings.Repeat("y", 43)
	if !errors.Is(service.ValidateCSRF(forged, forged.CSRFToken), ErrInvalidCSRF) {
		t.Fatal("mutating the public CSRF view forged authority")
	}
	accepted, err := service.Accept(current.Handle)
	if err != nil || !accepted.IdleExpiresAt.After(current.IdleExpiresAt) {
		t.Fatalf("accepted request did not advance idle expiry: %+v, %v", accepted, err)
	}
	restarted, err := NewService(bootstrap, Options{Now: clock})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := restarted.Lookup(logins[8].CookieValue); !errors.Is(err, ErrInvalidSession) {
		t.Fatal("Server restart retained browser session")
	}
	service.Destroy(current.Handle)
	if _, err := service.Lookup(logins[8].CookieValue); !errors.Is(err, ErrInvalidSession) {
		t.Fatal("destroyed session remained valid")
	}

	idleService, err := NewService(bootstrap, Options{Now: clock})
	if err != nil {
		t.Fatal(err)
	}
	idle, err := idleService.Login("admin", []byte(testPassword), "192.0.2.2")
	if err != nil {
		t.Fatal(err)
	}
	now = idle.Session.IdleExpiresAt
	if _, err := idleService.Lookup(idle.CookieValue); !errors.Is(err, ErrInvalidSession) {
		t.Fatal("idle expiry boundary was accepted")
	}

	now = time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	absoluteService, err := NewService(bootstrap, Options{Now: clock, IdleLimit: 23 * time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	absolute, err := absoluteService.Login("admin", []byte(testPassword), "192.0.2.3")
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(22 * time.Hour)
	if _, err := absoluteService.Accept(absolute.Session.Handle); err != nil {
		t.Fatal(err)
	}
	now = absolute.Session.AbsoluteExpiresAt
	if _, err := absoluteService.Lookup(absolute.CookieValue); !errors.Is(err, ErrInvalidSession) {
		t.Fatal("absolute expiry boundary was accepted after idle refresh")
	}
}

func TestFailureLimiterBoundsInflightAndRollingAttempts(t *testing.T) {
	limiter, err := newFailureLimiter(time.Minute, 5, 30)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	tickets := make([]limiterTicket, 0, 5)
	for range 5 {
		ticket, err := limiter.begin("192.0.2.1", now)
		if err != nil {
			t.Fatal(err)
		}
		tickets = append(tickets, ticket)
	}
	if _, err := limiter.begin("192.0.2.1", now); !errors.Is(err, ErrRateLimited) {
		t.Fatal("per-IP inflight limit was not enforced")
	}
	limiter.complete(tickets[0], false)
	if _, err := limiter.begin("192.0.2.1", now); err != nil {
		t.Fatalf("successful attempt did not release limiter reservation: %v", err)
	}
	if _, err := limiter.begin("192.0.2.1", now.Add(time.Minute)); err != nil {
		t.Fatalf("rolling window did not expire: %v", err)
	}
	global, err := newFailureLimiter(time.Minute, 5, 30)
	if err != nil {
		t.Fatal(err)
	}
	for index := range 30 {
		if _, err := global.begin(fmt.Sprintf("192.0.2.%d", index+1), now); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := global.begin("198.51.100.1", now); !errors.Is(err, ErrRateLimited) {
		t.Fatal("process-wide inflight limit was not enforced")
	}
}

func TestSessionRevocationFeedAndCheckDoNotExtendAuthority(t *testing.T) {
	now := time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	service, err := NewService(testBootstrap(t), Options{Now: func() time.Time { return now }})
	if err != nil {
		t.Fatal(err)
	}
	revocations, cancel := service.SubscribeRevocations()
	defer cancel()
	login, err := service.Login("admin", []byte(testPassword), "192.0.2.10")
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(time.Hour)
	checked, err := service.Check(login.Session.Handle)
	if err != nil || !checked.IdleExpiresAt.Equal(login.Session.IdleExpiresAt) {
		t.Fatalf("non-touching session check = (%+v, %v)", checked, err)
	}
	service.Destroy(login.Session.Handle)
	select {
	case revoked := <-revocations:
		if revoked != login.Session.Handle {
			t.Fatal("revocation feed returned another session")
		}
	case <-time.After(time.Second):
		t.Fatal("session destruction did not publish revocation")
	}
	if _, err := service.Check(login.Session.Handle); !errors.Is(err, ErrInvalidSession) {
		t.Fatalf("destroyed handle check error = %v", err)
	}
}

func TestOriginPolicyIsExactAndAllowsHTTPOnlyForLoopbackDevelopment(t *testing.T) {
	policy, err := NewOriginPolicy([]string{"https://ui.example.test", "https://ui.example.test:8443"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !policy.Allows("https://ui.example.test") || policy.Allows("https://UI.example.test") ||
		policy.Allows("https://ui.example.test/") {
		t.Fatal("origin policy was not an exact allowlist")
	}
	for _, invalid := range []string{"*", "null", "http://ui.example.test", "https://ui.example.test/path"} {
		if _, err := NewOriginPolicy([]string{invalid}, false); !errors.Is(err, ErrInvalidOrigin) {
			t.Fatalf("accepted invalid origin %q", invalid)
		}
	}
	if _, err := NewOriginPolicy([]string{"http://127.0.0.1:5173", "http://localhost:5173"}, true); err != nil {
		t.Fatalf("loopback development origins: %v", err)
	}
	if _, err := NewOriginPolicy([]string{"http://192.0.2.1:5173"}, true); !errors.Is(err, ErrInvalidOrigin) {
		t.Fatal("non-loopback HTTP origin was accepted")
	}
}

func TestConcurrentSessionCreationAndLoginReservationsStayBounded(t *testing.T) {
	service, err := NewService(testBootstrap(t), Options{})
	if err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	errorsChannel := make(chan error, 32)
	for range 32 {
		group.Add(1)
		go func() {
			defer group.Done()
			_, err := service.createSession(time.Now().UTC())
			errorsChannel <- err
		}()
	}
	group.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		if err != nil {
			t.Fatal(err)
		}
	}
	service.mu.Lock()
	sessionCount := len(service.sessions)
	service.mu.Unlock()
	if sessionCount != defaultMaxSessions {
		t.Fatalf("concurrent session count = %d", sessionCount)
	}

	limiter, err := newFailureLimiter(time.Minute, 5, 30)
	if err != nil {
		t.Fatal(err)
	}
	results := make(chan error, 64)
	now := time.Now().UTC()
	for range 64 {
		group.Add(1)
		go func() {
			defer group.Done()
			_, err := limiter.begin("192.0.2.1", now)
			results <- err
		}()
	}
	group.Wait()
	close(results)
	allowed := 0
	for err := range results {
		if err == nil {
			allowed++
		} else if !errors.Is(err, ErrRateLimited) {
			t.Fatal(err)
		}
	}
	if allowed != 5 {
		t.Fatalf("concurrent per-IP reservations = %d", allowed)
	}
}

func testBootstrap(t *testing.T) Bootstrap {
	t.Helper()
	hash, err := HashPassword([]byte(testPassword))
	if err != nil {
		t.Fatal(err)
	}
	bootstrap, err := NewBootstrap("local-admin", "admin", hash)
	if err != nil {
		t.Fatal(err)
	}
	return bootstrap
}
