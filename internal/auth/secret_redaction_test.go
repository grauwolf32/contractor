package auth

import (
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// Sessions, logins, the bootstrap, the password hash and the Service redact
// themselves for every fmt verb, flag and nesting, for log/slog and for
// encoding/json (SEC-R88); before V414-011 %d printed the CSRF token.
func TestAuthSecretHoldersRedactForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	expires := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	session := Session{
		Principal: Principal{UserID: "user-1"}, CSRFToken: "hunter2-csrf",
		IdleExpiresAt: expires, AbsoluteExpiresAt: expires, csrfDigest: [32]byte{1, 2, 3, 4, 5, 6},
	}
	sessionForm := `auth.Session{UserID:"user-1" CSRFToken:[REDACTED] IdleExpiresAt:2026-01-02T03:04:05Z ` +
		`AbsoluteExpiresAt:2026-01-02T03:04:05Z}`
	contractstest.AssertRedacted(t, session, "hunter2-csrf", sessionForm)
	login := Login{CookieValue: "hunter2-cookie", Session: session}
	contractstest.AssertRedacted(t, login, "hunter2-cookie", "auth.Login{CookieValue:[REDACTED] Session:"+sessionForm+"}")
	contractstest.AssertRedacted(t, login, "hunter2-csrf", "")
	handle := SessionHandle{digest: [32]byte{'h', 'u', 'n', 't', 'e', 'r'}}
	contractstest.AssertRedacted(t, handle, "hunter", "auth.SessionHandle([REDACTED])")
	hash := passwordHash{salt: [ArgonSaltBytes]byte{'h', 'u', 'n', 't', 'e', 'r'}}
	contractstest.AssertRedacted(t, hash, "hunter", "auth.passwordHash([REDACTED])")
	bootstrap := Bootstrap{Principal: Principal{UserID: "user-1", Username: "admin"}, hash: hash, valid: true}
	contractstest.AssertRedacted(t, bootstrap, "hunter",
		`auth.Bootstrap{UserID:"user-1" Username:"admin" PasswordHash:[REDACTED]}`)
	service := &Service{hash: hash, sessions: map[[32]byte]storedSession{{1}: {csrfToken: "hunter2-stored"}}}
	contractstest.AssertRedacted(t, service, "hunter2-stored", "auth.Service{secrets:[REDACTED]}")
	contractstest.AssertRedacted(t, storedSession{csrfToken: "hunter2-stored"}, "hunter2-stored", "auth.storedSession([REDACTED])")
}
