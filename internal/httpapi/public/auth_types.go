package public

// Session login request and response bodies.

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
)

type loginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type authSessionResponse struct {
	Principal         auth.Principal `json:"principal"`
	CSRFToken         string         `json:"csrfToken"`
	IdleExpiresAt     time.Time      `json:"idleExpiresAt"`
	AbsoluteExpiresAt time.Time      `json:"absoluteExpiresAt"`
}
