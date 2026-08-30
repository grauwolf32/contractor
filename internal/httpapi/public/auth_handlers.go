package public

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/auth"
)

const maximumLoginBodyBytes int64 = 2048

var (
	corsMethods = []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodDelete}
	corsHeaders = []string{
		"Authorization", "Content-Type", "Idempotency-Key", "If-Match", "If-None-Match", "X-CSRF-Token",
	}
)

type browserSessionContextKey struct{}

func (h *handler) login(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.handleError(w, errInvalidRequest)
		return
	}
	var request loginRequest
	if err := decodeJSONBounded(w, r, &request, maximumLoginBodyBytes); err != nil {
		h.handleError(w, err)
		return
	}
	peerIP, err := auth.PeerIP(r.RemoteAddr)
	if err != nil {
		h.handleError(w, errInvalidRequest)
		return
	}
	password := []byte(request.Password)
	request.Password = ""
	defer wipePublicBytes(password)
	result, err := h.dependencies.Authentication.Login(request.Username, password, peerIP)
	if err != nil {
		if limited, ok := auth.IsRateLimited(err); ok {
			retryAfter := int(math.Ceil(limited.RetryAfter.Seconds()))
			if retryAfter < 1 {
				retryAfter = 1
			}
			if retryAfter > 60 {
				retryAfter = 60
			}
			w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
			h.writeError(w, http.StatusTooManyRequests, "rate_limited", "authentication attempts are temporarily limited", true)
			return
		}
		if errors.Is(err, auth.ErrInvalidCredentials) {
			w.Header().Set("WWW-Authenticate", "Session")
			h.writeError(w, http.StatusUnauthorized, "unauthorized", "credentials are invalid", false)
			return
		}
		h.handleError(w, err)
		return
	}
	h.setSessionCookie(w, result.CookieValue, result.Session.AbsoluteExpiresAt)
	result.CookieValue = ""
	writeJSON(w, http.StatusOK, authSession(result.Session))
}

func (h *handler) getSession(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		h.writeSessionUnauthorized(w)
		return
	}
	writeJSON(w, http.StatusOK, authSession(session))
}

func (h *handler) logout(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil || r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
		h.handleError(w, errInvalidRequest)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		h.writeSessionUnauthorized(w)
		return
	}
	h.dependencies.Authentication.Destroy(session.Handle)
	h.clearSessionCookie(w)
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/auth/") {
			w.Header().Set("Cache-Control", "no-store")
		}
		if r.Method == http.MethodOptions || r.URL.Path == "/v1/auth/login" {
			next.ServeHTTP(w, r)
			return
		}
		authorization := r.Header.Values("Authorization")
		if len(authorization) != 0 {
			if len(authorization) != 1 || !strings.HasPrefix(authorization[0], "Bearer ") {
				h.writeBearerUnauthorized(w)
				return
			}
			candidate := strings.TrimPrefix(authorization[0], "Bearer ")
			candidateDigest := sha256.Sum256([]byte(candidate))
			if candidate == "" || len(candidate) > 4096 ||
				subtle.ConstantTimeCompare(candidateDigest[:], h.tokenDigest[:]) != 1 {
				h.writeBearerUnauthorized(w)
				return
			}
			principal := h.dependencies.Authentication.Principal()
			next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), principal)))
			return
		}
		cookieValue, err := h.sessionCookie(r)
		if err != nil {
			h.writeSessionUnauthorized(w)
			return
		}
		session, err := h.dependencies.Authentication.Lookup(cookieValue)
		if err != nil {
			h.writeSessionUnauthorized(w)
			return
		}
		if requestIsUnsafe(r.Method) {
			origins := r.Header.Values("Origin")
			csrf := r.Header.Values("X-CSRF-Token")
			if len(origins) != 1 || !h.dependencies.BrowserOrigins.Allows(origins[0]) || len(csrf) != 1 ||
				h.dependencies.Authentication.ValidateCSRF(session, csrf[0]) != nil {
				h.writeError(w, http.StatusForbidden, "forbidden", "request origin or CSRF token is invalid", false)
				return
			}
		}
		session, err = h.dependencies.Authentication.Accept(session.Handle)
		if err != nil {
			h.writeSessionUnauthorized(w)
			return
		}
		ctx := auth.WithPrincipal(r.Context(), session.Principal)
		ctx = context.WithValue(ctx, browserSessionContextKey{}, session)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (h *handler) cors(next http.Handler, routes *http.ServeMux) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/auth/") {
			w.Header().Set("Cache-Control", "no-store")
		}
		origins := r.Header.Values("Origin")
		if len(origins) != 0 {
			setVary(w.Header(), "Origin")
		}
		if len(origins) > 1 || len(origins) == 1 && !h.dependencies.BrowserOrigins.Allows(origins[0]) {
			h.writeError(w, http.StatusForbidden, "forbidden", "browser origin is not allowed", false)
			return
		}
		if len(origins) == 0 {
			if r.Method == http.MethodOptions || len(r.Header.Values("Access-Control-Request-Method")) != 0 {
				h.writeError(w, http.StatusForbidden, "forbidden", "browser origin is not allowed", false)
				return
			}
			next.ServeHTTP(w, r)
			return
		}
		origin := origins[0]
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID, ETag, Content-Disposition")
		if r.Method != http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		if !validPreflight(r, routes) {
			h.writeError(w, http.StatusForbidden, "forbidden", "CORS preflight is not allowed", false)
			return
		}
		setVary(w.Header(), "Access-Control-Request-Method")
		setVary(w.Header(), "Access-Control-Request-Headers")
		w.Header().Set("Access-Control-Allow-Methods", strings.Join(corsMethods, ", "))
		w.Header().Set("Access-Control-Allow-Headers", strings.Join(corsHeaders, ", "))
		w.Header().Set("Access-Control-Max-Age", "600")
		w.WriteHeader(http.StatusNoContent)
	})
}

func validPreflight(r *http.Request, routes *http.ServeMux) bool {
	methods := r.Header.Values("Access-Control-Request-Method")
	if len(methods) != 1 || !containsFold(corsMethods, methods[0]) || routes == nil {
		return false
	}
	probe := r.Clone(r.Context())
	probe.Method = strings.ToUpper(methods[0])
	_, pattern := routes.Handler(probe)
	if !strings.HasPrefix(pattern, probe.Method+" ") {
		return false
	}
	headers := r.Header.Values("Access-Control-Request-Headers")
	if len(headers) > 1 || len(headers) == 1 && len(headers[0]) > 2048 {
		return false
	}
	if len(headers) == 0 || strings.TrimSpace(headers[0]) == "" {
		return true
	}
	for _, value := range strings.Split(headers[0], ",") {
		value = strings.TrimSpace(value)
		if value == "" || !containsFold(corsHeaders, value) {
			return false
		}
	}
	return true
}

func containsFold(values []string, candidate string) bool {
	for _, value := range values {
		if strings.EqualFold(value, candidate) {
			return true
		}
	}
	return false
}

func setVary(header http.Header, value string) {
	for _, line := range header.Values("Vary") {
		for _, existing := range strings.Split(line, ",") {
			if strings.EqualFold(strings.TrimSpace(existing), value) {
				return
			}
		}
	}
	header.Add("Vary", value)
}

func requestIsUnsafe(method string) bool {
	return method != http.MethodGet && method != http.MethodHead && method != http.MethodOptions
}

func (h *handler) sessionCookie(r *http.Request) (string, error) {
	configured := r.CookiesNamed(h.cookieName)
	if len(configured) != 1 || configured[0].Value == "" {
		return "", auth.ErrInvalidSession
	}
	return configured[0].Value, nil
}

func (h *handler) setSessionCookie(w http.ResponseWriter, value string, absoluteExpiresAt time.Time) {
	http.SetCookie(w, &http.Cookie{
		Name: h.cookieName, Value: value, Path: "/", Expires: absoluteExpiresAt,
		HttpOnly: true, Secure: h.secureCookie, SameSite: http.SameSiteLaxMode,
	})
}

func (h *handler) clearSessionCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name: h.cookieName, Value: "", Path: "/", Expires: time.Unix(1, 0).UTC(), MaxAge: -1,
		HttpOnly: true, Secure: h.secureCookie, SameSite: http.SameSiteLaxMode,
	})
}

func (h *handler) writeBearerUnauthorized(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", "Bearer")
	h.writeError(w, http.StatusUnauthorized, "unauthorized", "valid authentication is required", false)
}

func (h *handler) writeSessionUnauthorized(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", "Session")
	h.writeError(w, http.StatusUnauthorized, "unauthorized", "valid authentication is required", false)
}

func authSession(source auth.Session) authSessionResponse {
	return authSessionResponse{
		Principal: source.Principal, CSRFToken: source.CSRFToken,
		IdleExpiresAt: source.IdleExpiresAt, AbsoluteExpiresAt: source.AbsoluteExpiresAt,
	}
}

func browserSessionFromContext(ctx context.Context) (auth.Session, bool) {
	session, ok := ctx.Value(browserSessionContextKey{}).(auth.Session)
	return session, ok
}

func principalUserID(ctx context.Context) string {
	principal, ok := auth.PrincipalFromContext(ctx)
	if !ok {
		return ""
	}
	return principal.UserID
}

func wipePublicBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
