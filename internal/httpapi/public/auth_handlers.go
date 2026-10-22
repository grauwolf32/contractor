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

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/auth"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

// A supported 1024-byte password can expand to 6144 JSON bytes. With a
// 64-byte username and compact JSON framing, the request needs at most 6237
// bytes. Keep raw-body admission separate from decoded password validation.
const maximumLoginBodyBytes int64 = 8 * 1024
const loginBodyReadTimeout = 5 * time.Second

var (
	corsMethods = []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete}
	corsHeaders = []string{
		"Authorization", "Content-Type", "Idempotency-Key", "If-Match", "If-None-Match", "X-CSRF-Token",
	}
)

type browserSessionContextKey struct{}

// SessionDependencies serve browser login, the session cookie, bearer and
// cookie authentication, CORS and the live event socket.
type SessionDependencies struct {
	Authentication *auth.Service
	BrowserOrigins auth.OriginPolicy
	// TrustedPeers resolves the login rate-limit client behind reverse proxies;
	// its zero value attributes every failure to the socket peer.
	TrustedPeers           auth.PeerPolicy
	InsecureLoopbackCookie bool
	Events                 *publicevents.Hub
}

type sessionHandlers struct {
	SessionDependencies
	tokenDigest [sha256.Size]byte
}

// cookieName names the session cookie: the __Host- cookie, or the
// separately named insecure cookie of loopback development (API-R18).
func (h *sessionHandlers) cookieName() string {
	if h.InsecureLoopbackCookie {
		return auth.LoopbackCookieName
	}
	return auth.SecureCookieName
}

func (h *sessionHandlers) login(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		handleError(w, errInvalidRequest)
		return
	}
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(time.Now().Add(loginBodyReadTimeout))
	var request loginRequest
	decodeErr := decodeJSONBounded(w, r, &request, maximumLoginBodyBytes)
	_ = controller.SetReadDeadline(time.Time{})
	if decodeErr != nil {
		handleError(w, decodeErr)
		return
	}
	peerIP, err := h.TrustedPeers.ClientIP(r.RemoteAddr, r.Header.Values("X-Forwarded-For"))
	if err != nil {
		handleError(w, errInvalidRequest)
		return
	}
	password := []byte(request.Password)
	request.Password = ""
	defer clear(password)
	result, err := h.Authentication.Login(request.Username, password, peerIP)
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
			writeError(w, apierror.RateLimited, "authentication attempts are temporarily limited")
			return
		}
		if errors.Is(err, auth.ErrInvalidCredentials) {
			w.Header().Set("WWW-Authenticate", "Session")
			writeError(w, apierror.Unauthorized, "credentials are invalid")
			return
		}
		handleError(w, err)
		return
	}
	h.setSessionCookie(w, result.CookieValue, result.Session.AbsoluteExpiresAt)
	result.CookieValue = ""
	writeJSON(w, http.StatusOK, authSession(result.Session))
}

func (h *sessionHandlers) getSession(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		writeSessionUnauthorized(w)
		return
	}
	writeJSON(w, http.StatusOK, authSession(session))
}

func (h *sessionHandlers) logout(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil || r.ContentLength > 0 || len(r.TransferEncoding) != 0 {
		handleError(w, errInvalidRequest)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		writeSessionUnauthorized(w)
		return
	}
	h.Authentication.Destroy(session.Handle)
	h.clearSessionCookie(w)
	w.WriteHeader(http.StatusNoContent)
}

func (h *sessionHandlers) authenticate(next http.Handler) http.Handler {
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
				writeBearerUnauthorized(w)
				return
			}
			candidate := strings.TrimPrefix(authorization[0], "Bearer ")
			candidateDigest := sha256.Sum256([]byte(candidate))
			if candidate == "" || len(candidate) > 4096 ||
				subtle.ConstantTimeCompare(candidateDigest[:], h.tokenDigest[:]) != 1 {
				writeBearerUnauthorized(w)
				return
			}
			principal := h.Authentication.Principal()
			next.ServeHTTP(w, r.WithContext(auth.WithPrincipal(r.Context(), principal)))
			return
		}
		cookieValue, err := h.sessionCookie(r)
		if err != nil {
			writeSessionUnauthorized(w)
			return
		}
		session, err := h.Authentication.Lookup(cookieValue)
		if err != nil {
			writeSessionUnauthorized(w)
			return
		}
		if requestIsUnsafe(r.Method) {
			origins := r.Header.Values("Origin")
			csrf := r.Header.Values("X-CSRF-Token")
			if len(origins) != 1 || !h.BrowserOrigins.Allows(origins[0]) || len(csrf) != 1 ||
				h.Authentication.ValidateCSRF(session, csrf[0]) != nil {
				writeError(w, apierror.Forbidden, "request origin or CSRF token is invalid")
				return
			}
		}
		session, err = h.Authentication.Accept(session.Handle)
		if err != nil {
			writeSessionUnauthorized(w)
			return
		}
		ctx := auth.WithPrincipal(r.Context(), session.Principal)
		ctx = context.WithValue(ctx, browserSessionContextKey{}, session)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (h *sessionHandlers) cors(next http.Handler, routes *http.ServeMux) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/auth/") {
			w.Header().Set("Cache-Control", "no-store")
		}
		origins := r.Header.Values("Origin")
		if len(origins) != 0 {
			setVary(w.Header(), "Origin")
		}
		if len(origins) > 1 || len(origins) == 1 && !h.BrowserOrigins.Allows(origins[0]) {
			writeError(w, apierror.Forbidden, "browser origin is not allowed")
			return
		}
		if len(origins) == 0 {
			if r.Method == http.MethodOptions || len(r.Header.Values("Access-Control-Request-Method")) != 0 {
				writeError(w, apierror.Forbidden, "browser origin is not allowed")
				return
			}
			next.ServeHTTP(w, r)
			return
		}
		origin := origins[0]
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set(
			"Access-Control-Expose-Headers",
			"X-Request-ID, X-Contractor-API-Version, ETag, Content-Disposition, Retry-After",
		)
		if r.Method != http.MethodOptions {
			next.ServeHTTP(w, r)
			return
		}
		if !validPreflight(r, routes) {
			writeError(w, apierror.Forbidden, "CORS preflight is not allowed")
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

func (h *sessionHandlers) sessionCookie(r *http.Request) (string, error) {
	configured := r.CookiesNamed(h.cookieName())
	if len(configured) != 1 || configured[0].Value == "" {
		return "", auth.ErrInvalidSession
	}
	return configured[0].Value, nil
}

func (h *sessionHandlers) setSessionCookie(w http.ResponseWriter, value string, absoluteExpiresAt time.Time) {
	http.SetCookie(w, &http.Cookie{
		Name: h.cookieName(), Value: value, Path: "/", Expires: absoluteExpiresAt,
		HttpOnly: true, Secure: !h.InsecureLoopbackCookie, SameSite: http.SameSiteLaxMode,
	})
}

func (h *sessionHandlers) clearSessionCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name: h.cookieName(), Value: "", Path: "/", Expires: time.Unix(1, 0).UTC(), MaxAge: -1,
		HttpOnly: true, Secure: !h.InsecureLoopbackCookie, SameSite: http.SameSiteLaxMode,
	})
}

func writeSessionUnauthorized(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", "Session")
	writeError(w, apierror.Unauthorized, "")
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
