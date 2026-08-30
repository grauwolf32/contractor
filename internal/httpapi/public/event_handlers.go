package public

import (
	"errors"
	"net/http"
	"strconv"

	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

func (h *handler) connectEventsWebSocket(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		h.writeSessionUnauthorized(w)
		return
	}
	err := h.dependencies.Events.Serve(w, r, session)
	switch {
	case err == nil:
		return
	case errors.Is(err, publicevents.ErrInvalidOrigin):
		h.writeError(w, http.StatusForbidden, "forbidden", "browser origin is not allowed", false)
	case errors.Is(err, publicevents.ErrSession):
		h.writeSessionUnauthorized(w)
	case errors.Is(err, publicevents.ErrSocketLimit):
		w.Header().Set("Retry-After", strconv.Itoa(1))
		h.writeError(w, http.StatusTooManyRequests, "rate_limited", "browser session socket limit reached", true)
	case errors.Is(err, publicevents.ErrInvalidHandshake):
		h.writeError(w, http.StatusBadRequest, "invalid_request", "WebSocket handshake is invalid", false)
	default:
		h.handleError(w, err)
	}
}
