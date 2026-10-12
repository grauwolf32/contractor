package public

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/apierror"
	publicevents "github.com/grauwolf32/contractor/internal/httpapi/public/events"
)

func (h *sessionHandlers) connectEventsWebSocket(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	session, ok := browserSessionFromContext(r.Context())
	if !ok {
		writeSessionUnauthorized(w)
		return
	}
	err := h.Events.Serve(w, r, session)
	switch {
	case err == nil:
		return
	case errors.Is(err, publicevents.ErrInvalidOrigin):
		writeError(w, apierror.Forbidden, "browser origin is not allowed")
	case errors.Is(err, publicevents.ErrSession):
		writeSessionUnauthorized(w)
	case errors.Is(err, publicevents.ErrSocketLimit):
		w.Header().Set("Retry-After", strconv.Itoa(1))
		writeError(w, apierror.RateLimited, "browser session socket limit reached")
	case errors.Is(err, publicevents.ErrInvalidHandshake):
		writeError(w, apierror.InvalidRequest, "WebSocket handshake is invalid")
	default:
		handleError(w, err)
	}
}
