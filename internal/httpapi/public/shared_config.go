package public

import (
	"fmt"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts/control"
)

// getSharedConfig exposes only the shared non-secret startup policy. Ordinary
// public authentication wraps this route; no Operations capability is needed.
func getSharedConfig(w http.ResponseWriter, r *http.Request) {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
		handleError(w, fmt.Errorf("%w: config accepts no body", errInvalidRequest))
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	config, err := control.DefaultSharedConfig()
	if err != nil {
		handleError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, config)
}
