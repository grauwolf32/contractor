package public

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"net/http"
)

// writeArtifactBytes serves stored artifact bytes under their recorded media
// type; the media type is caller-controlled, so browsers must neither sniff
// nor render the payload in the API origin.
func writeArtifactBytes(w http.ResponseWriter, result artifacts.ReadResult) {
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "sandbox; default-src 'none'")
	w.Header().Set("Content-Disposition", "attachment")
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}
