package public

import (
	"fmt"
	"mime"
	"net/http"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
)

// writeArtifactBytes serves stored artifact bytes under their recorded media
// type; the media type is caller-controlled, so browsers must neither sniff
// nor render the payload in the API origin.
func writeArtifactBytes(w http.ResponseWriter, result artifacts.ReadResult, filename string) {
	w.Header().Set("Content-Type", result.Payload.MediaType)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Security-Policy", "sandbox; default-src 'none'")
	disposition := "attachment"
	if filename != "" {
		disposition = mime.FormatMediaType(disposition, map[string]string{"filename": filename})
	}
	w.Header().Set("Content-Disposition", disposition)
	w.Header().Set("ETag", httpx.QuotedETag(result.Ref.Revision))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(result.Payload.Data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(result.Payload.Data)
}
