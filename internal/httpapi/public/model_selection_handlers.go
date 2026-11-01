package public

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

func (h *operationsHandlers) getModelSelection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if h.ModelSelection == nil {
		writeJSON(w, http.StatusOK, config.ModelSelectionInfo{Models: []config.ModelChoice{}})
		return
	}
	value, err := h.ModelSelection.GetModelSelection(r.Context())
	if err != nil {
		handleError(w, err)
		return
	}
	writeModelSelection(w, value)
}

func (h *operationsHandlers) putModelSelection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	if requireJSONMediaType(r) != nil {
		writeError(w, apierror.UnsupportedMediaType, "")
		return
	}
	revision, digest, err := modelSelectionPrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request settingsstore.ModelSelectionValue
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	if request.Validate() != nil || h.ModelSelection == nil {
		handleError(w, settingsstore.ErrModelSelectionInvalid)
		return
	}
	value, err := h.ModelSelection.ReplaceModelSelection(r.Context(), request, revision, digest)
	if err != nil {
		handleError(w, err)
		return
	}
	if value.Revision != strconv.FormatUint(revision, 10) {
		if err := h.OperationsInvalidator.InvalidateOperations(controlplane.OperationsModelSelection, ""); err != nil {
			handleError(w, err)
			return
		}
		h.Logger.InfoContext(r.Context(), "Model selection Operations mutation", "audit_action", "model_selection.replace", "actor_id", principalUserID(r.Context()), "request_id", requestid.From(r.Context()), "revision", value.Revision, "default_model", request.DefaultModel)
	}
	writeModelSelection(w, value)
}

func writeModelSelection(w http.ResponseWriter, value config.ModelSelectionInfo) {
	if value.Configured {
		w.Header().Set("ETag", strconv.Quote(value.Revision+":"+strings.TrimPrefix(value.CatalogDigest, "sha256:")))
	}
	writeJSON(w, http.StatusOK, value)
}

func modelSelectionPrecondition(r *http.Request) (uint64, string, error) {
	invalid := fmt.Errorf("%w: model selection replacement requires one current strong If-Match", errInvalidRequest)
	raw, err := httpx.RequiredIfMatch(r.Header)
	if err != nil {
		return 0, "", invalid
	}
	value, err := httpx.ParseStrongETag(raw)
	if err != nil {
		return 0, "", invalid
	}
	revision, digest, ok := strings.Cut(value, ":")
	if !ok || len(digest) != 64 || strings.Trim(digest, "0123456789abcdef") != "" {
		return 0, "", invalid
	}
	parsed, err := httpx.ParseRevision(revision)
	if err != nil || parsed == 0 {
		return 0, "", invalid
	}
	return parsed, "sha256:" + digest, nil
}
