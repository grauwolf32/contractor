package public

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

func (h *operationsHandlers) getModelSelection(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
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
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
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
	values := r.Header.Values("If-Match")
	if len(values) != 1 || len(r.Header.Values("If-None-Match")) != 0 {
		return 0, "", invalid
	}
	if len(values[0]) < 3 || values[0][0] != '"' || values[0][len(values[0])-1] != '"' {
		return 0, "", invalid
	}
	value := values[0][1 : len(values[0])-1]
	revision, digest, ok := strings.Cut(value, ":")
	if !ok || len(digest) != 64 || strings.Trim(digest, "0123456789abcdef") != "" {
		return 0, "", invalid
	}
	parsed, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || parsed == 0 || strconv.FormatUint(parsed, 10) != revision {
		return 0, "", invalid
	}
	return parsed, "sha256:" + digest, nil
}
