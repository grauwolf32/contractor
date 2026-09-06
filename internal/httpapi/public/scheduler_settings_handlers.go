package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

func (h *handler) getSchedulerSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if h.rejectHead(w, r) || !h.requireOperationsCapability(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	settings, err := h.dependencies.SchedulerSettings.GetSchedulerSettings(r.Context())
	if err != nil {
		h.handleError(w, err)
		return
	}
	h.writeSchedulerSettings(w, settings)
}

func (h *handler) putSchedulerSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if !h.requireOperationsCapability(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		h.writeError(
			w, http.StatusUnsupportedMediaType, "unsupported_media_type",
			"Content-Type must be application/json", false,
		)
		return
	}
	expectedRevision, err := schedulerSettingsPrecondition(r)
	if err != nil {
		h.handleError(w, err)
		return
	}
	var request updateSchedulerSettingsRequest
	if err := decodeJSON(w, r, &request); err != nil {
		h.handleError(w, err)
		return
	}
	if request.MaxConcurrentRuns < settingsstore.MinimumConcurrentRuns ||
		request.MaxConcurrentRuns > settingsstore.MaximumConcurrentRuns {
		h.handleError(w, settingsstore.ErrInvalid)
		return
	}
	settings, err := h.dependencies.SchedulerSettings.UpdateSchedulerSettings(
		r.Context(), settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: request.MaxConcurrentRuns,
			ExpectedRevision:  expectedRevision,
		},
	)
	if err != nil {
		h.handleError(w, err)
		return
	}
	changed := settings.Revision != expectedRevision
	if changed {
		if h.dependencies.RunNotifier != nil {
			h.dependencies.RunNotifier.Wake()
		}
		if err := h.dependencies.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsSchedulerSettings, "",
		); err != nil {
			h.handleError(w, err)
			return
		}
		h.auditSchedulerSettingsMutation(r, settings)
	}
	h.writeSchedulerSettings(w, settings)
}

func (h *handler) writeSchedulerSettings(w http.ResponseWriter, settings settingsstore.SchedulerSettings) {
	revision := strconv.FormatUint(settings.Revision, 10)
	w.Header().Set("ETag", strconv.Quote(revision))
	writeJSON(w, http.StatusOK, schedulerSettingsResponse{
		MaxConcurrentRuns: settings.MaxConcurrentRuns,
		Revision:          revision,
		UpdatedAt:         settings.UpdatedAt.UTC(),
	})
}

func schedulerSettingsPrecondition(r *http.Request) (uint64, error) {
	if len(r.Header.Values("If-None-Match")) != 0 || len(r.Header.Values("If-Match")) != 1 {
		return 0, fmt.Errorf("%w: Scheduler settings replacement requires one If-Match", errInvalidRequest)
	}
	return parseRuntimeRevisionETag(r.Header.Values("If-Match")[0])
}

func (h *handler) requireOperationsCapability(w http.ResponseWriter, r *http.Request) bool {
	principal, ok := auth.PrincipalFromContext(r.Context())
	if !ok {
		h.writeBearerUnauthorized(w)
		return false
	}
	for _, capability := range principal.Capabilities {
		if capability == auth.CapabilityOperations {
			return true
		}
	}
	h.writeError(w, http.StatusForbidden, "forbidden", "Operations capability is required", false)
	return false
}

func (h *handler) auditSchedulerSettingsMutation(
	r *http.Request,
	settings settingsstore.SchedulerSettings,
) {
	if h.dependencies.Logger == nil {
		return
	}
	h.dependencies.Logger.InfoContext(
		r.Context(), "Scheduler Operations mutation",
		"audit_action", "scheduler_settings.replace",
		"actor_id", principalUserID(r.Context()),
		"request_id", requestid.From(r.Context()),
		"max_concurrent_runs", settings.MaxConcurrentRuns,
		"revision", settings.Revision,
	)
}
