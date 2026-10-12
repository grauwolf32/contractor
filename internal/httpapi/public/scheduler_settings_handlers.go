package public

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/requestid"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// Both Scheduler settings routes are authorized by requireOperationsRoutes.

func (h *operationsHandlers) getSchedulerSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if rejectHead(w, r) {
		return
	}
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	settings, err := h.SchedulerSettings.GetSchedulerSettings(r.Context())
	if err != nil {
		handleError(w, err)
		return
	}
	h.writeSchedulerSettings(w, settings)
}

func (h *operationsHandlers) putSchedulerSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		handleError(w, err)
		return
	}
	mediaType, err := requestMediaType(r)
	if err != nil || mediaType != "application/json" {
		writeError(w, apierror.UnsupportedMediaType, "")
		return
	}
	expectedRevision, err := schedulerSettingsPrecondition(r)
	if err != nil {
		handleError(w, err)
		return
	}
	var request updateSchedulerSettingsRequest
	if err := decodeJSON(w, r, &request); err != nil {
		handleError(w, err)
		return
	}
	if request.MaxConcurrentRuns < settingsstore.MinimumConcurrentRuns ||
		request.MaxConcurrentRuns > settingsstore.MaximumConcurrentRuns {
		handleError(w, settingsstore.ErrInvalid)
		return
	}
	settings, err := h.SchedulerSettings.UpdateSchedulerSettings(
		r.Context(), settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns: request.MaxConcurrentRuns,
			ExpectedRevision:  expectedRevision,
		},
	)
	if err != nil {
		handleError(w, err)
		return
	}
	changed := settings.Revision != expectedRevision
	if changed {
		h.RunNotifier.Wake()
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsSchedulerSettings, "",
		); err != nil {
			handleError(w, err)
			return
		}
		h.auditSchedulerSettingsMutation(r, settings)
	}
	h.writeSchedulerSettings(w, settings)
}

func (h *operationsHandlers) writeSchedulerSettings(w http.ResponseWriter, settings settingsstore.SchedulerSettings) {
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

func (h *operationsHandlers) auditSchedulerSettingsMutation(
	r *http.Request,
	settings settingsstore.SchedulerSettings,
) {
	h.Logger.InfoContext(
		r.Context(), "Scheduler Operations mutation",
		"audit_action", "scheduler_settings.replace",
		"actor_id", principalUserID(r.Context()),
		"request_id", requestid.From(r.Context()),
		"max_concurrent_runs", settings.MaxConcurrentRuns,
		"revision", settings.Revision,
	)
}
