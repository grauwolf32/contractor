package public

import (
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/helpers/requestid"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// Both Scheduler settings routes are authorized by requireOperationsRoutes.

func (h *operationsHandlers) getSchedulerSettings(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	settings, err := h.SchedulerSettings.GetSchedulerSettings(r.Context())
	if err != nil {
		return err
	}
	h.writeSchedulerSettings(w, settings)
	return nil
}

func (h *operationsHandlers) putSchedulerSettings(w http.ResponseWriter, r *http.Request) error {
	w.Header().Set("Cache-Control", "no-store")
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		return err
	}
	if requireJSONMediaType(r) != nil {
		writeError(w, apierror.UnsupportedMediaType, "")
		return nil
	}
	expectedRevision, err := schedulerSettingsPrecondition(r)
	if err != nil {
		return err
	}
	var request updateSchedulerSettingsRequest
	if err := decodeJSON(w, r, &request); err != nil {
		return err
	}
	if request.MaxConcurrentRuns < settingsstore.MinimumConcurrentRuns ||
		request.MaxConcurrentRuns > settingsstore.MaximumConcurrentRuns ||
		(request.MaxConcurrentToolRuns != nil &&
			(*request.MaxConcurrentToolRuns < settingsstore.MinimumConcurrentToolRuns ||
				*request.MaxConcurrentToolRuns > settingsstore.MaximumConcurrentToolRuns)) {
		return settingsstore.ErrInvalid
	}
	settings, err := h.SchedulerSettings.UpdateSchedulerSettings(
		r.Context(), settingsstore.UpdateSchedulerSettingsParams{
			MaxConcurrentRuns:     request.MaxConcurrentRuns,
			MaxConcurrentToolRuns: request.MaxConcurrentToolRuns,
			ExpectedRevision:      expectedRevision,
		},
	)
	if err != nil {
		return err
	}
	changed := settings.Revision != expectedRevision
	if changed {
		h.RunNotifier.Wake()
		if err := h.OperationsInvalidator.InvalidateOperations(
			controlplane.OperationsSchedulerSettings, "",
		); err != nil {
			return err
		}
		h.auditSchedulerSettingsMutation(r, settings)
	}
	h.writeSchedulerSettings(w, settings)
	return nil
}

func (h *operationsHandlers) writeSchedulerSettings(w http.ResponseWriter, settings settingsstore.SchedulerSettings) {
	revision := strconv.FormatUint(settings.Revision, 10)
	w.Header().Set("ETag", strconv.Quote(revision))
	writeJSON(w, http.StatusOK, schedulerSettingsResponse{
		MaxConcurrentRuns:     settings.MaxConcurrentRuns,
		MaxConcurrentToolRuns: settings.MaxConcurrentToolRuns,
		Revision:              revision,
		UpdatedAt:             settings.UpdatedAt.UTC(),
	})
}

func schedulerSettingsPrecondition(r *http.Request) (uint64, error) {
	return requiredRevision(r, "Scheduler settings replacement")
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
		"max_concurrent_tool_runs", settings.MaxConcurrentToolRuns,
		"revision", settings.Revision,
	)
}
