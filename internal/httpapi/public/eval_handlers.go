package public

import (
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
	"github.com/grauwolf32/contractor/internal/evalstore"
)

func evalScope(r *http.Request) evalstore.Scope {
	return evalstore.Scope{OwnerID: principalUserID(r.Context()), ProjectID: r.PathValue("projectId")}
}
func (h *handler) evalExactQuery(w http.ResponseWriter, r *http.Request) bool {
	if _, err := exactQuery(r.URL.RawQuery); err != nil {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return false
	}
	return true
}
func (h *handler) evalCapabilities(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r, "kind")
	if err != nil {
		h.evalError(w, err)
		return
	}
	if q.Has("kind") && !evalStringIn(q.Get("kind"), "workflow", "audit") {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		h.evalError(w, err)
		return
	}
	type binding struct {
		Kind      string  `json:"kind"`
		Selector  string  `json:"selector"`
		Available bool    `json:"available"`
		Reason    *string `json:"reason"`
	}
	bindings := []binding{}
	if q.Get("kind") != "audit" {
		for _, workflow := range h.dependencies.Config.Workflows() {
			bindings = append(bindings, binding{Kind: "workflow", Selector: workflow.Ref.Name + "@" + workflow.Ref.Version, Available: true})
		}
	}
	if q.Get("kind") != "workflow" {
		for _, profile := range h.dependencies.Audits.Profiles() {
			var reason *string
			if !profile.Compatibility.ServerCompatible {
				message := "This AuditProfile requires unsupported Server capabilities."
				reason = &message
			}
			bindings = append(bindings, binding{Kind: "audit", Selector: profile.Profile.Ref.Name + "@" + profile.Profile.Ref.Version, Available: profile.Compatibility.ServerCompatible, Reason: reason})
		}
	}
	sort.Slice(bindings, func(i, j int) bool {
		return bindings[i].Kind+":"+bindings[i].Selector < bindings[j].Kind+":"+bindings[j].Selector
	})
	raw, err := json.Marshal(bindings)
	if err != nil {
		h.handleError(w, err)
		return
	}
	snapshot := evaldomain.Digest(raw)
	if cursor.Snapshot != "" && cursor.Snapshot != snapshot {
		h.evalError(w, evaldomain.Failure("eval_view_changed"))
		return
	}
	after := ""
	if len(cursor.Position) > 0 {
		after = cursor.Position[0]
	}
	items := []binding{}
	more := false
	last := ""
	for _, b := range bindings {
		key := b.Kind + ":" + b.Selector
		if key <= after {
			continue
		}
		if len(items) == limit {
			more = true
			break
		}
		items = append(items, b)
		last = key
	}
	page, err := h.evalPage(r, q, more, snapshot, last)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, 200, "Capabilities", map[string]any{"controlModes": []string{"server", "external"}, "executionKinds": []string{"workflow", "audit"}, "checks": []any{map[string]any{"evaluator": "human-review@1", "implementationSha256": evalservice.HumanPolicySHA256(), "available": true, "reason": nil}}, "importVersions": []string{"dataset@1", "contractor.eval-registration@1"}, "bindings": items, "page": page})
}
func (h *handler) importEvalDataset(w http.ResponseWriter, r *http.Request) {
	doc, m, ok := h.evalBody(w, r, "DatasetInput", false)
	if !ok {
		return
	}
	receipt, err := h.dependencies.Evals.PutDataset(r.Context(), evalScope(r), doc, m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	ref, err := receipt.Dataset()
	if err != nil {
		h.handleError(w, err)
		return
	}
	data, err := h.dependencies.Evals.Dataset(r.Context(), evalScope(r), ref.DatasetID, ref.DatasetRevision)
	if err != nil {
		h.evalError(w, err)
		return
	}
	if receipt.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(data.Metadata.Revision))
	h.evalJSON(w, 201, "Dataset", data.Metadata)
}
func (h *handler) listEvalDatasets(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r)
	if err != nil {
		h.evalError(w, err)
		return
	}
	cursor, err := h.readEvalCursor(r, q, 2)
	if err != nil {
		h.evalError(w, err)
		return
	}
	afterID, afterRevision := "", ""
	if len(cursor.Position) > 0 {
		afterID, afterRevision = cursor.Position[0], cursor.Position[1]
	}
	revision, err := cursorRevision(cursor)
	if err != nil {
		h.evalError(w, err)
		return
	}
	data, err := h.dependencies.Evals.Datasets(r.Context(), evalScope(r), afterID, afterRevision, limit, revision)
	if err != nil {
		h.evalError(w, err)
		return
	}
	if len(data.Items) > 0 {
		last := data.Items[len(data.Items)-1]
		afterID, afterRevision = last.DatasetID, last.Revision
	}
	page, err := h.evalPage(r, q, data.HasMore, strconv.FormatInt(data.Revision, 10), afterID, afterRevision)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, 200, "DatasetPage", map[string]any{"items": data.Items, "page": page})
}
func (h *handler) listEvalCases(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r)
	if err != nil {
		h.evalError(w, err)
		return
	}
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		h.evalError(w, err)
		return
	}
	data, err := h.dependencies.Evals.Dataset(r.Context(), evalScope(r), r.PathValue("datasetId"), r.PathValue("revision"))
	if err != nil {
		h.evalError(w, err)
		return
	}
	if cursor.Snapshot != "" && cursor.Snapshot != data.Metadata.VisibleSHA256 {
		h.evalError(w, evaldomain.Failure("eval_view_changed"))
		return
	}
	var input evaldomain.DatasetInput
	if err = evaldomain.DecodeInto("DatasetInput", data.Document.Bytes(), &input); err != nil {
		h.evalError(w, err)
		return
	}
	offset := 0
	if len(cursor.Position) > 0 {
		offset, err = strconv.Atoi(cursor.Position[0])
		if err != nil || offset < 0 || offset > len(input.Cases) {
			h.evalError(w, evaldomain.Failure("eval_invalid"))
			return
		}
	}
	end := min(offset+limit, len(input.Cases))
	items := input.Cases[offset:end]
	page, err := h.evalPage(r, q, end < len(input.Cases), data.Metadata.VisibleSHA256, strconv.Itoa(end))
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, 200, "CasePage", map[string]any{"items": items, "page": page})
}
func (h *handler) createEvalExperiment(w http.ResponseWriter, r *http.Request) {
	doc, m, ok := h.evalBody(w, r, "CreateExperiment", false)
	if !ok {
		return
	}
	receipt, err := h.dependencies.Evals.Create(r.Context(), evalScope(r), doc, m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalReceipt(w, 201, receipt)
}
func (h *handler) getEvalExperiment(w http.ResponseWriter, r *http.Request) {
	if !h.evalExactQuery(w, r) {
		return
	}
	e, err := h.dependencies.Evals.Get(r.Context(), principalUserID(r.Context()), r.PathValue("id"))
	if err != nil {
		h.evalError(w, err)
		return
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(e.Revision, 10)))
	h.evalJSON(w, 200, "Experiment", e)
}
func (h *handler) listEvalExperiments(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r, "projectId", "state", "datasetId", "controlMode")
	if err != nil {
		h.evalError(w, err)
		return
	}
	if (q.Has("state") && !evalStringIn(q.Get("state"), "draft", "preparing", "ready", "running", "settling", "finished", "pausing", "paused", "cancelling", "cancelled", "interrupted")) || (q.Has("controlMode") && !evalStringIn(q.Get("controlMode"), "server", "external")) || q.Has("projectId") && q.Get("projectId") == "" || q.Has("datasetId") && q.Get("datasetId") == "" {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	cursor, err := h.readEvalCursor(r, q, 2)
	if err != nil {
		h.evalError(w, err)
		return
	}
	revision, err := cursorRevision(cursor)
	if err != nil {
		h.evalError(w, err)
		return
	}
	p := evalstore.SummaryPageParams{ListParams: evalstore.ListParams{OwnerID: principalUserID(r.Context()), ProjectID: q.Get("projectId"), State: q.Get("state"), DatasetID: q.Get("datasetId"), ControlMode: q.Get("controlMode"), Limit: limit, Revision: revision}}
	if len(cursor.Position) > 0 {
		at, err := time.Parse(time.RFC3339Nano, cursor.Position[0])
		if err != nil {
			h.evalError(w, evaldomain.Failure("eval_invalid"))
			return
		}
		p.AfterTime = &at
		p.AfterID = cursor.Position[1]
	}
	data, err := h.dependencies.Evals.List(r.Context(), p)
	if err != nil {
		h.evalError(w, err)
		return
	}
	lastTime, lastID := "", ""
	if len(data.Items) > 0 {
		last := data.Items[len(data.Items)-1]
		lastTime, lastID = last.UpdatedAt.Format(time.RFC3339Nano), last.ID
	}
	page, err := h.evalPage(r, q, data.HasMore, strconv.FormatInt(data.Revision, 10), lastTime, lastID)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, 200, "ExperimentPage", map[string]any{"items": data.Items, "page": page})
}
func evalStringIn(value string, options ...string) bool {
	for _, option := range options {
		if value == option {
			return true
		}
	}
	return false
}
func (h *handler) updateEvalDraft(w http.ResponseWriter, r *http.Request) {
	doc, m, ok := h.evalBody(w, r, "DraftUpdate", true)
	if !ok {
		return
	}
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), r.PathValue("id"), "draft-update", m.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.UpdateDraft(r.Context(), scope, r.PathValue("id"), doc, m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalReceipt(w, 200, receipt)
}
func (h *handler) deleteEvalExperiment(w http.ResponseWriter, r *http.Request) {
	_, m, ok := h.evalBody(w, r, "Delete", true)
	if !ok {
		return
	}
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), r.PathValue("id"), "delete", m.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.Delete(r.Context(), scope, r.PathValue("id"), m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalReceipt(w, 202, receipt)
}
func (h *handler) commandEvalExperiment(w http.ResponseWriter, r *http.Request) {
	doc, m, ok := h.evalBody(w, r, "Command", true)
	if !ok {
		return
	}
	var command evaldomain.Command
	if err := json.Unmarshal(doc.Bytes(), &command); err != nil {
		h.evalError(w, err)
		return
	}
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), r.PathValue("id"), "command", m.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.Command(r.Context(), scope, r.PathValue("id"), command, m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	if command.Kind == "duplicate" {
		h.evalReceipt(w, 201, receipt)
		return
	}
	ref, err := receipt.Command()
	if err != nil {
		h.handleError(w, err)
		return
	}
	if receipt.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	w.Header().Set("ETag", strconv.Quote(strconv.FormatInt(ref.ExperimentRevision, 10)))
	var digest *string
	if command.PlanSHA256 != "" {
		digest = &command.PlanSHA256
	}
	h.evalJSON(w, 202, "CommandReceipt", map[string]any{"commandId": ref.CommandID, "experimentId": r.PathValue("id"), "kind": command.Kind, "state": ref.State, "experimentRevision": ref.ExperimentRevision, "planSha256": digest, "diagnostics": []any{}})
	if h.dependencies.EvalNotifier != nil {
		h.dependencies.EvalNotifier.Wake()
	}
}
func (h *handler) getEvalCommand(w http.ResponseWriter, r *http.Request) {
	if !h.evalExactQuery(w, r) {
		return
	}
	record, digest, err := h.dependencies.Evals.GetCommand(r.Context(), principalUserID(r.Context()), r.PathValue("id"), r.PathValue("commandId"))
	if err != nil {
		h.evalError(w, err)
		return
	}
	state := record.State
	if state == "succeeded" {
		state = "completed"
	}
	diagnostics := []json.RawMessage{}
	if len(record.Diagnostic) > 0 {
		diagnostics = append(diagnostics, record.Diagnostic)
	}
	h.evalJSON(w, 200, "CommandReceipt", map[string]any{"commandId": record.ID, "experimentId": record.ExperimentID, "kind": record.Kind, "state": state, "experimentRevision": record.Revision, "planSha256": digest, "diagnostics": diagnostics})
}
func (h *handler) submitEvalMember(w http.ResponseWriter, r *http.Request) {
	doc, m, ok := h.evalBody(w, r, "Submission", false)
	if !ok {
		return
	}
	var request evaldomain.Submission
	if err := json.Unmarshal(doc.Bytes(), &request); err != nil {
		h.evalError(w, err)
		return
	}
	scope, err := h.dependencies.Evals.ScopeForMutation(r.Context(), principalUserID(r.Context()), r.PathValue("id")+":"+r.PathValue("memberId"), "submission", m.Key)
	if err != nil {
		h.evalError(w, err)
		return
	}
	receipt, err := h.dependencies.Evals.Submit(r.Context(), scope, r.PathValue("id"), r.PathValue("memberId"), request, m)
	if err != nil {
		h.evalError(w, err)
		return
	}
	ref, err := receipt.Submission()
	if err != nil {
		h.handleError(w, err)
		return
	}
	state := "accepted"
	if ref.State == "rejected" {
		state = "rejected"
	}
	if receipt.Replayed {
		w.Header().Set("Idempotency-Replayed", "true")
	}
	h.evalJSON(w, 202, "SubmissionReceipt", map[string]any{"memberId": r.PathValue("memberId"), "planSha256": request.PlanSHA256, "state": state, "execution": nil})
	if h.dependencies.EvalNotifier != nil {
		h.dependencies.EvalNotifier.Wake()
	}
}
func (h *handler) listEvalMembers(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r, "viewSnapshot", "filter", "variantId")
	if err != nil {
		h.evalError(w, err)
		return
	}
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		h.evalError(w, err)
		return
	}
	p := evalservice.MemberPageParams{OwnerID: principalUserID(r.Context()), ExperimentID: r.PathValue("id"), Snapshot: q.Get("viewSnapshot"), Filter: q.Get("filter"), VariantID: q.Get("variantId"), AfterOrdinal: -1, Limit: limit}
	if len(cursor.Position) > 0 {
		p.AfterOrdinal, err = strconv.Atoi(cursor.Position[0])
		if err != nil || p.Snapshot != "" && p.Snapshot != cursor.Snapshot {
			h.evalError(w, evaldomain.Failure("eval_invalid"))
			return
		}
		p.Snapshot = cursor.Snapshot
	}
	data, err := h.dependencies.Evals.Members(r.Context(), p)
	if err != nil {
		h.evalError(w, err)
		return
	}
	page, err := h.evalPage(r, q, data.HasMore, data.Snapshot, strconv.Itoa(data.LastOrdinal))
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, 200, "MemberPage", map[string]any{"viewSnapshot": data.Snapshot, "freshness": "current", "experimentSummary": data.Summary, "filteredCount": data.FilteredCount, "items": data.Items, "page": page})
}
