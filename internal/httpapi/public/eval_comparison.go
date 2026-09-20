package public

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/grauwolf32/contractor/internal/evaldomain"
	"github.com/grauwolf32/contractor/internal/evalservice"
)

func (h *handler) evalSelectedPageRequest(r *http.Request) (evalservice.MemberPageParams, url.Values, error) {
	q, limit, err := evalQuery(r, "viewSnapshot", "filter", "variantId", "suiteId", "measurementScope", "binFilter")
	p := evalservice.MemberPageParams{OwnerID: principalUserID(r.Context()), ExperimentID: r.PathValue("id"), AfterOrdinal: -1, Limit: limit}
	if err != nil {
		return p, q, err
	}
	p.Snapshot, p.Filter, p.VariantID, p.SuiteID, p.MeasurementScope = q.Get("viewSnapshot"), q.Get("filter"), q.Get("variantId"), q.Get("suiteId"), q.Get("measurementScope")
	cursor, err := h.readEvalCursor(r, q, 1)
	if err != nil {
		return p, q, err
	}
	if len(cursor.Position) > 0 {
		p.AfterOrdinal, err = strconv.Atoi(cursor.Position[0])
		if err != nil || p.Snapshot != "" && p.Snapshot != cursor.Snapshot {
			return p, q, evaldomain.Failure("eval_invalid")
		}
		p.Snapshot = cursor.Snapshot
	}
	if q.Has("binFilter") {
		err = h.readEvalBin(r, q, &p)
	}
	return p, q, err
}

func (h *handler) listEvalPairs(w http.ResponseWriter, r *http.Request) {
	p, q, err := h.evalSelectedPageRequest(r)
	if err != nil {
		h.evalError(w, err)
		return
	}
	data, err := h.dependencies.Evals.Pairs(r.Context(), evalservice.PairPageParams{MemberPageParams: p})
	if err != nil {
		h.evalError(w, err)
		return
	}
	page, err := h.evalPage(r, q, data.HasMore, data.Snapshot, strconv.Itoa(data.LastOrdinal))
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, http.StatusOK, "PairPage", map[string]any{
		"viewSnapshot":      data.Snapshot,
		"freshness":         data.Freshness,
		"experimentSummary": data.Summary,
		"filteredCount":     data.FilteredCount,
		"items":             data.Items,
		"page":              page,
	})
}

func evalDetailRequest(r *http.Request, extra ...string) (evalservice.MemberPageParams, url.Values, error) {
	q, err := exactQuery(r.URL.RawQuery, append([]string{"viewSnapshot"}, extra...)...)
	p := evalservice.MemberPageParams{OwnerID: principalUserID(r.Context()), ExperimentID: r.PathValue("id"), Snapshot: q.Get("viewSnapshot")}
	if err != nil {
		return p, q, evaldomain.Failure("eval_invalid")
	}
	return p, q, nil
}

func (h *handler) getEvalPair(w http.ResponseWriter, r *http.Request) {
	p, _, err := evalDetailRequest(r)
	if err != nil {
		h.evalError(w, err)
		return
	}
	data, err := h.dependencies.Evals.Pair(r.Context(), p, r.PathValue("pairId"))
	if err != nil {
		h.evalError(w, err)
		return
	}
	h.evalJSON(w, http.StatusOK, "PairDetail", data)
}

func (h *handler) getEvalChart(w http.ResponseWriter, r *http.Request) {
	q, limit, err := evalQuery(r, "viewSnapshot", "suiteId", "measurementScope", "metric", "sort")
	if err != nil {
		h.evalError(w, err)
		return
	}
	chart := r.PathValue("chart")
	if chart != "pair-deltas" && (q.Has("limit") || q.Has("cursor") || q.Has("metric") || q.Has("sort")) || q.Has("sort") && !evalStringIn(q.Get("sort"), "frozen", "absolute") {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	p := evalservice.ChartParams{Chart: chart, PairPageParams: evalservice.PairPageParams{
		MemberPageParams: evalservice.MemberPageParams{
			OwnerID:          principalUserID(r.Context()),
			ExperimentID:     r.PathValue("id"),
			Snapshot:         q.Get("viewSnapshot"),
			SuiteID:          q.Get("suiteId"),
			MeasurementScope: q.Get("measurementScope"),
			Limit:            limit,
			AfterOrdinal:     -1,
		},
		Metric: q.Get("metric"), Absolute: q.Get("sort") == "absolute",
	}}
	if chart == "pair-deltas" {
		cursor, err := h.readEvalCursor(r, q, 2)
		if err != nil {
			h.evalError(w, err)
			return
		}
		if len(cursor.Position) > 0 {
			p.AfterOrdinal, err = strconv.Atoi(cursor.Position[0])
			difference, parseErr := strconv.ParseFloat(cursor.Position[1], 64)
			if err != nil || parseErr != nil || math.IsNaN(difference) || math.IsInf(difference, 0) || p.Snapshot != "" && p.Snapshot != cursor.Snapshot {
				h.evalError(w, evaldomain.Failure("eval_invalid"))
				return
			}
			p.AfterDifference = &difference
			p.Snapshot = cursor.Snapshot
		}
	}
	data, err := h.dependencies.Evals.Chart(r.Context(), p)
	if err != nil {
		h.evalError(w, err)
		return
	}
	if data.Bins != nil {
		for i := range *data.Bins {
			bin := &(*data.Bins)[i]
			bin.FilterToken, err = h.evalBinToken(r, data, *bin)
			if err != nil {
				h.evalError(w, err)
				return
			}
		}
	}
	// The page adapter supplies signed cursors; the service owns every chart value.
	response := struct {
		evalservice.ChartView
		Page *evalPageInfo `json:"page,omitempty"`
	}{ChartView: data}
	if chart == "pair-deltas" {
		difference := "0"
		if data.LastDifference != nil {
			difference = strconv.FormatFloat(*data.LastDifference, 'g', -1, 64)
		}
		page, err := h.evalPage(r, q, data.HasMore, data.Snapshot, strconv.Itoa(data.LastOrdinal), difference)
		if err != nil {
			h.evalError(w, err)
			return
		}
		response.Page = &page
	}
	h.evalJSON(w, http.StatusOK, "Chart", response)
}

func (h *handler) getEvalReport(w http.ResponseWriter, r *http.Request) {
	p, q, err := evalDetailRequest(r, "format")
	if err != nil {
		h.evalError(w, err)
		return
	}
	format := q.Get("format")
	if format != "" && !evalStringIn(format, "json", "markdown") {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	data, err := h.dependencies.Evals.Report(r.Context(), p)
	if err != nil {
		h.evalError(w, err)
		return
	}
	if format != "markdown" {
		h.evalJSON(w, http.StatusOK, "Report", data)
		return
	}
	raw, err := json.Marshal(data)
	if err != nil || evaldomain.Validate("Report", raw) != nil {
		h.evalError(w, evaldomain.Failure("eval_invalid"))
		return
	}
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="eval-report.md"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(evalMarkdownReport(data)))
}

func evalMarkdownReport(data evalservice.Report) string {
	var out strings.Builder
	conclusion := data.Summary.Conclusion
	if conclusion == "pass" {
		conclusion = "meets declared gates"
	}
	fmt.Fprintf(&out, "# Evaluation report\n\nExperiment: %s\n\nPlan: %s\n\nSnapshot: %s\n\nConclusion: %s\n\n", data.ExperimentID, data.PlanSHA256, data.Snapshot, conclusion)
	fmt.Fprintln(&out, "| Variant | Expected | Terminal | Scored | End-to-end passed |\n| --- | ---: | ---: | ---: | ---: |")
	for _, arm := range sortedEvalKeys(data.Summary.Counts) {
		c := data.Summary.Counts[arm]
		fmt.Fprintf(&out, "| %s | %d | %d | %d | %d |\n", arm, c.Expected, c.Terminal, c.Scored, c.EndToEndPassed)
	}
	fmt.Fprintf(&out, "\nPairs: %d. Complete quality pairs: %d. Complete token pairs: %d.\n", data.PairCount, data.Summary.CompleteQualityPairs, data.Summary.CompleteTokenPairs)
	if len(data.Sources) > 0 {
		fmt.Fprintln(&out, "\nSources:")
		for _, source := range data.Sources {
			// Source system/ID use the closed Id grammar, not free-text Markdown.
			fmt.Fprintf(&out, "- %s / %s\n", source.System, source.ID)
		}
	}
	return out.String()
}
