package evaldomain

import "math"

func validateChart(v map[string]any) error {
	if err := validateSummary(asObject(v["experimentSummary"])); err != nil {
		return err
	}
	coverage := asObject(v["coverage"])
	if numeric(coverage["includedPairs"])+numeric(coverage["excludedPairs"]) != numeric(coverage["expectedPairs"]) {
		return Failure("eval_invalid")
	}
	allowed := map[string]bool{}
	switch v["chart"] {
	case "quality":
		allowed["quality"] = true
	case "tokens", "duration":
		allowed["bins"], allowed["distributions"] = true, true
		if v["chart"] == "tokens" && v["unit"] != "tokens" || v["chart"] == "duration" && v["unit"] != "milliseconds" {
			return Failure("eval_invalid")
		}
		bins := asRows(v["bins"])
		totals := map[string]float64{"a": 0, "b": 0}
		for i, raw := range bins {
			bin := asObject(raw)
			lo, hi := numeric(bin["lower"]), numeric(bin["upper"])
			if lo > hi || bin["upperInclusive"] != (i == len(bins)-1) {
				return Failure("eval_invalid")
			}
			if i > 0 && lo != numeric(asObject(bins[i-1])["upper"]) {
				return Failure("eval_invalid")
			}
			for arm, n := range asObject(bin["counts"]) {
				totals[arm] += numeric(n)
			}
		}
		for arm, raw := range asObject(v["distributions"]) {
			d := asObject(raw)
			n := numeric(d["count"])
			if n != numeric(coverage["includedPairs"]) || totals[arm] != n {
				return Failure("eval_invalid")
			}
			for _, key := range []string{"total", "p50", "p90"} {
				if (d[key] == nil) != (n == 0) {
					return Failure("eval_invalid")
				}
			}
			if n > 0 && (numeric(d["p50"]) > numeric(d["p90"]) || numeric(d["p90"]) > numeric(d["total"])) {
				return Failure("eval_invalid")
			}
		}
	case "pair-deltas":
		allowed["differences"], allowed["page"] = true, true
		if v["unit"] != "tokens" && v["unit"] != "milliseconds" {
			return Failure("eval_invalid")
		}
		if !uniqueRows(asRows(v["differences"]), "pairId") {
			return Failure("eval_member_conflict")
		}
		for _, raw := range asRows(v["differences"]) {
			d := asObject(raw)
			if math.Abs(numeric(d["b"])-numeric(d["a"])-numeric(d["difference"])) > 1e-9 {
				return Failure("eval_invalid")
			}
		}
	case "progress":
		allowed["points"], allowed["bucketMs"] = true, true
		previous := float64(-1)
		for _, raw := range asRows(v["points"]) {
			p := asObject(raw)
			elapsed := numeric(p["elapsedMs"])
			if elapsed <= previous {
				return Failure("eval_invalid")
			}
			previous = elapsed
		}
	}
	for _, key := range []string{"quality", "bins", "distributions", "differences", "page", "points", "bucketMs"} {
		if _, ok := v[key]; ok && !allowed[key] {
			return Failure("eval_invalid")
		}
	}
	if quality, exists := v["quality"]; exists {
		for _, arm := range asObject(quality) {
			if err := validateQuality(asObject(arm)); err != nil {
				return err
			}
		}
	}
	if page, exists := v["page"]; exists {
		return typedCheck(page, validatePage)
	}
	return nil
}
