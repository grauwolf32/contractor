package sarif

import (
	"bytes"
	"math"
	"net/url"
	"strings"

	"github.com/grauwolf32/contractor/internal/findinganalysis"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func obj(value any) object  { v, _ := value.(map[string]any); return v }
func array(value any) []any { v, _ := value.([]any); return v }
func str(value any) string  { v, _ := value.(string); return v }
func index(value any, size int) (int, bool) {
	v, ok := value.(float64)
	return int(v), ok && v >= 0 && v < float64(size) && math.Trunc(v) == v
}
func same(a, b any) bool {
	x, e := strictjson.Canonical(a)
	y, f := strictjson.Canonical(b)
	return e == nil && f == nil && bytes.Equal(x, y)
}
func contractor(value object) object { return obj(obj(value["properties"])["contractor"]) }

// The official schema cannot express matching indices, exact source identity,
// independent triage or the absence of a fabricated edge across a native gap.
func validateSemantics(document object, snapshot findingexport.Snapshot) error {
	fail := findingexport.ErrInvalid
	runs := array(document["runs"])
	if len(runs) != 1 {
		return fail
	}
	run := obj(runs[0])
	driver := obj(obj(run["tool"])["driver"])
	if str(driver["name"]) != "Contractor" || str(driver["version"]) != snapshot.ExporterVersion || str(run["columnKind"]) != "unicodeCodePoints" {
		return fail
	}
	groups, err := snapshot.ResultGroups()
	if err != nil {
		return err
	}
	results := array(run["results"])
	if len(results) != len(groups) {
		return fail
	}
	rp := contractor(run)
	if str(rp["audit_id"]) != snapshot.AuditID || rp["include_duplicates"] != snapshot.Request.IncludeDuplicates || rp["selected_count"] != float64(len(snapshot.Results)) || rp["result_count"] != float64(len(groups)) {
		return fail
	}
	receipts := map[string]findingexport.Receipt{}
	findings := map[string]findingexport.Result{}
	for _, r := range snapshot.Receipts {
		receipts[r.ReceiptID] = r
	}
	for _, r := range snapshot.Results {
		findings[r.FindingID] = r
	}
	rules := array(driver["rules"])
	seenRules := map[string]bool{}
	for _, v := range rules {
		rule := obj(v)
		id := str(rule["id"])
		if id == "" || seenRules[id] {
			return fail
		}
		seenRules[id] = true
		if _, exists := obj(rule["properties"])["security-severity"]; exists {
			return fail
		}
	}
	files := array(run["artifacts"])
	bases := obj(run["originalUriBaseIds"])
	expectedFiles := map[string]string{}
	expectedRoots := map[string]bool{}
	expectedGit := map[string]findingexport.GitSource{}
	for _, group := range groups {
		receipt := receipts[group.Result.PresentationReceiptID]
		if receipt.Analysis == nil {
			continue
		}
		sources := map[string]findinganalysis.ResolvedSource{}
		for _, s := range receipt.Analysis.Sources {
			sources[s.ID] = s
			expectedRoots[baseID(s.RootKey)] = true
			if str(obj(bases[baseID(s.RootKey)])["uri"]) != "contractor-source://"+s.RootKey+"/" {
				return fail
			}
			if s.Kind == "archive" && s.Document.MediaType == "application/zip" {
				for _, e := range receipt.Evidence {
					if e.Original.ID == s.Document.ID && e.Git != nil {
						expectedGit[s.RootKey] = *e.Git
					}
				}
			}
		}
		for _, l := range receipt.Analysis.Analysis.Locations {
			s := sources[l.SourceID]
			expectedFiles[baseID(s.RootKey)+"\x00"+l.Path] = strings.TrimPrefix(l.FileDigest, "sha256:")
		}
	}
	if len(bases) != len(expectedRoots) || len(files) != len(expectedFiles) {
		return fail
	}
	actualFiles := map[string]bool{}
	for _, value := range files {
		file := obj(value)
		key, ok := artifactIdentity(obj(file["location"]), bases)
		expected, exists := expectedFiles[key]
		if !ok || !exists || actualFiles[key] || str(obj(file["hashes"])["sha-256"]) != expected {
			return fail
		}
		actualFiles[key] = true
	}
	vcs := array(run["versionControlProvenance"])
	if len(vcs) != len(expectedGit) {
		return fail
	}
	seenGit := map[string]bool{}
	for _, value := range vcs {
		v := obj(value)
		root := str(contractor(v)["source_root"])
		expected, ok := expectedGit[root]
		if !ok || seenGit[root] || str(v["repositoryUri"]) != expected.RepositoryURI || str(v["revisionId"]) != expected.Commit || str(obj(v["mappedTo"])["uri"]) != "contractor-source://"+root+"/" {
			return fail
		}
		seenGit[root] = true
	}
	taxonomies := array(run["taxonomies"])
	if _, exists := driver["supportedTaxonomies"]; exists {
		return fail
	}
	for _, value := range taxonomies {
		taxonomy := obj(value)
		name, version, found := strings.Cut(str(taxonomy["name"]), "@")
		if !found || str(taxonomy["version"]) != version || str(contractor(taxonomy)["scheme"]) != name || taxonomy["isComprehensive"] != false {
			return fail
		}
	}
	for i, value := range results {
		item := obj(value)
		group := groups[i]
		native := contractor(item)
		receipt := receipts[group.Result.PresentationReceiptID]
		if !same(native["finding"], group.Result) || !same(native["selected_finding_ids"], group.FindingIDs) || !same(native["presentation"], receipt) {
			return fail
		}
		members := make([]findingexport.Result, 0, len(group.FindingIDs))
		for _, id := range group.FindingIDs {
			members = append(members, findings[id])
		}
		if !same(native["selected_findings"], members) {
			return fail
		}
		ri, ok := index(item["ruleIndex"], len(rules))
		if !ok || str(item["ruleId"]) != ruleID(receipt) || str(obj(rules[ri])["id"]) != str(item["ruleId"]) {
			return fail
		}
		kind, level := expectedRating(group.Result)
		if str(item["kind"]) != kind || str(item["level"]) != level {
			return fail
		}
		for _, field := range []string{"baselineState", "suppressions", "fixes", "partialFingerprints"} {
			if _, exists := item[field]; exists {
				return fail
			}
		}
		if group.Result.PrimaryKey == nil {
			if _, exists := item["fingerprints"]; exists {
				return fail
			}
		} else if !same(item["fingerprints"], map[string]string{group.Result.PrimaryKey.Algorithm: group.Result.PrimaryKey.Value}) {
			return fail
		}
		primary := ""
		locations := map[string]findinganalysis.Location{}
		if receipt.Analysis != nil {
			primary = receipt.Analysis.Analysis.PrimaryLocationID
			for _, l := range receipt.Analysis.Analysis.Locations {
				locations[l.ID] = l
			}
		}
		first := array(item["locations"])
		if primary == "" && len(first) != 0 || primary != "" && len(first) != 1 {
			return fail
		}
		seenLocations := map[string]bool{}
		seenIDs := map[float64]bool{}
		for _, field := range []string{"locations", "relatedLocations"} {
			for _, value := range array(item[field]) {
				l := obj(value)
				id := str(contractor(l)["native_location_id"])
				nativeLocation, exists := locations[id]
				number, numberOK := l["id"].(float64)
				if !exists || seenLocations[id] || !numberOK || seenIDs[number] || field == "locations" && id != primary || !validLocation(l, nativeLocation, receipt, files, bases) {
					return fail
				}
				seenLocations[id], seenIDs[number] = true, true
			}
		}
		if len(seenLocations) != len(locations) {
			return fail
		}
		if err := validFlows(item, receipt, locations, files, bases); err != nil {
			return err
		}
		expectedTaxa := map[string]bool{}
		for _, ref := range receipt.Proposal.StandardRefs {
			if mappedTaxonomy(ref) {
				expectedTaxa[taxonomyKey(ref)+"\x00"+ref.RequirementID] = true
			}
		}
		seenTaxa := map[string]bool{}
		for _, value := range array(item["taxa"]) {
			ref := obj(value)
			component := obj(ref["toolComponent"])
			ti, ok := index(component["index"], len(taxonomies))
			if !ok {
				return fail
			}
			taxonomy := obj(taxonomies[ti])
			key := str(taxonomy["name"]) + "\x00" + str(ref["id"])
			if str(component["name"]) != str(taxonomy["name"]) || !expectedTaxa[key] || seenTaxa[key] {
				return fail
			}
			found := false
			for _, entry := range array(taxonomy["taxa"]) {
				if str(obj(entry)["id"]) == str(ref["id"]) {
					found = true
					break
				}
			}
			if !found {
				return fail
			}
			seenTaxa[key] = true
		}
		if len(seenTaxa) != len(expectedTaxa) || !validHTTP(item, receipt.HTTP) {
			return fail
		}
	}
	return nil
}

func expectedRating(result findingexport.Result) (string, string) {
	if result.Review.State != "confirmed" {
		return "review", "none"
	}
	switch result.Review.Severity {
	case "informational", "low":
		return "fail", "note"
	case "high", "critical":
		return "fail", "error"
	default:
		return "fail", "warning"
	}
}

func artifactIdentity(location, bases object) (string, bool) {
	base := str(location["uriBaseId"])
	uri := str(location["uri"])
	if base == "" || obj(bases[base]) == nil {
		return "", false
	}
	parsed, err := url.Parse(uri)
	if err != nil || parsed.IsAbs() || parsed.Host != "" || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return "", false
	}
	decoded, err := url.PathUnescape(uri)
	if err != nil || strings.HasPrefix(decoded, "/") {
		return "", false
	}
	return base + "\x00" + decoded, true
}

func validLocation(value object, native findinganalysis.Location, receipt findingexport.Receipt, files []any, bases object) bool {
	physical := obj(value["physicalLocation"])
	ref := obj(physical["artifactLocation"])
	i, ok := index(ref["index"], len(files))
	if !ok {
		return false
	}
	key, ok := artifactIdentity(ref, bases)
	other, otherOK := artifactIdentity(obj(obj(files[i])["location"]), bases)
	if !ok || !otherOK || key != other {
		return false
	}
	var root string
	for _, source := range receipt.Analysis.Sources {
		if source.ID == native.SourceID {
			root = source.RootKey
			break
		}
	}
	if key != baseID(root)+"\x00"+native.Path {
		return false
	}
	region := object{}
	if r := native.Region; r != nil {
		region["startLine"] = r.StartLine
		if r.EndLine != nil {
			region["endLine"] = *r.EndLine
		}
		if r.StartColumn != nil {
			region["startColumn"] = *r.StartColumn
		}
		if r.EndColumn != nil {
			region["endColumn"] = *r.EndColumn
		}
	}
	if len(region) == 0 {
		if _, exists := physical["region"]; exists {
			return false
		}
	} else if !same(physical["region"], region) {
		return false
	}
	logical := array(value["logicalLocations"])
	if native.Symbol == nil {
		return len(logical) == 0
	}
	if len(logical) != 1 {
		return false
	}
	symbol := obj(logical[0])
	expectedSymbol := object{"name": native.Symbol.Name}
	if native.Symbol.QualifiedName != "" {
		expectedSymbol["fullyQualifiedName"] = native.Symbol.QualifiedName
	}
	if native.Symbol.Kind != "" {
		expectedSymbol["kind"] = native.Symbol.Kind
	}
	details := object{}
	if native.Symbol.Language != "" {
		details["language"] = native.Symbol.Language
	}
	if native.Symbol.Signature != "" {
		details["signature"] = native.Symbol.Signature
	}
	if len(details) > 0 {
		expectedSymbol["properties"] = object{"contractor": details}
	}
	return same(symbol, expectedSymbol)
}

func validHTTP(item object, h *findinganalysis.ResolvedHTTP) bool {
	request, response := obj(item["webRequest"]), obj(item["webResponse"])
	if h == nil {
		return request == nil && response == nil
	}
	if h.Method != "" && h.URI != "" {
		if !same(request, object{"method": h.Method, "target": h.URI}) {
			return false
		}
	} else if request != nil {
		return false
	}
	if h.Status == nil {
		return response == nil
	}
	expected := object{"statusCode": *h.Status}
	if h.ContentType != "" {
		expected["headers"] = object{"Content-Type": h.ContentType}
	}
	return same(response, expected)
}

func validFlows(item object, receipt findingexport.Receipt, locations map[string]findinganalysis.Location, files []any, bases object) error {
	fail := findingexport.ErrInvalid
	flows := map[string]findinganalysis.Flow{}
	required := map[string]bool{}
	if receipt.Analysis != nil {
		for _, flow := range receipt.Analysis.Analysis.Flows {
			flows[flow.ID] = flow
			gaps := map[string]bool{}
			for _, g := range flow.Gaps {
				gaps[g.AfterStepID+"\x00"+g.BeforeStepID] = true
			}
			for i := 1; i < len(flow.Steps); i++ {
				a, c := flow.Steps[i-1], flow.Steps[i]
				if locations[a.LocationID].Region != nil && locations[c.LocationID].Region != nil && !gaps[a.ID+"\x00"+c.ID] {
					required[flow.ID+"\x00"+a.ID+"\x00"+c.ID] = true
				}
			}
		}
	}
	seen := map[string]bool{}
	counts := map[string]int{}
	for _, value := range array(item["codeFlows"]) {
		code := obj(value)
		id := str(contractor(code)["flow_id"])
		flow, exists := flows[id]
		if !exists {
			return fail
		}
		partial := flow.Completeness != "complete" || len(flow.Gaps) > 0
		for _, step := range flow.Steps {
			if locations[step.LocationID].Region == nil {
				partial = true
			}
		}
		meta := contractor(code)
		if str(meta["kind"]) != flow.Kind || str(meta["claimed_completeness"]) != flow.Completeness || meta["partial_projection"] != partial || meta["segment"] != float64(counts[id]+1) {
			return fail
		}
		threads := array(code["threadFlows"])
		if len(threads) != 1 {
			return fail
		}
		points := array(obj(threads[0])["locations"])
		if len(points) < 2 {
			return fail
		}
		counts[id]++
		order := map[string]int{}
		steps := map[string]findinganalysis.Step{}
		for i, s := range flow.Steps {
			order[s.ID] = i
			steps[s.ID] = s
		}
		previous := ""
		for i, value := range points {
			point := obj(value)
			details := contractor(point)
			sid := str(details["step_id"])
			step, ok := steps[sid]
			location, located := locations[step.LocationID]
			if !ok || !located || location.Region == nil || point["executionOrder"] != float64(i+1) || !same(point["kinds"], step.Roles) || !same(details["native_step"], step) || str(details["basis"]) != "producer-asserted" || str(contractor(obj(point["location"]))["native_location_id"]) != step.LocationID || !validLocation(obj(point["location"]), location, receipt, files, bases) {
				return fail
			}
			if details["native_order"] != float64(order[sid]+1) {
				return fail
			}
			expectedState := object{}
			if step.Value != "" {
				expectedState["value"] = object{"text": step.Value}
			}
			if step.ValueState != "" {
				expectedState["valueState"] = object{"text": step.ValueState}
			}
			if len(expectedState) == 0 {
				if _, exists := point["state"]; exists {
					return fail
				}
			} else if !same(point["state"], expectedState) {
				return fail
			}
			message := str(obj(obj(point["location"])["message"])["text"])
			if !strings.HasPrefix(message, step.Message) {
				return fail
			}
			for _, role := range step.Roles {
				if (role == "sanitizer" || role == "validation") && !strings.Contains(message, "Producer claim; this does not establish safety.") {
					return fail
				}
			}
			if i > 0 {
				var edge *findinganalysis.Edge
				for j := range flow.Edges {
					if flow.Edges[j].From == previous && flow.Edges[j].To == sid {
						edge = &flow.Edges[j]
						break
					}
				}
				if edge == nil || !same(details["edge_from_previous"], *edge) {
					return fail
				}
				key := id + "\x00" + previous + "\x00" + sid
				if order[sid] != order[previous]+1 || !required[key] || seen[key] {
					return fail
				}
				seen[key] = true
			}
			if i == 0 {
				if _, exists := details["edge_from_previous"]; exists {
					return fail
				}
			}
			previous = sid
		}
	}
	if len(seen) != len(required) {
		return fail
	}
	projected := obj(contractor(item)["flow_segments"])
	for id := range flows {
		if projected[id] != float64(counts[id]) {
			return fail
		}
	}
	return nil
}
