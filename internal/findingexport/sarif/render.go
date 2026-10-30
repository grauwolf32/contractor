// Package sarif renders frozen Audit exports without storage, network or model work.
package sarif

import (
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

type builder struct {
	snapshot    findingexport.Snapshot
	receipts    map[string]findingexport.Receipt
	findings    map[string]findingexport.Result
	budget      byteBudget
	registry    *registry
	diagnostics map[string]findingexport.Diagnostic
}

// Render validates frozen inputs, constructs bounded deterministic projections,
// then validates the actual bytes against the embedded official schema and
// semantic reference invariants. Replaying publication never calls Render.
func Render(snapshot findingexport.Snapshot) ([]byte, findingexport.Diagnostics, error) {
	if snapshot.ExporterVersion != findingexport.ExporterVersion {
		return nil, findingexport.Diagnostics{}, findingexport.ErrInvalid
	}
	frozen, err := findingexport.EncodeSnapshot(snapshot)
	if err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	groups, err := snapshot.ResultGroups()
	if err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	b := builder{snapshot: snapshot, receipts: map[string]findingexport.Receipt{}, findings: map[string]findingexport.Result{}, diagnostics: map[string]findingexport.Diagnostic{}}
	b.registry = newRegistry(&b.budget)
	for _, r := range snapshot.Receipts {
		b.receipts[r.ReceiptID] = r
	}
	for _, r := range snapshot.Results {
		b.findings[r.FindingID] = r
	}
	for _, group := range groups {
		if err := b.registry.register(b.receipts[group.Result.PresentationReceiptID]); err != nil {
			return nil, findingexport.Diagnostics{}, err
		}
	}
	run, err := b.registry.metadata()
	if err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	properties := object{"contractor": object{"audit_id": snapshot.AuditID, "snapshot_at": snapshot.SnapshotAt, "snapshot_digest": contentdigest.Bytes(frozen), "request_digest": snapshot.RequestDigest, "policy_version": snapshot.PolicyVersion, "include_duplicates": snapshot.Request.IncludeDuplicates, "selected_count": len(snapshot.Results), "result_count": len(groups)}}
	if err := b.budget.add(properties); err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	run["properties"] = properties
	results := make([]any, 0, len(groups))
	for _, group := range groups {
		result, err := b.result(group)
		if err != nil {
			return nil, findingexport.Diagnostics{}, err
		}
		results = append(results, result)
	}
	run["results"] = results
	document := object{"$schema": schemaURI, "version": "2.1.0", "runs": []any{run}}
	data, err := strictjson.Canonical(document)
	if err != nil {
		return nil, findingexport.Diagnostics{}, findingexport.ErrInvalid
	}
	if len(data)+1 > findingexport.MaxBytes {
		return nil, findingexport.Diagnostics{}, findingexport.ErrLimit
	}
	data = append(data, '\n')
	validated, err := validateSchema(data)
	if err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	if err := validateSemantics(validated, snapshot); err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	diagnostics := findingexport.Diagnostics{Schema: findingexport.DiagnosticsSchema, AuditID: snapshot.AuditID, Items: []findingexport.Diagnostic{}}
	for _, key := range sortedKeys(b.diagnostics) {
		diagnostics.Items = append(diagnostics.Items, b.diagnostics[key])
	}
	if _, err := findingexport.EncodeDiagnostics(diagnostics); err != nil {
		return nil, diagnostics, err
	}
	return data, diagnostics, nil
}

func (b *builder) result(group findingexport.ResultGroup) (object, error) {
	result := group.Result
	receipt := b.receipts[result.PresentationReceiptID]
	warn := func(code, field string) {
		key := result.FindingID + "\x00" + code + "\x00" + field
		if len(b.diagnostics) <= 4096 {
			b.diagnostics[key] = findingexport.Diagnostic{Code: code, Severity: "warning", FindingID: result.FindingID, Field: field}
		}
	}
	kind, level := "review", "none"
	if result.Review.State == "confirmed" {
		kind = "fail"
		switch result.Review.Severity {
		case "informational", "low":
			level = "note"
		case "medium":
			level = "warning"
		case "high", "critical":
			level = "error"
		default:
			level = "warning"
			warn("severity_unrated", "review.severity")
		}
	}
	if receipt.Analysis == nil {
		warn("analysis_missing", "analysis")
	}
	if ruleID(receipt) == unclassifiedRule {
		warn("rule_missing", "analysis.rule")
	}
	if result.PrimaryKey == nil {
		warn("fingerprint_unavailable", "primary_key")
	}
	members := make([]findingexport.Result, 0, len(group.FindingIDs))
	supporting := map[string]bool{}
	for _, id := range group.FindingIDs {
		row := b.findings[id]
		members = append(members, row)
		for _, receiptID := range row.ReceiptIDs {
			if receiptID != receipt.ReceiptID {
				supporting[receiptID] = true
			}
		}
	}
	supportingRefs := make([]any, 0, len(supporting))
	for _, id := range sortedKeys(supporting) {
		r := b.receipts[id]
		supportingRefs = append(supportingRefs, object{"receipt_id": r.ReceiptID, "proposal_id": r.ProposalID, "proposal_artifact": r.ProposalArtifact, "origin": r.Origin, "evidence": r.Evidence})
	}
	native := object{"audit_id": b.snapshot.AuditID, "finding": result, "selected_finding_ids": group.FindingIDs, "selected_findings": members, "presentation": receipt, "supporting_receipts": supportingRefs, "snapshot_at": b.snapshot.SnapshotAt}
	locations, err := b.registry.locations(receipt)
	if err != nil {
		return nil, err
	}
	projected := object{}
	primary := ""
	if receipt.Analysis != nil {
		primary = receipt.Analysis.Analysis.PrimaryLocationID
	}
	if primary != "" {
		location, exists := locations[primary]
		if !exists {
			return nil, findingexport.ErrInvalid
		}
		value := locationWithID(location)
		if err := b.budget.add(value); err != nil {
			return nil, err
		}
		projected["locations"] = []any{value}
		if location.native.Region == nil {
			warn("primary_region_missing", "locations")
		}
	} else {
		warn("primary_location_missing", "locations")
	}
	related := []any{}
	for _, id := range sortedKeys(locations) {
		if id != primary {
			value := locationWithID(locations[id])
			if err := b.budget.add(value); err != nil {
				return nil, err
			}
			related = append(related, value)
		}
	}
	if len(related) > 0 {
		projected["relatedLocations"] = related
	}
	flows, counts, err := b.flows(receipt, locations, warn)
	if err != nil {
		return nil, err
	}
	if len(flows) > 0 {
		projected["codeFlows"] = flows
	}
	if len(counts) > 0 {
		native["flow_segments"] = counts
	}
	taxa := []any{}
	seen := map[string]bool{}
	for _, ref := range receipt.Proposal.StandardRefs {
		if !mappedTaxonomy(ref) {
			warn("classification_unmapped", "standard_refs")
			continue
		}
		key := taxonomyKey(ref)
		identity := key + "\x00" + ref.RequirementID
		if seen[identity] {
			continue
		}
		seen[identity] = true
		entry := object{"id": ref.RequirementID, "toolComponent": object{"name": key, "index": b.registry.taxIndex[key]}}
		if err := b.budget.add(entry); err != nil {
			return nil, err
		}
		taxa = append(taxa, entry)
	}
	sort.Slice(taxa, func(i, j int) bool {
		a, _ := strictjson.Canonical(taxa[i])
		c, _ := strictjson.Canonical(taxa[j])
		return string(a) < string(c)
	})
	if len(taxa) > 0 {
		projected["taxa"] = taxa
	}
	severity := result.Review.Severity
	if severity == "" {
		severity = "not rated"
	}
	message := []string{receipt.Proposal.Title, receipt.Proposal.Description, "Review state: " + result.Review.State + "; analyst severity: " + severity + "; proposal suggestion: " + receipt.Proposal.SeveritySuggestion}
	if receipt.Proposal.Hypothesis != "" {
		message = append(message, "Hypothesis: "+receipt.Proposal.Hypothesis)
	}
	if len(receipt.Proposal.Preconditions) > 0 {
		message = append(message, "Preconditions: "+strings.Join(receipt.Proposal.Preconditions, "; "))
	}
	if len(receipt.Proposal.Limitations) > 0 {
		message = append(message, "Limitations: "+strings.Join(receipt.Proposal.Limitations, "; "))
	}
	if result.Review.Rationale != "" {
		message = append(message, "Analyst rationale: "+result.Review.Rationale)
	}
	if _, partial := b.diagnostics[result.FindingID+"\x00flow_partial\x00analysis.flows"]; partial {
		message = append(message, "Path projection is incomplete; retained gaps and unlocated steps remain explicit.")
	}
	item := object{"ruleId": ruleID(receipt), "ruleIndex": b.registry.ruleIndex[ruleID(receipt)], "kind": kind, "level": level, "message": object{"text": strings.Join(message, "\n\n") + flowSummary(counts)}, "properties": object{"contractor": native}}
	if result.PrimaryKey != nil {
		item["fingerprints"] = object{result.PrimaryKey.Algorithm: result.PrimaryKey.Value}
	}
	if h := receipt.HTTP; h != nil {
		if h.Method != "" && h.URI != "" {
			item["webRequest"] = object{"method": h.Method, "target": h.URI}
		}
		if h.Status != nil {
			response := object{"statusCode": *h.Status}
			if h.ContentType != "" {
				response["headers"] = object{"Content-Type": h.ContentType}
			}
			item["webResponse"] = response
		}
		warn("http_fields_omitted", "http")
	}
	if len(b.diagnostics) > 4096 {
		return nil, findingexport.ErrLimit
	}
	// Large child arrays were counted while being built; count the remaining
	// result envelope once before attaching them.
	if err := b.budget.add(item); err != nil {
		return nil, err
	}
	for key, value := range projected {
		item[key] = value
	}
	return item, nil
}
