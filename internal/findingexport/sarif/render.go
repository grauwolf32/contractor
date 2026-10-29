// Package sarif renders a frozen export without storage, network or model work.
package sarif

import (
	"encoding/json"
	"sort"

	"github.com/grauwolf32/contractor/internal/findingexport"
)

const schemaURI = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"

// Render consumes the snapshot alone. The base projection retains all selected
// identities and native data; structured locations and flow mapping extend it.
func Render(snapshot findingexport.Snapshot) ([]byte, findingexport.Diagnostics, error) {
	if snapshot.ExporterVersion != findingexport.ExporterVersion {
		return nil, findingexport.Diagnostics{}, findingexport.ErrInvalid
	}
	if _, err := findingexport.EncodeSnapshot(snapshot); err != nil {
		return nil, findingexport.Diagnostics{}, err
	}
	diagnostics := findingexport.Diagnostics{Schema: findingexport.DiagnosticsSchema, AuditID: snapshot.AuditID, Items: []findingexport.Diagnostic{}}
	receipts := map[string]findingexport.Receipt{}
	for _, r := range snapshot.Receipts {
		receipts[r.ReceiptID] = r
	}
	groups, err := snapshot.ResultGroups()
	if err != nil {
		return nil, diagnostics, err
	}
	results := make([]any, 0, len(groups))
	resultBytes := 0
	for _, group := range groups {
		result := group.Result
		receipt := receipts[result.PresentationReceiptID]
		warn := func(code, field string) {
			diagnostics.Items = append(diagnostics.Items, findingexport.Diagnostic{Code: code, Severity: "warning", FindingID: result.FindingID, Field: field})
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
		} else {
			warn("structured_projection_pending", "analysis")
		}
		warn("primary_location_missing", "locations")
		if result.PrimaryKey == nil {
			warn("fingerprint_unavailable", "primary_key")
		}
		if len(receipt.Proposal.StandardRefs) > 0 {
			warn("classification_unmapped", "standard_refs")
		}
		message := receipt.Proposal.Title + "\n\n" + receipt.Proposal.Description + "\n\nReview state: " + result.Review.State + "; analyst severity: " + result.Review.Severity + "; proposal suggestion: " + receipt.Proposal.SeveritySuggestion
		item := map[string]any{"ruleId": "contractor/unclassified", "ruleIndex": 0, "kind": kind, "level": level, "message": map[string]any{"text": message}, "properties": map[string]any{"contractor": map[string]any{"audit_id": snapshot.AuditID, "selected_finding_ids": group.FindingIDs, "finding": result, "presentation": receipt, "snapshot_at": snapshot.SnapshotAt}}}
		if result.PrimaryKey != nil {
			item["fingerprints"] = map[string]string{result.PrimaryKey.Algorithm: result.PrimaryKey.Value}
		}
		if h := receipt.HTTP; h != nil {
			if h.URI != "" && h.Method != "" {
				item["webRequest"] = map[string]any{"method": h.Method, "target": h.URI}
			}
			if h.Status != nil {
				response := map[string]any{"statusCode": *h.Status}
				if h.ContentType != "" {
					response["headers"] = map[string]string{"Content-Type": h.ContentType}
				}
				item["webResponse"] = response
			}
		}
		encodedItem, err := json.Marshal(item)
		if err != nil {
			return nil, diagnostics, findingexport.ErrInvalid
		}
		resultBytes += len(encodedItem) + 1
		if resultBytes > findingexport.MaxBytes {
			return nil, diagnostics, findingexport.ErrLimit
		}
		results = append(results, item)
	}
	sort.Slice(diagnostics.Items, func(i, j int) bool {
		a, b := diagnostics.Items[i], diagnostics.Items[j]
		if a.FindingID != b.FindingID {
			return a.FindingID < b.FindingID
		}
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		return a.Field < b.Field
	})
	document := map[string]any{"$schema": schemaURI, "version": "2.1.0", "runs": []any{map[string]any{"tool": map[string]any{"driver": map[string]any{"name": "Contractor", "version": snapshot.ExporterVersion, "rules": []any{map[string]any{"id": "contractor/unclassified", "name": "Unclassified possible issue", "shortDescription": map[string]string{"text": "Possible issue without a rendered check definition"}}}}}, "columnKind": "unicodeCodePoints", "results": results, "properties": map[string]any{"contractor": map[string]any{"audit_id": snapshot.AuditID, "snapshot_at": snapshot.SnapshotAt, "policy_version": snapshot.PolicyVersion, "include_duplicates": snapshot.Request.IncludeDuplicates, "selected_count": len(snapshot.Results), "result_count": len(groups)}}}}}
	encoded, err := json.Marshal(document)
	if err != nil {
		return nil, diagnostics, findingexport.ErrInvalid
	}
	if len(encoded)+1 > findingexport.MaxBytes {
		return nil, diagnostics, findingexport.ErrLimit
	}
	if _, err := findingexport.EncodeDiagnostics(diagnostics); err != nil {
		return nil, diagnostics, err
	}
	return append(encoded, '\n'), diagnostics, nil
}
