package projectsitemap

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"os"
	"strings"
	"testing"
)

func scannerFixture(t *testing.T, kind string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + kind + "-report.json")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func editScannerFixture(t *testing.T, kind string, edit func(map[string]any, map[string]any)) []byte {
	t.Helper()
	var envelope map[string]any
	if err := json.Unmarshal(scannerFixture(t, kind), &envelope); err != nil {
		t.Fatal(err)
	}
	edit(envelope, envelope["observation"].(map[string]any))
	data, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestScannerReportsProjectNormalizedFixtures(t *testing.T) {
	for _, kind := range []string{"ffuf", "katana"} {
		t.Run(kind, func(t *testing.T) {
			command, err := scannerReport(scannerFixture(t, kind), kind, "2026-10-29T12:00:00Z", "scanner")
			if err != nil {
				t.Fatal(err)
			}
			if len(command.Observations) != 1 || len(command.Gaps) != 0 {
				t.Fatalf("unexpected projection: %+v", command)
			}
			observed := command.Observations[0]
			if observed.Method != "GET" || observed.Status != 200 || observed.URL != "https://example.test/users?expand=profile" || observed.At != "2026-10-29T12:00:00Z" {
				t.Fatalf("wrong metadata: %+v", observed)
			}
		})
	}
}

func TestScannerReportsRejectMalformedEnvelopesAndRows(t *testing.T) {
	tests := []struct {
		name, kind string
		edit       func(map[string]any, map[string]any)
	}{
		{"unknown envelope", "ffuf", func(e, o map[string]any) { e["extra"] = true }},
		{"case alias", "ffuf", func(e, o map[string]any) { delete(e, "schemaVersion"); e["SchemaVersion"] = 1 }},
		{"wrong version", "ffuf", func(e, o map[string]any) { e["schemaVersion"] = 2 }},
		{"wrong tool", "ffuf", func(e, o map[string]any) { e["tool"] = "scan_katana" }},
		{"missing required null", "ffuf", func(e, o map[string]any) { delete(o, "errorCode") }},
		{"missing array", "ffuf", func(e, o map[string]any) { delete(o, "results") }},
		{"null array", "ffuf", func(e, o map[string]any) { o["results"] = nil }},
		{"unknown row", "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["headers"] = map[string]any{} }},
		{"boolean integer", "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["status"] = true }},
		{"invalid status", "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["status"] = 0 }},
		{"non http url", "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["url"] = "file:///tmp/report" }},
		{"userinfo", "ffuf", func(e, o map[string]any) {
			o["results"].([]any)[0].(map[string]any)["url"] = "https://alice:pass@example.test/"
		}},
		{"position", "ffuf", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["position"] = 2 }},
		{"false completion", "ffuf", func(e, o map[string]any) { o["payloadsAttempted"] = nil }},
		{"raw diagnostics", "ffuf", func(e, o map[string]any) { o["stdout"] = "raw" }},
		{"too many results", "ffuf", func(e, o map[string]any) {
			rows := o["results"].([]any)
			for len(rows) <= 100 {
				rows = append(rows, rows[0])
			}
			o["results"] = rows
		}},
		{"large results", "ffuf", func(e, o map[string]any) {
			o["results"].([]any)[0].(map[string]any)["contentType"] = strings.Repeat("a", maxScannerResultBytes)
		}},
		{"foreign origin", "katana", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["url"] = "https://other.test/" }},
		{"invented method", "katana", func(e, o map[string]any) { o["results"].([]any)[0].(map[string]any)["method"] = "POST" }},
		{"wrong target digest", "katana", func(e, o map[string]any) { o["targetsDigest"] = "sha256:" + strings.Repeat("b", 64) }},
		{"unknown limitation", "katana", func(e, o map[string]any) { o["coverage"].(map[string]any)["limitations"] = []any{"unknown"} }},
		{"inconsistent counters", "katana", func(e, o map[string]any) { o["coverage"].(map[string]any)["responses"] = 2 }},
		{"foreign target ref", "katana", func(e, o map[string]any) { o["targetsArtifact"].(map[string]any)["revision"] = "other" }},
		{"oversized envelope", "ffuf", func(e, o map[string]any) { e["inputDigest"] = strings.Repeat("a", maxScannerReportBytes) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := scannerReport(editScannerFixture(t, test.kind, test.edit), test.kind, "2026-10-29T12:00:00Z", "scanner")
			if err == nil {
				t.Fatal("malformed report accepted")
			}
		})
	}
	for _, kind := range []string{"ffuf", "katana"} {
		data := scannerFixture(t, kind)
		duplicate := strings.Replace(string(data), `"schemaVersion": 1`, `"schemaVersion": 1, "schemaVersion": 1`, 1)
		if _, err := scannerReport([]byte(duplicate), kind, "2026-10-29T12:00:00Z", "scanner"); err == nil {
			t.Fatal("duplicate field accepted")
		}
	}
}

func TestScannerReportsRetainPartialResultsAndTruthfulGaps(t *testing.T) {
	ffuf := editScannerFixture(t, "ffuf", func(e, o map[string]any) {
		o["status"], o["errorCode"], o["scanComplete"] = "failed", "scan_request_failed", false
		o["wordlistEntries"], o["payloadsAttempted"], o["requestErrors"] = 10, 7, 2
		o["resultsTruncated"], o["invalidResultLines"] = true, 3
	})
	command, err := scannerReport(ffuf, "ffuf", "2026-10-29T12:00:00Z", "scanner")
	if err != nil {
		t.Fatal(err)
	}
	assertReportGaps(t, command, map[string]uint64{"incomplete": 1, "projection": 3, "report-truncated": 1, "request-errors": 2})
	if len(command.Observations) != 1 {
		t.Fatal("partial report discarded retained response")
	}
	katana := editScannerFixture(t, "katana", func(e, o map[string]any) {
		o["status"], o["errorCode"] = "failed", "scan_request_failed"
		o["requestErrors"], o["invalidResultLines"], o["resultsTruncated"] = 2, 3, true
		c := o["coverage"].(map[string]any)
		c["responses"], c["uniqueTargets"], c["omittedTargets"] = 5, 5, 4
		c["requestErrors"], c["invalidResultLines"], c["depthLimited"], c["unsupportedMethods"] = 2, 3, 1, 1
		c["observations"] = 12
		c["limitations"] = []string{"bounded_crawl", "exhaustion_unverified", "response_size_limit", "default_extension_filter", "depth_limit", "request_errors", "unsupported_methods", "invalid_output", "target_export_limit"}
	})
	command, err = scannerReport(katana, "katana", "2026-10-29T12:00:00Z", "scanner")
	if err != nil {
		t.Fatal(err)
	}
	assertReportGaps(t, command, map[string]uint64{"incomplete": 1, "projection": 5, "members": 4, "report-truncated": 1, "request-errors": 2})
}

func assertReportGaps(t *testing.T, command Mutation, expected map[string]uint64) {
	t.Helper()
	if len(command.Gaps) != len(expected) {
		t.Fatalf("unexpected gaps: %+v", command.Gaps)
	}
	for _, gap := range command.Gaps {
		if gap.SourceID != "scanner" || expected[gap.Reason] != gap.Count {
			t.Fatalf("unexpected gap: %+v", gap)
		}
	}
}

func ffufBoundaryReport(t *testing.T, count int) []byte {
	return editScannerFixture(t, "ffuf", func(e, o map[string]any) {
		o["wordlistEntries"], o["payloadsAttempted"] = count, count
		sample := o["results"].([]any)[0].(map[string]any)
		rows := make([]any, count)
		for n := range count {
			row := make(map[string]any, len(sample))
			for k, v := range sample {
				row[k] = v
			}
			row["position"], row["url"] = n+1, fmt.Sprintf("https://example.test/item/%03d", n)
			rows[n] = row
		}
		o["results"] = rows
	})
}

func TestSharedSitemapScannerReports(t *testing.T) {
	for _, kind := range []string{"ffuf", "katana"} {
		shared := contractstest.ReadFile(t, "api", "testdata", "sitemap-reports", kind+"-report.json")
		if !bytes.Equal(shared, scannerFixture(t, kind)) {
			t.Fatalf("%s integration fixture differs from producer contract", kind)
		}
		if _, err := scannerReport(shared, kind, "2026-10-29T12:00:00Z", "scanner"); err != nil {
			t.Fatalf("%s shared report rejected: %v", kind, err)
		}
	}
}
