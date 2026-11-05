package auditdomain

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedCoverageBucketsMatchSharedFixture holds the one coverage classification
// to the retained report fixture.
func TestSharedCoverageBucketsMatchSharedFixture(t *testing.T) {
	data := contractstest.ReadFile(t, "api", "testdata", "public", "audit-coverage-buckets.json")
	var fixture struct {
		Description string   `json:"description"`
		Completed   []string `json:"completed"`
		Issue       []string `json:"issue"`
		Gap         []string `json:"gap"`
		Unchecked   []string `json:"unchecked"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		bucket CoverageBucket
		want   []string
	}{
		{CoverageBucketCompleted, fixture.Completed},
		{CoverageBucketGap, fixture.Gap},
		{CoverageBucketUnchecked, fixture.Unchecked},
	} {
		for _, status := range test.want {
			if bucket, ok := CoverageStatusBucket(status); !ok || bucket != test.bucket {
				t.Errorf("CoverageStatusBucket(%q) = (%q, %t), want %q", status, bucket, ok, test.bucket)
			}
		}
	}
	for _, status := range fixture.Issue {
		if bucket, _ := CoverageStatusBucket(status); bucket != CoverageBucketCompleted {
			t.Errorf("issue status %q is in bucket %q, want completed", status, bucket)
		}
	}
	all := slices.Concat(fixture.Completed, fixture.Gap, fixture.Unchecked)
	if len(all) != len(coverageStatusTable) {
		t.Fatalf("fixture classifies %d statuses, table %d", len(all), len(coverageStatusTable))
	}
	for _, entry := range coverageStatusTable {
		if !validCoverageStatus(entry.status) {
			t.Errorf("table status %q is not a valid coverage status", entry.status)
		}
	}
	for _, unknown := range []string{"", "Excluded", "pending", "traced"} {
		if bucket, ok := CoverageStatusBucket(unknown); ok || bucket != "" || validCoverageStatus(unknown) {
			t.Errorf("unknown status %q classified as (%q, %t)", unknown, bucket, ok)
		}
	}
}
