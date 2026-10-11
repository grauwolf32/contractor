package auditdomain

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// TestCoverageBucketsMatchSharedFixture holds the one coverage classification
// to the fixture the UI result groups are tested against.
func TestCoverageBucketsMatchSharedFixture(t *testing.T) {
	data, err := os.ReadFile("../../api/testdata/public/audit-coverage-buckets.json")
	if err != nil {
		t.Fatal(err)
	}
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
		got := CoverageStatusesIn(test.bucket)
		if !sameSet(got, test.want) {
			t.Errorf("bucket %s = %v, fixture %v", test.bucket, got, test.want)
		}
		for _, status := range test.want {
			if bucket, ok := CoverageStatusBucket(status); !ok || bucket != test.bucket {
				t.Errorf("CoverageStatusBucket(%q) = (%q, %t), want %q", status, bucket, ok, test.bucket)
			}
		}
	}
	if got := CoverageIssueStatuses(); !sameSet(got, fixture.Issue) {
		t.Errorf("issue statuses = %v, fixture %v", got, fixture.Issue)
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

func sameSet(left, right []string) bool {
	a, b := slices.Clone(left), slices.Clone(right)
	slices.Sort(a)
	slices.Sort(b)
	return slices.Equal(a, b) && len(slices.Compact(a)) == len(left)
}
