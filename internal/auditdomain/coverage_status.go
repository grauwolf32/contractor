package auditdomain

// Coverage statuses of an Audit item. Requirement rows use satisfied,
// violated, inconclusive, not-tested and not-applicable; operation rows use
// traced-complete, traced-partial, unmapped and not-tested; blocked and
// excluded record execution and human outcomes on either kind.
const (
	CoverageNotTested      = "not-tested"
	CoverageInconclusive   = "inconclusive"
	CoverageSatisfied      = "satisfied"
	CoverageViolated       = "violated"
	CoverageNotApplicable  = "not-applicable"
	CoverageBlocked        = "blocked"
	CoverageExcluded       = "excluded"
	CoverageTracedComplete = "traced-complete"
	CoverageTracedPartial  = "traced-partial"
	CoverageUnmapped       = "unmapped"
)

// CoverageBucket is the progress class of a coverage status. Every status is
// in exactly one bucket. The Audit workspace counts the current round by
// bucket, and the Audit report concludes completed-with-gaps when any item is
// outside CoverageBucketCompleted, so both read this one table (REV-R193).
type CoverageBucket string

const (
	// CoverageBucketCompleted holds concluded assessments and policy-authorized
	// not-applicable decisions.
	CoverageBucketCompleted CoverageBucket = "completed"
	// CoverageBucketGap holds work that ended without a conclusion: blocked,
	// inconclusive, partially traced and unmapped items, and items the owner
	// rejected, which settle as excluded with a visible gap (AUD-R178).
	CoverageBucketGap CoverageBucket = "gap"
	// CoverageBucketUnchecked holds items with no recorded result yet.
	CoverageBucketUnchecked CoverageBucket = "unchecked"
)

// coverageStatusTable classifies every coverage status. issue marks the
// completed status that records an issue found. The shared fixture
// api/testdata/public/audit-coverage-buckets.json holds the UI result groups
// to the same classification.
var coverageStatusTable = [...]struct {
	status string
	bucket CoverageBucket
	issue  bool
}{
	{CoverageNotTested, CoverageBucketUnchecked, false},
	{CoverageInconclusive, CoverageBucketGap, false},
	{CoverageSatisfied, CoverageBucketCompleted, false},
	{CoverageViolated, CoverageBucketCompleted, true},
	{CoverageNotApplicable, CoverageBucketCompleted, false},
	{CoverageBlocked, CoverageBucketGap, false},
	{CoverageExcluded, CoverageBucketGap, false},
	{CoverageTracedComplete, CoverageBucketCompleted, false},
	{CoverageTracedPartial, CoverageBucketGap, false},
	{CoverageUnmapped, CoverageBucketGap, false},
}

// CoverageStatusBucket returns the bucket of status, and false when status is
// not a coverage status.
func CoverageStatusBucket(status string) (CoverageBucket, bool) {
	for _, entry := range coverageStatusTable {
		if entry.status == status {
			return entry.bucket, true
		}
	}
	return "", false
}

// CoverageStatusesIn returns the statuses of bucket in table order.
func CoverageStatusesIn(bucket CoverageBucket) []string {
	result := []string{}
	for _, entry := range coverageStatusTable {
		if entry.bucket == bucket {
			result = append(result, entry.status)
		}
	}
	return result
}

// CoverageIssueStatuses returns the statuses that record an issue found; each
// of them is in CoverageBucketCompleted.
func CoverageIssueStatuses() []string {
	result := []string{}
	for _, entry := range coverageStatusTable {
		if entry.issue {
			result = append(result, entry.status)
		}
	}
	return result
}
