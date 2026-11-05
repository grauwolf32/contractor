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
// in exactly one bucket for retained report readers (REV-R193). Common
// research reads do not infer an assessment from coordination counters.
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

// coverageStatusTable classifies every retained coverage status. The fixture
// api/testdata/public/audit-coverage-buckets.json holds the retained Server
// coverage readers to the same classification.
var coverageStatusTable = [...]struct {
	status string
	bucket CoverageBucket
}{
	{CoverageNotTested, CoverageBucketUnchecked},
	{CoverageInconclusive, CoverageBucketGap},
	{CoverageSatisfied, CoverageBucketCompleted},
	{CoverageViolated, CoverageBucketCompleted},
	{CoverageNotApplicable, CoverageBucketCompleted},
	{CoverageBlocked, CoverageBucketGap},
	{CoverageExcluded, CoverageBucketGap},
	{CoverageTracedComplete, CoverageBucketCompleted},
	{CoverageTracedPartial, CoverageBucketGap},
	{CoverageUnmapped, CoverageBucketGap},
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
