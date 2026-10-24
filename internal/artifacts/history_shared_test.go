package artifacts_test

import (
	"bytes"
	"encoding/json"
	"testing"

	. "github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// artifactHistoryCases is api/testdata/v1alpha1/artifact-history-cases.json.
type artifactHistoryCases struct {
	Description string `json:"description"`
	Policy      struct {
		DefaultMaxRevisions         int64 `json:"defaultMaxRevisions"`
		DefaultMaxBytes             int64 `json:"defaultMaxBytes"`
		MaxRevisionsFloor           int64 `json:"maxRevisionsFloor"`
		MaxBytesFloor               int64 `json:"maxBytesFloor"`
		MaxRevisionsCeiling         int64 `json:"maxRevisionsCeiling"`
		MaxBytesCeiling             int64 `json:"maxBytesCeiling"`
		MaxCandidatesPerTransaction int   `json:"maxCandidatesPerTransaction"`
		MaxRemovalsPerTransaction   int   `json:"maxRemovalsPerTransaction"`
		MaxReclaimBytesPerPass      int64 `json:"maxReclaimBytesPerPass"`
		RetryAfterSeconds           int   `json:"retryAfterSeconds"`
	} `json:"policy"`
	Errors []json.RawMessage `json:"errors"`
}

// TestSharedArtifactHistoryPolicy checks the one typed policy definition
// against api/testdata/v1alpha1/artifact-history-cases.json, whose errors the
// private Artifact API, both MemoryTools adapters and the Runtime's CAS
// writers read too.
func TestSharedArtifactHistoryPolicy(t *testing.T) {
	t.Parallel()
	var cases artifactHistoryCases
	decoder := json.NewDecoder(bytes.NewReader(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "artifact-history-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cases); err != nil {
		t.Fatal(err)
	}
	policy := cases.Policy
	defaults := DefaultHistoryPolicy()
	for _, check := range []struct {
		name      string
		got, want int64
	}{
		{"default revisions", defaults.MaxRevisions(), policy.DefaultMaxRevisions},
		{"default bytes", defaults.MaxBytes(), policy.DefaultMaxBytes},
		{"revisions floor", HistoryRevisionsFloor, policy.MaxRevisionsFloor},
		{"bytes floor", HistoryBytesFloor, policy.MaxBytesFloor},
		{"revisions ceiling", HistoryRevisionsCeiling, policy.MaxRevisionsCeiling},
		{"bytes ceiling", HistoryBytesCeiling, policy.MaxBytesCeiling},
		{"candidates per transaction", MaxHistoryCandidatesPerTransaction, int64(policy.MaxCandidatesPerTransaction)},
		{"removals per transaction", MaxHistoryRemovalsPerTransaction, int64(policy.MaxRemovalsPerTransaction)},
		{"reclaimed bytes per pass", MaxHistoryReclaimBytesPerPass, policy.MaxReclaimBytesPerPass},
		{"Retry-After seconds", int64(HistoryRetryAfter.Seconds()), int64(policy.RetryAfterSeconds)},
	} {
		if check.got != check.want {
			t.Errorf("%s = %d, want the shared %d", check.name, check.got, check.want)
		}
	}
	for _, limits := range []struct {
		revisions, bytes int64
		valid            bool
	}{
		{policy.MaxRevisionsFloor, policy.MaxBytesFloor, true},
		{policy.MaxRevisionsCeiling, policy.MaxBytesCeiling, true},
		{policy.MaxRevisionsFloor - 1, policy.MaxBytesFloor, false},
		{policy.MaxRevisionsCeiling + 1, policy.MaxBytesFloor, false},
		{policy.MaxRevisionsFloor, policy.MaxBytesFloor - 1, false},
		{policy.MaxRevisionsFloor, policy.MaxBytesCeiling + 1, false},
	} {
		if _, err := NewHistoryPolicy(limits.revisions, limits.bytes); (err == nil) != limits.valid {
			t.Errorf("NewHistoryPolicy(%d, %d) = %v, want valid %t", limits.revisions, limits.bytes, err, limits.valid)
		}
	}
}
