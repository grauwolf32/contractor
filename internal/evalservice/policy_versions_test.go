package evalservice

import "testing"

// The policy pins change only with intentional behavior changes. The review
// and native-check pins are those stored by Experiments created before the
// source-hash implementation was replaced with explicit policy versions; the
// normalizer is version 3 since collection reads stored StageMetrics strictly
// and every counter as reported (EVM-R81).
func TestPolicyPinsPreserveExistingExperiments(t *testing.T) {
	for _, test := range []struct {
		name string
		got  string
		want string
	}{
		{"human review", HumanPolicySHA256(), "sha256:25740d3d9f24f4894b4f44d4e90585df006712bc84017be40114c6c1fea6d8c1"},
		{"native checks", NativePolicySHA256(), "sha256:d63f0a408b377c728a9d472819216fd3f34d666bd251321ee3ccadeb66db766f"},
		{"normalizers", NormalizerSHA256(), "sha256:e980902feccb330a94376ecae497965d1698a1d9b20faf3479eb9b573db01295"},
		{"normalizer version", normalizerID, "contractor-output@3"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.got != test.want {
				t.Fatalf("pin = %q, want %q", test.got, test.want)
			}
		})
	}
}
