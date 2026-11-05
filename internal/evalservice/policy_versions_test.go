package evalservice

import "testing"

// Policy pins change only with intentional behavior changes. Native V2 uses
// common Audit schemas; normalizer V3 retains exact reported counters.
func TestPolicyPinsPreserveExistingExperiments(t *testing.T) {
	for _, test := range []struct {
		name string
		got  string
		want string
	}{
		{"human review", HumanPolicySHA256(), "sha256:25740d3d9f24f4894b4f44d4e90585df006712bc84017be40114c6c1fea6d8c1"},
		{"native checks", NativePolicySHA256(), "sha256:8694fa9204632b1ad79072d685d20c6e14cbcfcda21179ce013f14d0a728b17e"},
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
