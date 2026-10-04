package evalservice

import "testing"

// These are the pins stored by Experiments created before the source-hash
// implementation was replaced with explicit policy versions.
func TestPolicyPinsPreserveExistingExperiments(t *testing.T) {
	for _, test := range []struct {
		name string
		got  string
		want string
	}{
		{"human review", HumanPolicySHA256(), "sha256:25740d3d9f24f4894b4f44d4e90585df006712bc84017be40114c6c1fea6d8c1"},
		{"native checks", NativePolicySHA256(), "sha256:d63f0a408b377c728a9d472819216fd3f34d666bd251321ee3ccadeb66db766f"},
		{"normalizers", NormalizerSHA256(), "sha256:276d5fdb3b9da5c99ac30e1468bf28935fa1d287d72be935de64e6bbe9a73a8e"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if test.got != test.want {
				t.Fatalf("pin = %q, want %q", test.got, test.want)
			}
		})
	}
}
