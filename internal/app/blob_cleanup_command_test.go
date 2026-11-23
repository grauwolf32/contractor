package app

import "testing"

func TestBlobCleanupRequiresExplicitOfflineApply(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "CONTRACTOR_DATABASE_URL":
			return "postgres://example"
		case "CONTRACTOR_ARTIFACT_BLOB_PATH":
			return "/var/blobs"
		}
		return ""
	}
	for _, tc := range []struct {
		args         []string
		valid, apply bool
	}{
		{[]string{"cleanup"}, true, false},
		{[]string{"cleanup", "--apply"}, false, false},
		{[]string{"cleanup", "--apply", "--offline"}, true, true},
		{[]string{"cleanup", "--artifact-blob-path=relative"}, false, false},
	} {
		cfg, err := parseBlobCleanupConfig(tc.args, env)
		if (err == nil) != tc.valid {
			t.Fatalf("args %v: %v", tc.args, err)
		}
		if err == nil && cfg.apply != tc.apply {
			t.Fatal("wrong apply mode")
		}
	}
}
