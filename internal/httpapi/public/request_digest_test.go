package public

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestRequestDigestsArePinned pins the canonical encoding of each request
// digest, so a change to what an idempotency key binds is deliberate rather
// than a side effect of a refactoring. The Run creation digest is pinned by
// runservice, which owns it.
func TestRequestDigestsArePinned(t *testing.T) {
	project, err := projectRequestDigest(createProjectRequest{Kind: projectstore.Kind("repository"), Name: "demo"})
	if err != nil {
		t.Fatal(err)
	}
	projectID := "project-1"
	audit, _, err := createAuditRequestDigest(projectID, createAuditRequest{
		Profile: auditProfileSelectorRequest{Name: "profile"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, got, want string }{
		{"project", project, "sha256:9e884a914037d5512900231cd8bb2290c6f066b9f0c49ed3a8d79db137ebc9b9"},
		{"audit create", audit, "sha256:6cbc66c38782dec0a016581539d2fa6602c3ecdff50d6ac91291f98dbe5015ba"},
		{"audit start", auditRequestDigest("start", "audit-1", 3), "sha256:aaa21d0388ca0a5e7fc3b9b66ba8cd9f93ba2847c42950363e601b547bdf54be"},
		{"audit start limit", auditRequestDigest("start", "audit-1", 3, timeLimitPointer(0)), "sha256:c9864979e1a2a956eb06cf99e96075ed3eea75d72f9e3a5ad740f4713fab91be"},
		{"audit pause", auditRequestDigest("pause", "audit-1", 3), "sha256:d651d1f2a09390f72cb8e3bb6847ded96ba65d4c561fe5922a19811bb8e4219a"},
		{"audit resume limit", auditRequestDigest("resume", "audit-1", 3, timeLimitPointer(3600)), "sha256:55de97d14069029d34332b1642373072a616e96c0b091f990effb512f881c443"},
		{"review", reviewRequestDigest("create", "audit-1", "finding-1", uint64(2), map[string]string{"note": "x"}), "sha256:292e2a42c275f051ceac32c485c3b09bafb12458adfb4379f97b2a5d075e14a4"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s digest = %s, want %s", tc.name, tc.got, tc.want)
		}
	}
}

// Every Audit lifecycle request binds its action, a start included: no action
// is encoded by omission.
func TestAuditRequestDigestBindsEveryAction(t *testing.T) {
	seen := map[string]string{}
	for _, action := range []string{"start", "pause", "resume", "cancel", "delete"} {
		digest := auditRequestDigest(action, "audit-1", 3)
		want := contentdigest.Bytes([]byte(`{"action":"` + action + `","auditId":"audit-1","revision":3}`))
		if digest != want {
			t.Errorf("%s digest = %s, want the digest of its canonical request %s", action, digest, want)
		}
		if other, found := seen[digest]; found {
			t.Errorf("%s and %s share a request digest", action, other)
		}
		seen[digest] = action
	}
}
