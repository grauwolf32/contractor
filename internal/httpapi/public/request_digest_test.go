package public

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/projectstore"
)

// TestRequestDigestsArePinned guards idempotency replay: request digests are
// stored with each mutation, so their encoding must not change.
func TestRequestDigestsArePinned(t *testing.T) {
	project, err := projectRequestDigest(createProjectRequest{Kind: projectstore.Kind("repository"), Name: "demo"})
	if err != nil {
		t.Fatal(err)
	}
	run, err := createRunRequestDigest(createRunRequest{Workflow: "scan", RuntimeLabels: runRuntimeLabels{"linux"}})
	if err != nil {
		t.Fatal(err)
	}
	projectID := "project-1"
	projectRun, err := createRunRequestDigestForProject(createRunRequest{Workflow: "scan"}, &projectID)
	if err != nil {
		t.Fatal(err)
	}
	audit, _, err := createAuditRequestDigest(projectID, createAuditRequest{
		Profile: auditProfileSelectorRequest{Name: "profile"},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, got, want string }{
		{"project", project, "sha256:50fa781fd06af1977e1ce9d540bb407870096abd97428af01b2c05bfecc31c3c"},
		{"run", run, "sha256:1b762ef9ccd8b775816c9e4eec1ce46c60d08115647966ac58cccc40aecb711a"},
		{"project run", projectRun, "sha256:558a08c1a75206f4c99e8318fcad51e411d581b4fbf87b4614806a31299d05b9"},
		{"audit create", audit, "sha256:6cbc66c38782dec0a016581539d2fa6602c3ecdff50d6ac91291f98dbe5015ba"},
		{"audit start", auditRequestDigest("", "audit-1", 3), "sha256:aca7ac166a41c4b091572c8db65eaaf33cea54bdfc8f8814199d0d0a6dc99106"},
		{"audit start limit", auditRequestDigest("", "audit-1", 3, timeLimitPointer(0)), "sha256:8575b62c8a59e46ce0d005766c18fcb8070f518e18d3352d1c12f3a24bdef50c"},
		{"audit pause", auditRequestDigest("pause", "audit-1", 3), "sha256:d651d1f2a09390f72cb8e3bb6847ded96ba65d4c561fe5922a19811bb8e4219a"},
		{"audit resume limit", auditRequestDigest("resume", "audit-1", 3, timeLimitPointer(3600)), "sha256:55de97d14069029d34332b1642373072a616e96c0b091f990effb512f881c443"},
		{"review", reviewRequestDigest("create", "audit-1", "finding-1", uint64(2), map[string]string{"note": "x"}), "sha256:292e2a42c275f051ceac32c485c3b09bafb12458adfb4379f97b2a5d075e14a4"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s digest = %s, want %s", tc.name, tc.got, tc.want)
		}
	}
}
