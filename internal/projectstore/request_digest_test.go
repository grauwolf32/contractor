package projectstore

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// TestCreateRequestDigestIsPinned pins what a Project creation key binds, so
// a change to it is deliberate rather than a side effect of a refactoring:
// stored receipts compare these values.
func TestCreateRequestDigestIsPinned(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		params CreateParams
		want   string
	}{
		{"omitted policy", CreateParams{Kind: "repository", Name: "demo"},
			"sha256:9e884a914037d5512900231cd8bb2290c6f066b9f0c49ed3a8d79db137ebc9b9"},
		{"automatic policy", CreateParams{Kind: "repository", Name: "demo", ActiveChecks: contracts.ActiveChecksAutomatic},
			"sha256:9e884a914037d5512900231cd8bb2290c6f066b9f0c49ed3a8d79db137ebc9b9"},
		{"description", CreateParams{Kind: "repository", Name: "demo", Description: "d", ActiveChecks: contracts.ActiveChecksAutomatic},
			"sha256:cf4b050bd10572d7fb405eb01951fde8bc3b23308db747c96b705509aeeefe67"},
	} {
		// The Project, owner and key are not part of the request body.
		tc.params.ProjectID, tc.params.OwnerID, tc.params.IdempotencyKey = "project-1", "user-1", "key-1"
		got, err := CreateRequestDigest(tc.params)
		if err != nil || got != tc.want {
			t.Errorf("%s digest = %s (%v), want %s", tc.name, got, err, tc.want)
		}
	}
	other, err := CreateRequestDigest(CreateParams{Kind: "repository", Name: "demo", ActiveChecks: contracts.ActiveChecksProhibited})
	if err != nil || other == "sha256:9e884a914037d5512900231cd8bb2290c6f066b9f0c49ed3a8d79db137ebc9b9" {
		t.Fatalf("another active-checks policy shares the digest (%v)", err)
	}
}
