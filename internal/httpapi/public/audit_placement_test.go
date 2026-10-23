package public

import "testing"

func TestAuditPlacementDigestUsesCanonicalSelection(t *testing.T) {
	request := createAuditRequest{RequiredPlacementLabels: []string{"cloud/aws", "cloud"}}
	first, _, err := createAuditRequestDigest("project", request)
	if err != nil {
		t.Fatal(err)
	}
	request.RequiredPlacementLabels = []string{"cloud", "cloud/aws"}
	reordered, _, err := createAuditRequestDigest("project", request)
	if err != nil || reordered != first {
		t.Fatalf("equivalent selection changed digest: %v", err)
	}
	request.RequiredPlacementLabels = []string{"local"}
	changed, _, err := createAuditRequestDigest("project", request)
	if err != nil || changed == first {
		t.Fatalf("different selection reused digest: %v", err)
	}
	request.RequiredPlacementLabels = nil
	omitted, _, err := createAuditRequestDigest("project", request)
	if err != nil {
		t.Fatal(err)
	}
	request.RequiredPlacementLabels = []string{}
	empty, _, err := createAuditRequestDigest("project", request)
	if err != nil || empty != omitted {
		t.Fatalf("empty selection differs from omission: %v", err)
	}
	request.RequiredPlacementLabels = []string{"cloud", "local"}
	if _, _, err := createAuditRequestDigest("project", request); err == nil {
		t.Fatal("mixed namespaces were accepted")
	}
}
