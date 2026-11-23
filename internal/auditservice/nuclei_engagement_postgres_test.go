package auditservice

import (
	"strings"
	"testing"
)

func TestEngagementNucleiStartDerivesControlledHTTPFromTools(t *testing.T) {
	f := newEngagementFixtureWithEdits(t, func(worker string) string {
		return strings.Replace(worker, "- ref: run-artifacts@1\n      tools: [read_artifact, write_artifact]", "- ref: scan@1\n      tools: [scan_nuclei]", 1)
	}, nil)
	draft, err := f.create(t, "nuclei-engagement", f.document("http://127.0.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.service.Start(f.ctx, f.start(draft)); err != nil {
		t.Fatalf("Start refused the controlled Nuclei path: %v", err)
	}
}
