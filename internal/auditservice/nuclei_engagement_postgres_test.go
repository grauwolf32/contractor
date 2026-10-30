package auditservice

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

func TestEngagementNucleiStartRequiresControlledHTTPAssignment(t *testing.T) {
	for _, assigned := range []bool{false, true} {
		t.Run(map[bool]string{false: "metadata_only", true: "http_assignment"}[assigned], func(t *testing.T) {
			f := newEngagementFixtureWithEdits(t, func(worker string) string {
				return strings.Replace(worker,
					"- ref: run-artifacts@1\n      tools: [read_artifact, write_artifact]",
					"- ref: scan@1\n      tools: [scan_nuclei]", 1)
			}, nil)
			document := f.document("http://127.0.0.1")
			if assigned {
				document.Policy.RequiredCapabilities = append(document.Policy.RequiredCapabilities,
					string(runtimesettings.RuntimeAdapterHTTPTarget))
			}
			draft, err := f.create(t, "nuclei-engagement", document)
			if err != nil {
				t.Fatal(err)
			}
			_, err = f.service.Start(f.ctx, f.start(draft))
			if assigned {
				if err != nil {
					t.Fatalf("Start refused the controlled Nuclei path: %v", err)
				}
			} else {
				var unsupported *UnsupportedError
				if !errors.As(err, &unsupported) ||
					!slices.Contains(unsupported.Reasons, ReasonEngagementCapabilityUnsupported) {
					t.Fatalf("metadata-only engagement admitted network execution: %v", err)
				}
			}
		})
	}
}
