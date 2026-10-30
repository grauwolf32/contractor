package resolved

import (
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestScanToolsetDescriptor(t *testing.T) {
	t.Parallel()
	descriptor, ok := registeredDescriptors().Toolsets["scan@1"]
	if !ok {
		t.Fatal("scan@1 descriptor is missing")
	}
	want := []string{"scan_ffuf", "scan_katana", "scan_naabu", "scan_nuclei", "scan_sqlmap"}
	if !slices.Equal(descriptor.Tools, want) || !slices.Equal(descriptor.ActiveCheckTools, want) {
		t.Fatalf("scan@1 must classify each scanner as an active check: %+v", descriptor)
	}
	for _, name := range want {
		channels := []infrastructureChannel{channelRuntimeSubprocessLauncher}
		if name == "scan_nuclei" {
			channels = []infrastructureChannel{channelRuntimeHTTPClient, channelRuntimeSubprocessLauncher}
		}
		if got := descriptor.InfrastructureChannels[name]; !slices.Equal(got, channels) {
			t.Fatalf("%s channels = %v, want %v", name, got, channels)
		}
	}
}

func TestControlledHTTPSelectionKeepsOtherScannersUncontrolled(t *testing.T) {
	for _, test := range []struct {
		tools        []string
		uncontrolled bool
	}{
		{[]string{"scan_nuclei"}, false},
		{[]string{"scan_ffuf"}, true},
		{[]string{"scan_sqlmap"}, true},
		{[]string{"scan_naabu"}, true},
		{[]string{"scan_katana"}, true},
		{[]string{"scan_nuclei", "scan_naabu"}, true},
	} {
		template := contracts.ResolvedAgentTemplate{
			SandboxProfile: contracts.SandboxProfileRef{SandboxProfileID: "local-workdir", Version: "1"},
			Toolsets: []contracts.ToolsetSelection{{
				Ref: contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}, Tools: test.tools,
			}},
		}
		if !TemplateRequiresNetwork(template) {
			t.Fatalf("standalone scanner lost network classification: %v", test.tools)
		}
		if got := TemplateRequiresUncontrolledNetwork(template); got != test.uncontrolled {
			t.Fatalf("tools=%v uncontrolled=%v want %v", test.tools, got, test.uncontrolled)
		}
	}
}
