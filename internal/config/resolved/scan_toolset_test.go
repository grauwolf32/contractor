package resolved

import (
	"slices"
	"testing"
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
		if got := descriptor.InfrastructureChannels[name]; len(got) != 1 || got[0] != channelRuntimeSubprocessLauncher {
			t.Fatalf("%s channels = %v, want RuntimeSubprocessLauncher", name, got)
		}
	}
}
