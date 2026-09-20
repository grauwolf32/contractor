package config

import "testing"

func TestScanToolsetDescriptor(t *testing.T) {
	t.Parallel()
	descriptors, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	descriptor, ok := descriptors.Toolsets["scan@1"]
	if !ok {
		t.Fatal("scan@1 descriptor is missing")
	}
	want := []string{"scan_ffuf", "scan_katana", "scan_naabu", "scan_nuclei", "scan_sqlmap"}
	if !equalStrings(descriptor.Tools, want) || !equalStrings(descriptor.ActiveCheckTools, want) {
		t.Fatalf("scan@1 must classify each scanner as an active check: %+v", descriptor)
	}
	for _, name := range want {
		if got := descriptor.InfrastructureChannels[name]; len(got) != 1 || got[0] != RuntimeSubprocessLauncher {
			t.Fatalf("%s channels = %v, want RuntimeSubprocessLauncher", name, got)
		}
	}
}
