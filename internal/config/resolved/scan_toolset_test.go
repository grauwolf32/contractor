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
	if !slices.Equal(descriptor.Tools, []string{"scan_ffuf", "scan_katana", "scan_naabu", "scan_nuclei", "scan_semgrep", "scan_sqlmap"}) || !slices.Equal(descriptor.ActiveCheckTools, want) {
		t.Fatalf("scan@1 must classify each scanner as an active check: %+v", descriptor)
	}
	for _, name := range want {
		channels := []infrastructureChannel{channelRuntimeSubprocessLauncher}
		if name == "scan_nuclei" || name == "scan_sqlmap" || name == "scan_ffuf" {
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
		{[]string{"scan_ffuf"}, false},
		{[]string{"scan_sqlmap"}, false},
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

func TestControlledHTTPSelectionIncludesIdentityAndSessionOperations(t *testing.T) {
	for _, test := range []struct {
		toolset, tool string
		want          bool
	}{
		{"http-tools", "http_request", true},
		{"http-tools", "http_credentials", true},
		{"http-tools", "http_credential_read", true},
		{"http-tools", "http_session_get", true},
		{"http-tools", "http_session_set", true},
		{"http-tools", "http_session_clear", true},
		{"scan", "scan_nuclei", true},
		{"scan", "scan_ffuf", true},
		{"scan", "scan_naabu", false},
		{"run-artifacts", "read_artifact", false},
	} {
		t.Run(test.tool, func(t *testing.T) {
			template := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: test.toolset, Version: "1"}, Tools: []string{test.tool}}}}
			if got := TemplateUsesControlledHTTP(template); got != test.want {
				t.Fatalf("HTTP owner needed=%v want=%v", got, test.want)
			}
		})
	}
}

func TestSemgrepIsLocalStaticAnalysis(t *testing.T) {
	template := contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{
		Ref: contracts.ToolsetRef{ToolsetID: "scan", Version: "1"}, Tools: []string{"scan_semgrep"},
	}}}
	if TemplateRequiresNetwork(template) || TemplateRequiresUncontrolledNetwork(template) || TemplateUsesControlledHTTP(template) {
		t.Fatal("Semgrep local rules must not require target network authority")
	}
}
