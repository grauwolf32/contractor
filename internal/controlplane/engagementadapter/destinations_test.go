package engagementadapter

import (
	"context"
	"errors"
	"net/netip"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/controlplane"
)

func TestProtectedEngagementDestinationsDenyAliasesWithoutSecrets(t *testing.T) {
	calls := 0
	lookup := func(_ context.Context, network, host string) ([]netip.Addr, error) {
		calls++
		if network != "ip" || host != "internal.example" {
			t.Fatal(network, host)
		}
		return []netip.Addr{netip.MustParseAddr("10.1.2.3"), netip.MustParseAddr("10.1.2.3"), netip.MustParseAddr("2001:db8::1")}, nil
	}
	got, err := resolveEngagementDestinations(t.Context(), []string{"https://internal.example/secret-path?token=private", "https://internal.example:443/other", "http://192.168.1.2:9000/api"}, lookup)
	expected := []engagementcontract.ProtectedDestination{{Hostname: "internal.example", Port: 443, Addresses: []string{"10.1.2.3", "2001:db8::1"}}, {Hostname: "192.168.1.2", Port: 9000, Addresses: []string{"192.168.1.2"}}}
	if err != nil || calls != 1 || !reflect.DeepEqual(got, expected) {
		t.Fatalf("destinations=%+v calls=%d err=%v", got, calls, err)
	}
}
func TestProtectedEngagementDestinationsFailClosed(t *testing.T) {
	for _, mode := range []string{"unresolved", "empty", "too-many", "bad-port", "bad-scheme"} {
		t.Run(mode, func(t *testing.T) {
			endpoint := "https://internal.example/private"
			if mode == "bad-port" {
				endpoint = "https://internal.example:70000/private"
			}
			if mode == "bad-scheme" {
				endpoint = "ftp://internal.example/private"
			}
			lookup := func(context.Context, string, string) ([]netip.Addr, error) {
				if mode == "unresolved" {
					return nil, errors.New("resolver failed secret")
				}
				if mode == "empty" {
					return nil, nil
				}
				if mode == "too-many" {
					return make([]netip.Addr, 33), nil
				}
				return []netip.Addr{netip.MustParseAddr("10.0.0.1")}, nil
			}
			_, err := resolveEngagementDestinations(t.Context(), []string{endpoint}, lookup)
			if err == nil || strings.Contains(err.Error(), "secret") || strings.Contains(err.Error(), "/private") {
				t.Fatalf("unsafe failure: %v", err)
			}
		})
	}
}
func TestProtectedEngagementDestinationsIncludeEveryInfrastructureSurface(t *testing.T) {
	settings := runtimesettings.RuntimeSettings{ArtifactAPIURL: "https://192.168.1.1", LLMGatewayURL: "https://192.168.1.2", Telemetry: &runtimesettings.TelemetrySettings{Endpoint: "https://192.168.1.3"}, Caido: &runtimesettings.CaidoSettings{Endpoint: "https://192.168.1.4"}, HTTPProxy: &runtimesettings.HTTPProxySettings{ProxyURL: "http://192.168.1.5:8080"}, Interactsh: &runtimesettings.InteractshSettings{Endpoint: "https://192.168.1.6"}}
	result, err := resolveProtectedDestinations(t.Context(), controlplane.Reservation{ControlURL: "https://192.168.1.7", A2AURL: "https://192.168.1.8"}, settings)
	if err != nil || len(result) != 8 {
		t.Fatalf("infrastructure surfaces=%d err=%v", len(result), err)
	}
}
