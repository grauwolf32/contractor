package app

import (
	"bytes"
	"log/slog"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/localpki"
)

func TestControlPlaneExpiryLogsAndWarnsWithInjectedClock(t *testing.T) {
	issued := time.Date(2026, time.January, 1, 0, 0, 0, 0, time.UTC)
	root := t.TempDir()
	generator := localpki.Generator{Now: func() time.Time { return issued }}
	if _, err := generator.InitCA(root, false); err != nil {
		t.Fatal(err)
	}
	caPaths := localpki.CAPaths(root)
	paths, err := generator.IssueControlPlane(root, localpki.ControlPlaneOptions{
		LeafOptions: localpki.LeafOptions{IPAddresses: []net.IP{net.ParseIP("127.0.0.1")}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		now    time.Time
		warn   bool
		caWarn bool
	}{
		{name: "far from expiry", now: issued.Add(100 * 24 * time.Hour)},
		{name: "inside leaf warning window", now: issued.Add(340 * 24 * time.Hour), warn: true},
		// The default CA lasts ten years; near its expiry both the leaf
		// (already renewed many times in practice) and the CA warn.
		{name: "inside CA warning window", now: issued.Add((10*365 - 20) * 24 * time.Hour), warn: true, caWarn: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer
			logger := slog.New(slog.NewTextHandler(&output, nil))
			if err := logControlPlaneCertificateExpiry(logger, paths.Certificate, caPaths.Certificate,
				30*24*time.Hour, func() time.Time { return test.now }); err != nil {
				t.Fatal(err)
			}
			logged := output.String()
			if !strings.Contains(logged, "level=INFO") ||
				!strings.Contains(logged, "Control Plane mTLS certificate expiry") ||
				!strings.Contains(logged, "deployment CA certificate expiry") {
				t.Fatalf("expiry log missing info lines = %q", logged)
			}
			if strings.Contains(logged, "Control Plane mTLS certificate expires within warning window") != test.warn {
				t.Fatalf("leaf warning = %q, want %v", logged, test.warn)
			}
			if strings.Contains(logged, "deployment CA certificate expires within warning window") != test.caWarn {
				t.Fatalf("CA warning = %q, want %v", logged, test.caWarn)
			}
		})
	}
}

func TestCertificateExpiryWarningWindowConfig(t *testing.T) {
	env := func(key string) string {
		if key == "CONTRACTOR_CERT_EXPIRY_WARNING_WINDOW" {
			return "48h"
		}
		return ""
	}
	cfg, err := ParseConfig(nil, env)
	if err != nil || cfg.CertificateExpiryWarningWindow != 48*time.Hour {
		t.Fatalf("environment warning window = %s, %v", cfg.CertificateExpiryWarningWindow, err)
	}
	cfg, err = ParseConfig([]string{"--cert-expiry-warning-window=24h"}, env)
	if err != nil || cfg.CertificateExpiryWarningWindow != 24*time.Hour {
		t.Fatalf("flag warning window = %s, %v", cfg.CertificateExpiryWarningWindow, err)
	}
	if _, err := ParseConfig([]string{"--cert-expiry-warning-window=0"}, env); err == nil {
		t.Fatal("zero warning window was accepted")
	}
}
