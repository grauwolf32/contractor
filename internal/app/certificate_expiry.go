package app

import (
	"fmt"
	"log/slog"
	"time"

	"github.com/grauwolf32/contractor/internal/mtls"
)

func logControlPlaneCertificateExpiry(
	logger *slog.Logger,
	certificatePath string,
	warningWindow time.Duration,
	now func() time.Time,
) error {
	expires, warning, err := mtls.LeafExpiry(certificatePath, now(), warningWindow)
	if err != nil {
		return fmt.Errorf("inspect Control Plane certificate expiry: %w", err)
	}
	logger.Info("Control Plane mTLS certificate expiry", "not_after", expires.UTC())
	if warning {
		logger.Warn("Control Plane mTLS certificate expires within warning window",
			"not_after", expires.UTC(), "warning_window", warningWindow)
	}
	return nil
}
