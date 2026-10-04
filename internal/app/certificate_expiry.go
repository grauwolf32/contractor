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
	caPath string,
	warningWindow time.Duration,
	now func() time.Time,
) error {
	at := now()
	expires, warning, err := mtls.LeafExpiry(certificatePath, at, warningWindow)
	if err != nil {
		return fmt.Errorf("inspect Control Plane certificate expiry: %w", err)
	}
	logger.Info("Control Plane mTLS certificate expiry", "not_after", expires.UTC())
	if warning {
		logger.Warn("Control Plane mTLS certificate expires within warning window",
			"not_after", expires.UTC(), "warning_window", warningWindow)
	}
	caExpires, caWarning, err := mtls.CAExpiry(caPath, at, warningWindow)
	if err != nil {
		return fmt.Errorf("inspect deployment CA expiry: %w", err)
	}
	logger.Info("deployment CA certificate expiry", "not_after", caExpires.UTC())
	if caWarning {
		// A leaf never outlives the CA, so CA expiry breaks every private mTLS
		// link and cannot be repaired by leaf renewal alone.
		logger.Warn("deployment CA certificate expires within warning window; rotate the CA and reissue leaves",
			"not_after", caExpires.UTC(), "warning_window", warningWindow)
	}
	return nil
}
