package privateartifacts

import (
	"context"
	"github.com/grauwolf32/contractor/internal/gatewayrecovery"
)

const maximumRecoveryRequestBytes = 4096

type gatewayRecoveryUpdater interface {
	Update(context.Context, string, gatewayrecovery.Request) (gatewayrecovery.Decision, error)
}
