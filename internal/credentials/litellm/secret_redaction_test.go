package litellm

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
)

// The admin key and every holder of it redact themselves for every fmt verb,
// flag and nesting, for log/slog and for encoding/json (SEC-R88); before
// V414-011 %v of an exactBinding printed the key.
func TestAdminKeyHoldersRedactForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	key := adminKey{value: "hunter2-admin"}
	contractstest.AssertRedacted(t, key, "hunter2-admin", "[REDACTED]")
	binding := exactBinding{managementURL: "https://gateway.example.test", key: key}
	contractstest.AssertRedacted(t, binding, "hunter2-admin",
		`litellm.exactBinding{managementURL:"https://gateway.example.test" key:[REDACTED]}`)
	bindings := &AdminBindings{bindings: map[llmgateway.LLMGatewayConfigRef]exactBinding{{GatewayID: "gateway"}: binding}}
	contractstest.AssertRedacted(t, bindings, "hunter2-admin", "litellm.AdminBindings([REDACTED])")
	contractstest.AssertRedacted(t, *bindings, "hunter2-admin", "litellm.AdminBindings([REDACTED])")
	manager := &Manager{bindings: bindings}
	contractstest.AssertRedacted(t, manager, "hunter2-admin", "litellm.Manager([REDACTED])")
}
