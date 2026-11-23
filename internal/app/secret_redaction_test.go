package app

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The parsed inputs and Config redact their secrets for every fmt verb, flag
// and nesting, for log/slog and for encoding/json (SEC-R88); before V414-011
// %v of the inputs printed the bearer tokens.
func TestServeInputsAndConfigRedactSecretsForEveryVerbAndEncoder(t *testing.T) {
	t.Parallel()
	inputs := serveConfigInputs{
		databaseURL:       "postgres://contractor:hunter2-db@database/contractor",
		publicBearerToken: contracts.NewSecretString("hunter2-bearer"),
	}
	contractstest.AssertRedacted(t, inputs, "hunter2-bearer", "app.serveConfigInputs([REDACTED])")
	contractstest.AssertRedacted(t, inputs, "hunter2-db", "")
	config := Config{
		DatabaseURL:       contracts.NewSecretString("postgres://contractor:hunter2-db@database/contractor"),
		PublicBearerToken: contracts.NewSecretString("hunter2-bearer"),
	}
	contractstest.AssertRedacted(t, config, "hunter2-db", "")
	contractstest.AssertRedacted(t, migrationInputs{databaseURL: "hunter2-db"}, "hunter2-db", "app.migrationInputs([REDACTED])")
	contractstest.AssertRedacted(t, blobCleanupConfig{databaseURL: "hunter2-db"}, "hunter2-db", "app.blobCleanupConfig([REDACTED])")
}
