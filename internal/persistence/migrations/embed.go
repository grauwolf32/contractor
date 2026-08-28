// Package migrations embeds Contractor's ordered, forward-only PostgreSQL
// migrations. Application code applies them through persistence/postgres.
package migrations

import "embed"

// Files contains every numbered SQL migration in this package.
//
//go:embed *.sql
var Files embed.FS
