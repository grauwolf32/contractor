// Package auditschema embeds the strict Audit document contract used by offline validation.
package auditschema

import _ "embed"

// Document is the single schema for Audit programs, tasks, trusted context and results.
//
//go:embed contract.schema.json
var Document []byte
