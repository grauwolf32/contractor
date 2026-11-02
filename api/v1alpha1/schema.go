// Package wireschema embeds shared wire primitives for document validators.
package wireschema

import _ "embed"

// Common contains the authoritative name, reference and media type definitions.
//
//go:embed common.schema.json
var Common []byte
