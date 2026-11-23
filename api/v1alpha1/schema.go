// Package wireschema embeds shared wire primitives for document validators.
package wireschema

import _ "embed"

// Common contains the authoritative name, reference and media type definitions.
//
//go:embed common.schema.json
var Common []byte

// MaterialSet contains the common exact material descriptor and flat set schema.
//
//go:embed material-set.schema.json
var MaterialSet []byte
