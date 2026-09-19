// Package evalschema contains the local, versioned managed evaluation contract.
// Portable schemas are pinned data; no evaluator package or network is needed.
package evalschema

import "embed"

// Files is the immutable schema catalog embedded into the Server binary.
//
//go:embed managed.schema.json portable/*.schema.json portable/provenance.json
var Files embed.FS

const ManagedID = "urn:contractor:eval:v1"
