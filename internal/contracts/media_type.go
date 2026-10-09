package contracts

import "regexp"

// MaxMediaTypeLength bounds a media type as the public MediaType schema does.
const MaxMediaTypeLength = 255

// FindingCollectionMediaType identifies a frozen finding collection ZIP. The
// Audit domain builds and publishes the package; the Toolset descriptor of the
// collection reader declares the Workflow input that must accept it.
const FindingCollectionMediaType = "application/vnd.contractor.findings-collection+zip"

// mediaTypePattern is the RFC 6838 restricted-name grammar in canonical
// lowercase, without parameters or wildcards. The shared cases in
// api/testdata/v1alpha1/media-type-cases.json hold Go, Python, the schemas and
// the UI to the same grammar and bound.
var mediaTypePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$`)

// ValidMediaType reports whether value is a canonical type/subtype media type
// of at most MaxMediaTypeLength characters.
func ValidMediaType(value string) bool {
	return len(value) <= MaxMediaTypeLength && mediaTypePattern.MatchString(value)
}
