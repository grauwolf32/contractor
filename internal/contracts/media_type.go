package contracts

import "regexp"

// mediaTypePattern is the RFC 6838 restricted-name grammar in canonical
// lowercase, without parameters or wildcards. The shared cases in
// api/testdata/v1alpha1/media-type-cases.json hold Go, Python, the schemas and
// the UI to the same grammar.
var mediaTypePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$`)

// ValidMediaType reports whether value is a canonical type/subtype media type.
func ValidMediaType(value string) bool {
	return mediaTypePattern.MatchString(value)
}
