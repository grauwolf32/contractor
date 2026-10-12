package contracts

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Bounds of the relative workspace path grammar (WSP-R13): allocation
// workspace targets and the files a Worker reports reading (LIM-WSP-2,
// LIM-WSP-3), and the larger Worker State workspace paths.
const (
	MaxWorkspaceTargetBytes      = 1024
	MaxWorkspaceTargetComponents = 32
	MaxWorkerStatePathBytes      = 4096
	MaxWorkerStatePathComponents = 128
)

// RelativePathProblem returns why value is not a relative workspace path of at
// most maxBytes UTF-8 bytes and maxComponents components, or "" when it is one
// (WSP-R13): non-empty NFC UTF-8 without a leading slash, backslash, NUL or
// URI, whose components are neither empty, "." nor ".." and hold no control
// character, and whose first component is not a Windows drive (an ASCII letter
// and a colon). It is the one Go definition of the grammar. The Runtime's
// contracts.base.relative_path_problem applies the same rule with the same
// problem texts, and api/testdata/v1alpha1/workspace-path-cases.json pins both.
// The byte bound is checked first, so the work is bounded by maxBytes.
func RelativePathProblem(value string, maxBytes, maxComponents int) string {
	switch {
	case value == "":
		return "must not be empty"
	case !utf8.ValidString(value):
		return "must be valid UTF-8"
	case len(value) > maxBytes:
		return fmt.Sprintf("exceeds the byte bound of %d UTF-8 bytes", maxBytes)
	case value != norm.NFC.String(value):
		return "must be NFC-normalized"
	case strings.HasPrefix(value, "/") || strings.ContainsAny(value, "\\\x00") || strings.Contains(value, "://"):
		return "must be a relative POSIX path without a backslash, NUL or URI"
	}
	parts := strings.Split(value, "/")
	if len(parts) > maxComponents {
		return fmt.Sprintf("exceeds the depth bound of %d components", maxComponents)
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return "contains an invalid component: empty, '.' or '..'"
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return "contains a control character"
			}
		}
	}
	if first := parts[0]; len(first) >= 2 && isASCIILetter(first[0]) && first[1] == ':' {
		return "must not start with a Windows drive"
	}
	return ""
}

func isASCIILetter(character byte) bool {
	return character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z'
}
