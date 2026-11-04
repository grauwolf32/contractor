package auditservice

import (
	"strings"
	"unicode/utf8"
)

const MaxFindingRationale = 64 << 10

func validRationale(value string) bool {
	return strings.TrimSpace(value) != "" && utf8.ValidString(value) &&
		!strings.ContainsRune(value, 0) && len([]byte(value)) <= MaxFindingRationale
}
