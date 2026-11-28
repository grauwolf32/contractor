package contracts

import (
	"strings"
	"unicode/utf8"
)

const MaxFindingCheckAssertionBytes = 32 << 10

// FindingCheck binds one assertion and immutable source to an ordinary Run.
// Identity, source membership and revision authority are checked by the owner.
type FindingCheck struct {
	FindingID string `json:"findingId"`
	Revision  uint64 `json:"revision"`
	ReceiptID string `json:"receiptId"`
	Assertion string `json:"assertion"`
}

func (v FindingCheck) Validate() error {
	if !ValidResourceID(v.FindingID) || !ValidResourceID(v.ReceiptID) || v.Revision < 1 || v.Revision > MaxExactInteger || !utf8.ValidString(v.Assertion) || strings.TrimSpace(v.Assertion) == "" || len(v.Assertion) > MaxFindingCheckAssertionBytes || strings.ContainsRune(v.Assertion, 0) {
		return Invalidf("invalid Finding check")
	}
	return nil
}
