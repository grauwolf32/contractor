package llmgateway

import (
	"testing"
)

func TestGatewayFailureSignatureUnicodeCategories(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name  string
		char  rune
		valid bool
	}{
		{name: "control", char: '\x00'},
		{name: "zero-width space", char: '\u200b'},
		{name: "byte-order mark", char: '\ufeff'},
		{name: "soft hyphen", char: '\u00ad'},
		{name: "private use", char: '\ue000'},
		{name: "unassigned", char: '\u0378', valid: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			signature := GatewayFailureSignature{
				Status: 400, MessageEquals: "Model" + string(test.char) + " unavailable",
			}
			if err := signature.validate(); (err == nil) != test.valid {
				t.Fatalf("valid=%t for %U: %v", test.valid, test.char, err)
			}
		})
	}
}
