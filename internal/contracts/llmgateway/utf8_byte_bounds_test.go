package llmgateway

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedFailureSignatureTextCountsUTF8Bytes checks both texts of a
// Gateway failure signature (LIM-RCF-1) with each shared character at and
// one byte over the declared bound.
func TestSharedFailureSignatureTextCountsUTF8Bytes(t *testing.T) {
	t.Parallel()
	characters, fields := contractstest.ReadUTF8ByteBoundCases(t)
	checked := map[string]bool{}
	for _, field := range fields {
		if field.Rule != "LIM-RCF-1" {
			continue
		}
		if field.MaxBytes != MaximumGatewayFailureSignatureText {
			t.Fatalf("%s %v: shared bound %d, Go bound %d", field.Schema, field.Path, field.MaxBytes, MaximumGatewayFailureSignatureText)
		}
		name := field.Path[len(field.Path)-1]
		checked[name] = true
		for _, value := range field.Values(characters) {
			signature := GatewayFailureSignature{Status: 404}
			switch name {
			case "messageEquals":
				signature.MessageEquals = value.Value
			case "litellmWrapped":
				signature.LiteLLMWrapped = value.Value
			default:
				t.Fatalf("unknown failure signature field %v", field.Path)
			}
			if err := signature.validate(); (err == nil) != value.Within {
				t.Errorf("%s of %d bytes of %q: %v, within %v", name, len(value.Value), value.Character, err, value.Within)
			}
		}
	}
	if len(checked) != 2 {
		t.Fatalf("shared failure signature fields checked: %v", checked)
	}
}
