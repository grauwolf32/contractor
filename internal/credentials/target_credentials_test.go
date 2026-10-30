package credentials

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// TestSharedTargetCredentialMaterialCases holds the target credential
// constructors, the creation decoder and the stored-plaintext decoder to the
// material cases of api/testdata/v1alpha1/target-credential-cases.json
// (RCF-R184). A valid material seals, opens and re-decodes to the same
// canonical plaintext; an invalid one is refused by both decoders, and no
// refusal repeats a material value. Every Basic kind shares the
// target-basic@1 rule (RCF-R114), so its cases are decoded as
// http-origin-basic@1 and http-proxy-basic@1 too.
func TestSharedTargetCredentialMaterialCases(t *testing.T) {
	cases := contractstest.ReadTargetCredentialCases(t)
	cipher, err := NewTokenCipher(bytes.Repeat([]byte{0x75}, 32))
	if err != nil {
		t.Fatal(err)
	}
	kinds := map[RuntimeCredentialKind]int{}
	type decoded struct {
		contractstest.TargetCredentialCase
		kind RuntimeCredentialKind
	}
	var decodes []decoded
	for _, test := range cases.Materials {
		decodes = append(decodes, decoded{test, RuntimeCredentialKind(test.Kind)})
		if RuntimeCredentialKind(test.Kind) == RuntimeCredentialTargetBasic {
			decodes = append(decodes,
				decoded{test, RuntimeCredentialOriginBasic}, decoded{test, RuntimeCredentialProxyBasic})
		}
	}
	for _, request := range decodes {
		test, kind := request.TargetCredentialCase, request.kind
		test.Name = string(kind) + " " + test.Name
		material, err := DecodeRuntimeCredentialMaterial(kind, test.Data)
		if (err == nil) != test.Valid {
			t.Errorf("%s: creation decoder accepted = %t, want %t: %v", test.Name, err == nil, test.Valid, err)
			material.Destroy()
			continue
		}
		if err != nil {
			if !errors.Is(err, ErrRuntimeCredentialInvalid) {
				t.Errorf("%s: creation error = %v, want ErrRuntimeCredentialInvalid", test.Name, err)
			}
			for _, value := range materialStrings(test.Data) {
				if strings.Contains(err.Error(), value) {
					t.Errorf("%s: creation error repeats a material value: %v", test.Name, err)
				}
			}
			if _, storedErr := runtimeCredentialMaterialFromCanonical(kind, test.Data); !errors.Is(storedErr, ErrCrypto) {
				t.Errorf("%s: stored plaintext decoder error = %v, want ErrCrypto", test.Name, storedErr)
			}
			continue
		}
		kinds[material.Kind()]++
		if material.Kind() != kind {
			t.Errorf("%s: kind = %q, want %q", test.Name, material.Kind(), kind)
		}
		envelope, err := cipher.SealRuntimeCredential("target-case", material)
		if err != nil {
			t.Fatalf("%s: seal: %v", test.Name, err)
		}
		opened, err := cipher.OpenRuntimeCredential("target-case", kind, envelope)
		if err != nil {
			t.Fatalf("%s: open: %v", test.Name, err)
		}
		if err := material.WithPlaintext(func(_ RuntimeCredentialKind, canonical []byte) error {
			return opened.WithPlaintext(func(openedKind RuntimeCredentialKind, reopened []byte) error {
				if openedKind != kind || !bytes.Equal(canonical, reopened) {
					return fmt.Errorf("opened %q differs from the created material", openedKind)
				}
				return nil
			})
		}); err != nil {
			t.Errorf("%s: %v", test.Name, err)
		}
		material.Destroy()
		opened.Destroy()
	}
	for _, kind := range []RuntimeCredentialKind{
		RuntimeCredentialTargetBasic, RuntimeCredentialTargetBearer, RuntimeCredentialTargetCookies,
		RuntimeCredentialOriginBasic, RuntimeCredentialProxyBasic,
	} {
		if kinds[kind] == 0 {
			t.Errorf("no valid material case of %s", kind)
		}
	}
}

// materialStrings returns the string values of at least three bytes in a
// material that parses as JSON; member names are not secret.
func materialStrings(data []byte) []string {
	var document any
	if json.Unmarshal(data, &document) != nil {
		return nil
	}
	var values []string
	var walk func(any)
	walk = func(value any) {
		switch typed := value.(type) {
		case map[string]any:
			for _, child := range typed {
				walk(child)
			}
		case []any:
			for _, child := range typed {
				walk(child)
			}
		case string:
			if len(typed) >= 3 {
				values = append(values, typed)
			}
		}
	}
	walk(document)
	return values
}

func TestTargetCookiesMaterialIsWipedAfterCallbackFailureAndNeverFormatted(t *testing.T) {
	const canary = "canary-login-secret-41c9"
	material, err := NewTargetCookiesCredential(map[string]string{"password": canary})
	if err != nil {
		t.Fatal(err)
	}
	var seen []byte
	failure := errors.New("consumer failed")
	err = material.WithPlaintext(func(_ RuntimeCredentialKind, canonical []byte) error {
		seen = canonical
		return failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("callback error = %v", err)
	}
	if text := fmt.Sprintf("%v %+v %#v %s", material, material, material, material); strings.Contains(text, canary) {
		t.Fatalf("formatted material exposes the secret: %s", text)
	}
	if encoded, err := json.Marshal(material); err == nil || strings.Contains(string(encoded), canary) {
		t.Fatalf("material encoded: %s, %v", encoded, err)
	}
	material.Destroy()
	if strings.Contains(string(seen), canary) {
		t.Fatal("Destroy left the plaintext in the material buffer")
	}
}
