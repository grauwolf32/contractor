package contracts_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestValidGoldenFixtures(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Valid {
		decode := fixtureCodecs[entry.Type].roundTrip
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			input := contractstest.ReadFixture(t, "valid", filename)
			output, err := decode(input)
			if err != nil {
				t.Fatalf("decode valid fixture: %v", err)
			}
			contractstest.AssertSemanticJSONEqual(t, input, output)
		})
	}
}

func TestInvalidGoldenFixtures(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Invalid {
		if !entry.Strict {
			continue
		}
		decode := fixtureCodecs[entry.Type].reject
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			if err := decode(contractstest.ReadFixture(t, "invalid", filename)); err == nil {
				t.Fatal("invalid fixture was accepted")
			}
		})
	}
}

func roundTrip[T contracts.Validatable](data []byte) ([]byte, error) {
	value, err := contracts.DecodeStrict[T](data)
	if err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

func reject[T contracts.Validatable](data []byte) error {
	_, err := contracts.DecodeStrict[T](data)
	return err
}

func TestPrivateValidCanonicalFixtures(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Valid {
		roundTrip := fixtureCodecs[entry.Type].privateRoundTrip
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			raw := contractstest.ReadFixture(t, "valid", filename)
			canonical, err := roundTrip(raw)
			if err != nil {
				t.Fatalf("decode valid private fixture: %v", err)
			}
			if !bytes.Equal(canonical, bytes.TrimSpace(raw)) {
				t.Fatalf("fixture is not the shared canonical form\n got: %s\nwant: %s", canonical, raw)
			}
		})
	}
}

func TestPrivateInvalidFixturesHaveSafeReasonClasses(t *testing.T) {
	t.Parallel()

	for filename, entry := range readFixtureIndex(t).Invalid {
		if entry.Reason == "" {
			continue
		}
		reject, class := fixtureCodecs[entry.Type].privateReject, contracts.PrivateProtocolErrorClass(entry.Reason)
		t.Run(filename, func(t *testing.T) {
			t.Parallel()
			err := reject(contractstest.ReadFixture(t, "invalid", filename))
			if err == nil {
				t.Fatal("invalid private fixture was accepted")
			}
			var privateError *contracts.PrivateProtocolError
			if !errors.As(err, &privateError) || privateError.Class != class {
				t.Fatalf("error class = %v, want %v", privateError, class)
			}
			formatted := fmt.Sprintf("%v %+v %#v", err, err, err)
			for _, canary := range []string{
				"recognizable-secret-canary", "recognizable-provenance-secret",
				"proxy-password-canary", "proxy-bearer-canary", "unknown-secret-adapter",
				"caido-invalid-secret-canary",
			} {
				if strings.Contains(formatted, canary) {
					t.Fatalf("private validation error leaked secret input: %s", formatted)
				}
			}
		})
	}
}

func privateRoundTrip[T contracts.Validatable](data []byte) ([]byte, error) {
	value, err := contracts.DecodePrivateStrict[T](data)
	if err != nil {
		return nil, err
	}
	return contracts.MarshalPrivateCanonical(value)
}

func privateReject[T contracts.Validatable](data []byte) error {
	_, err := contracts.DecodePrivateStrict[T](data)
	return err
}
