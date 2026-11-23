package contractstest

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// redactionVerbs are every fmt verb and redactionFlags the flag and width
// combinations that AssertRedacted formats a secret holder with (SEC-R88).
var (
	redactionVerbs = []rune("vsqxXdtoOcUbeEfFgGp")
	redactionFlags = []string{"", "+", "-", "#", " ", "0", "10", "-10", ".3", "+#", "#10"}
)

// AssertRedacted formats value, which holds plaintext, with every fmt verb,
// flag and width, directly, through a pointer, in an exported struct field, a
// map value and key, a slice and fmt.Errorf, logs it through log/slog's JSON
// and text handlers and encodes it with encoding/json, and fails when any
// output contains plaintext or its decimal, octal or hex byte encoding. When
// want is not empty, %v, %+v, %s and %d of value must print exactly want, the
// holder's redacted form.
func AssertRedacted(t testing.TB, value any, plaintext, want string) {
	t.Helper()
	if plaintext == "" {
		t.Fatal("AssertRedacted needs the plaintext the value holds")
	}
	if want != "" {
		for _, verb := range []string{"%v", "%+v", "%s", "%d"} {
			if got := fmt.Sprintf(verb, value); got != want {
				t.Errorf("%s = %q, want the redacted form %q", verb, got, want)
			}
		}
	}
	valueType := reflect.TypeOf(value)
	pointer := reflect.New(valueType)
	pointer.Elem().Set(reflect.ValueOf(value))
	wrapper := reflect.New(reflect.StructOf([]reflect.StructField{{Name: "Secret", Type: valueType}})).Elem()
	wrapper.Field(0).Set(reflect.ValueOf(value))
	mapValue := reflect.MakeMap(reflect.MapOf(reflect.TypeFor[string](), valueType))
	mapValue.SetMapIndex(reflect.ValueOf("key"), reflect.ValueOf(value))
	slice := reflect.MakeSlice(reflect.SliceOf(valueType), 0, 1)
	slice = reflect.Append(slice, reflect.ValueOf(value))
	forms := map[string]any{
		"value": value, "pointer": pointer.Interface(), "struct field": wrapper.Interface(),
		"struct pointer": wrapper.Addr().Interface(), "map value": mapValue.Interface(), "slice": slice.Interface(),
	}
	if valueType.Comparable() {
		mapKey := reflect.MakeMap(reflect.MapOf(valueType, reflect.TypeFor[bool]()))
		mapKey.SetMapIndex(reflect.ValueOf(value), reflect.ValueOf(true))
		forms["map key"] = mapKey.Interface()
	}
	for name, form := range forms {
		for _, verb := range redactionVerbs {
			if verb == 'p' && reflect.TypeOf(form).Kind() != reflect.Pointer {
				// fmt answers %p of a non-pointer through badVerb, which
				// never calls Format and prints the raw fields: a method
				// cannot redact it (recorded in V414-011).
				continue
			}
			for _, flags := range redactionFlags {
				directive := "%" + flags + string(verb)
				assertNoPlaintext(t, name+" "+directive, fmt.Sprintf(directive, form), plaintext)
			}
		}
		assertNoPlaintext(t, name+" fmt.Errorf", fmt.Errorf("wrapped: %w %d %v", errors.New("cause"), form, form).Error(), plaintext)
		assertNoPlaintext(t, name+" fmt.Sprint", fmt.Sprint(form)+fmt.Sprintln(form), plaintext)
		var logged bytes.Buffer
		slog.New(slog.NewJSONHandler(&logged, nil)).Info("secret", "value", form)
		slog.New(slog.NewTextHandler(&logged, nil)).Info("secret", "value", form)
		assertNoPlaintext(t, name+" log/slog", logged.String(), plaintext)
		encoded, err := json.Marshal(form)
		if err != nil {
			encoded = []byte(err.Error())
		}
		assertNoPlaintext(t, name+" encoding/json", string(encoded), plaintext)
	}
}

func assertNoPlaintext(t testing.TB, form, output, plaintext string) {
	t.Helper()
	data := []byte(plaintext)
	encodings := map[string]string{
		"text": plaintext, "hex": hex.EncodeToString(data), "upper-case hex": strings.ToUpper(hex.EncodeToString(data)),
		"decimal bytes": byteSequence(data, 10, " "), "octal bytes": byteSequence(data, 8, " "),
		"spaced hex": byteSequence(data, 16, " "),
	}
	for kind, encoding := range encodings {
		// The message names the encoding only: a test never logs a secret.
		if encoding != "" && strings.Contains(output, encoding) {
			t.Errorf("%s exposes the plaintext as %s", form, kind)
			return
		}
	}
}

// byteSequence writes the first bytes of data in base, as fmt prints a byte
// slice with %d, %o or % x.
func byteSequence(data []byte, base int, separator string) string {
	data = data[:min(len(data), 6)]
	parts := make([]string, len(data))
	for index, value := range data {
		parts[index] = strconv.FormatUint(uint64(value), base)
	}
	return strings.Join(parts, separator)
}
