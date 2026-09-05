package e2e

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"go.yaml.in/yaml/v4"
)

type matrixGate struct {
	ID      string `yaml:"id"`
	Command string `yaml:"command"`
}

func decodeStrictMatrix[T any](data []byte, label string) (T, error) {
	var matrix T
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&matrix); err != nil {
		return matrix, fmt.Errorf("decode strict %s matrix: %w", label, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return matrix, fmt.Errorf("%s matrix contains a trailing YAML document", label)
	} else if !errors.Is(err, io.EOF) {
		return matrix, fmt.Errorf("decode %s matrix trailer: %w", label, err)
	}
	return matrix, nil
}

func mustDecodeStrictMatrix[T any](t *testing.T, data []byte, label string) T {
	t.Helper()
	matrix, err := decodeStrictMatrix[T](data, label)
	if err != nil {
		t.Fatal(err)
	}
	return matrix
}

func validateMatrixSymbolOwner(repositoryRoot, source, symbol string) error {
	if source == "" || symbol == "" || filepath.IsAbs(source) || strings.Contains(source, "..") {
		return fmt.Errorf("invalid test reference: source=%q symbol=%q", source, symbol)
	}
	extension := filepath.Ext(source)
	if !slices.Contains([]string{".go", ".py"}, extension) {
		return fmt.Errorf("unsupported test source: source=%q symbol=%q", source, symbol)
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(source)))
	if err != nil {
		return fmt.Errorf("read referenced test %q: %w", source, err)
	}
	prefix := "func "
	if extension == ".py" {
		prefix = "def "
	}
	if !bytes.Contains(data, []byte(prefix+symbol+"(")) {
		return fmt.Errorf("%s does not define %s", source, symbol)
	}
	return nil
}

func validateMatrixNamedOwner(repositoryRoot, source, kind, name string) error {
	if source == "" || name == "" || filepath.IsAbs(source) || strings.Contains(source, "..") {
		return fmt.Errorf("invalid test owner: source=%q kind=%q name=%q", source, kind, name)
	}
	extension := filepath.Ext(source)
	expectedExtensions := map[string][]string{
		"go_test":    {".go"},
		"pytest":     {".py"},
		"test_title": {".ts", ".tsx"},
	}
	allowedExtensions, exists := expectedExtensions[kind]
	if !exists || !slices.Contains(allowedExtensions, extension) {
		return fmt.Errorf("unsupported test owner: source=%q kind=%q name=%q", source, kind, name)
	}
	data, err := os.ReadFile(filepath.Join(repositoryRoot, filepath.FromSlash(source)))
	if err != nil {
		return fmt.Errorf("read test owner %q: %w", source, err)
	}
	defined := false
	switch kind {
	case "go_test":
		defined = bytes.Contains(data, []byte("func "+name+"("))
	case "pytest":
		defined = bytes.Contains(data, []byte("def "+name+"("))
	case "test_title":
		for _, call := range []string{"it", "test"} {
			for _, quote := range []string{"\"", "'", "`"} {
				if bytes.Contains(data, []byte(call+"("+quote+name+quote)) {
					defined = true
				}
			}
		}
	}
	if !defined {
		return fmt.Errorf("%s does not define %s owner %s", source, kind, name)
	}
	return nil
}

func validateMatrixGates(label string, gates []matrixGate, required []string, rejectUnknown bool) error {
	seen := make(map[string]bool, len(gates))
	for _, gate := range gates {
		if gate.ID == "" || gate.Command == "" || seen[gate.ID] ||
			!strings.HasPrefix(gate.Command, "make ") ||
			(rejectUnknown && !slices.Contains(required, gate.ID)) {
			return fmt.Errorf("invalid or duplicate %s gate: %+v", label, gate)
		}
		seen[gate.ID] = true
	}
	for _, gate := range required {
		if !seen[gate] {
			return fmt.Errorf("required %s gate %q is absent", label, gate)
		}
	}
	return nil
}
