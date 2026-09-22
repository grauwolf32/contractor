package app

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"
)

func TestServerCommandsPrintUsageForHelp(t *testing.T) {
	var usage bytes.Buffer
	previous := commandUsageOutput
	commandUsageOutput = &usage
	t.Cleanup(func() { commandUsageOutput = previous })
	getenv := func(string) string { return "" }
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	for _, test := range []struct {
		args []string
		flag string
	}{
		{[]string{"serve", "--help"}, "-database-url"},
		{[]string{"migrate", "--help"}, "-database-url"},
		{[]string{"config", "validate", "-h"}, "-root"},
		{[]string{"blobs", "cleanup", "--help"}, "-offline"},
	} {
		usage.Reset()
		if err := RunCLI(context.Background(), test.args, getenv, logger); err != nil {
			t.Fatalf("%v: RunCLI = %v, want success", test.args, err)
		}
		if !strings.Contains(usage.String(), test.flag) {
			t.Fatalf("%v: usage %q does not mention %s", test.args, usage.String(), test.flag)
		}
	}

	usage.Reset()
	prompt := func(string) ([]byte, error) { t.Fatal("prompted after --help"); return nil, nil }
	if err := runAuthHashPassword([]string{"--help"}, prompt, io.Discard); !errors.Is(err, errHelpShown) {
		t.Fatalf("auth hash-password --help = %v", err)
	}
	if !strings.Contains(usage.String(), "-username") {
		t.Fatalf("auth usage = %q", usage.String())
	}
}

func TestBlobCleanupPreservesFlagParseError(t *testing.T) {
	_, err := parseBlobCleanupConfig([]string{"cleanup", "--bogus"}, func(string) string { return "" })
	if err == nil || !strings.Contains(err.Error(), "bogus") {
		t.Fatalf("parse error = %v, want the unknown flag", err)
	}
}
