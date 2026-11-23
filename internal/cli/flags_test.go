package cli

import (
	"flag"
	"io"
	"slices"
	"testing"
)

func TestCommandFlagsPreservePositionals(t *testing.T) {
	for _, test := range []struct {
		name        string
		args        []string
		positionals []string
		value       string
		force       bool
	}{
		{"protected directory", []string{"--", "-source"}, []string{"-source"}, "", false},
		{"protected flag", []string{"--", "--force"}, []string{"--force"}, "", false},
		{"protected separator", []string{"--", "--"}, []string{"--"}, "", false},
		{"option before separator", []string{"--name", "repo", "--", "-source"}, []string{"-source"}, "repo", false},
		{"option before resource", []string{"--name=repo", "source"}, []string{"source"}, "repo", false},
		{"option after resource", []string{"source", "--name", "repo"}, []string{"source"}, "repo", false},
		{"mixed options", []string{"one", "--force", "two", "--name", "repo"}, []string{"one", "two"}, "repo", true},
		{"stdin", []string{"-", "--name", "repo"}, []string{"-"}, "repo", false},
		{"hyphen flag value", []string{"source", "--name", "-repo"}, []string{"source"}, "-repo", false},
		{"separator is flag value", []string{"--name", "--", "source"}, []string{"source"}, "--", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var value string
			var force bool
			flags, err := parseFlags("test", io.Discard, test.args, func(f *flag.FlagSet) {
				f.StringVar(&value, "name", "", "name")
				f.BoolVar(&force, "force", false, "force")
			})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(flags.Args(), test.positionals) || value != test.value || force != test.force {
				t.Fatalf("positionals=%q name=%q force=%t; want %q %q %t", flags.Args(), value, force, test.positionals, test.value, test.force)
			}
		})
	}
}

func TestCommandFlagsStillRejectMalformedOptions(t *testing.T) {
	for _, args := range [][]string{{"--unknown", "resource"}, {"resource", "--name"}, {"resource", "--force=invalid"}} {
		_, err := parseFlags("test", io.Discard, args, func(f *flag.FlagSet) {
			f.String("name", "", "name")
			f.Bool("force", false, "force")
		})
		if err == nil {
			t.Errorf("invalid arguments accepted: %q", args)
		}
	}
}
