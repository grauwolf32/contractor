package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/grauwolf32/contractor/internal/agentskills"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "contractor-skill:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("expected validate or package")
	}
	switch args[0] {
	case "validate":
		flags := newFlagSet("validate", stderr)
		expectedName := flags.String("name", "", "expected skill name")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 1 {
			return fmt.Errorf("usage: contractor-skill validate <source-directory-or-zip>")
		}
		path := flags.Arg(0)
		info, err := os.Lstat(path)
		if err != nil {
			return fmt.Errorf("cannot inspect input")
		}
		var skill *agentskills.Package
		if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
			_, skill, err = agentskills.PackageDirectory(path)
		} else if info.Mode().IsRegular() {
			if info.Size() > agentskills.MaximumArchiveBytes {
				return &agentskills.ValidationError{Code: agentskills.CodeLimitExceeded}
			}
			payload, readErr := os.ReadFile(path)
			if readErr != nil {
				return fmt.Errorf("cannot read input")
			}
			skill, err = agentskills.Validate(payload, *expectedName)
		} else {
			return &agentskills.ValidationError{Code: agentskills.CodeMemberForbidden}
		}
		if err != nil {
			return err
		}
		return writeSummary(stdout, skill)
	case "package":
		flags := newFlagSet("package", stderr)
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if flags.NArg() != 2 {
			return fmt.Errorf("usage: contractor-skill package <source-directory> <output.zip>")
		}
		payload, skill, err := agentskills.PackageDirectory(flags.Arg(0))
		if err != nil {
			return err
		}
		if err := writeAtomic(flags.Arg(1), payload); err != nil {
			return fmt.Errorf("cannot write output")
		}
		return writeSummary(stdout, skill)
	default:
		return fmt.Errorf("unknown operation %q", args[0])
	}
}

func newFlagSet(name string, stderr io.Writer) *flag.FlagSet {
	result := flag.NewFlagSet("contractor-skill "+name, flag.ContinueOnError)
	result.SetOutput(stderr)
	return result
}

func writeSummary(output io.Writer, skill *agentskills.Package) error {
	return json.NewEncoder(output).Encode(struct {
		Name      string                 `json:"name"`
		Digest    string                 `json:"digest"`
		Manifest  agentskills.Manifest   `json:"manifest"`
		Resources []agentskills.Resource `json:"resources"`
	}{Name: skill.Manifest.Name, Digest: skill.Digest, Manifest: skill.Manifest, Resources: skill.Resources})
}

func writeAtomic(destination string, payload []byte) error {
	directory := filepath.Dir(destination)
	temporary, err := os.CreateTemp(directory, ".contractor-skill-*.zip")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	committed := false
	defer func() {
		_ = temporary.Close()
		if !committed {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err := temporary.Chmod(0o644); err != nil {
		return err
	}
	if _, err := temporary.Write(payload); err != nil {
		return err
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, destination); err != nil {
		return err
	}
	committed = true
	return nil
}
