package app

import (
	"crypto/subtle"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/grauwolf32/contractor/internal/auth"
	"golang.org/x/term"
)

type passwordPrompt func(string) ([]byte, error)

func runAuthCLI(args []string) error {
	if len(args) == 0 || args[0] != "hash-password" {
		return errors.New("auth command requires the hash-password subcommand")
	}
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		return errors.New("auth hash-password requires an interactive terminal")
	}
	prompt := func(message string) ([]byte, error) {
		if _, err := fmt.Fprint(os.Stderr, message); err != nil {
			return nil, err
		}
		value, err := term.ReadPassword(int(os.Stdin.Fd()))
		_, _ = fmt.Fprintln(os.Stderr)
		return value, err
	}
	return runAuthHashPassword(args[1:], prompt, os.Stdout)
}

func runAuthHashPassword(args []string, prompt passwordPrompt, output io.Writer) error {
	if prompt == nil || output == nil {
		return errors.New("password prompt and output are required")
	}
	userID := "local-admin"
	username := "admin"
	flags := flag.NewFlagSet("contractor-server auth hash-password", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.StringVar(&userID, "user-id", userID, "immutable local principal ID")
	flags.StringVar(&username, "username", username, "case-sensitive local username")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("parse auth hash-password flags: %w", err)
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	first, err := prompt("Password: ")
	if err != nil {
		return errors.New("read password from terminal")
	}
	defer wipeCommandBytes(first)
	if err := auth.ValidatePassword(first); err != nil {
		return errors.New("password must contain 12 through 1024 UTF-8 bytes")
	}
	second, err := prompt("Repeat password: ")
	if err != nil {
		return errors.New("read repeated password from terminal")
	}
	defer wipeCommandBytes(second)
	if len(first) != len(second) || subtle.ConstantTimeCompare(first, second) != 1 {
		return errors.New("passwords do not match")
	}
	hash, err := auth.HashPassword(first)
	if err != nil {
		return errors.New("hash password")
	}
	document, err := auth.BootstrapYAML(userID, username, hash)
	if err != nil {
		return fmt.Errorf("create local-auth bootstrap: %w", err)
	}
	if _, err := output.Write(document); err != nil {
		return errors.New("write local-auth bootstrap")
	}
	return nil
}

func wipeCommandBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
