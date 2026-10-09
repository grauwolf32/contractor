package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/grauwolf32/contractor/internal/cli"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "contractor-pki:", err)
		os.Exit(1)
	}
}

// Keep the standalone entry point on the same command table, defaults and
// validation as contractor pki. Both issue-agent and issue-runtime are aliases.
func run(args []string, stdout, stderr io.Writer) error {
	return cli.New(nil, stdout, stderr, nil).Run(context.Background(), append([]string{"pki"}, args...))
}
