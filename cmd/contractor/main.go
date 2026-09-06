package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/grauwolf32/contractor/internal/cli"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	err := cli.New(os.Stdin, os.Stdout, os.Stderr, os.Getenv).Run(ctx, os.Args[1:])
	if err == nil {
		return
	}
	_, _ = fmt.Fprintln(os.Stderr, "contractor:", err)
	var usage *cli.UsageError
	if errors.As(err, &usage) {
		os.Exit(2)
	}
	os.Exit(1)
}
