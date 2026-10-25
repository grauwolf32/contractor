package cli

import (
	"bytes"
	"context"
	"flag"
	"io"
	"net/http"

	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

// projectSitemap calls the generated owner API for reads and typed mutations.
// Mutations keep the caller's explicit expected revision, including "absent";
// the CLI never refreshes a stale edit or source generation automatically.
func (c *CLI) projectSitemap(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) (resultErr error) {
	if len(args) == 0 {
		return &UsageError{Message: "usage: contractor project sitemap get|mutate <project-id> [flags]"}
	}
	var revision, key, path string
	flags, err := parseFlags("contractor project sitemap "+args[0], c.stderr, args[1:], func(flags *flag.FlagSet) {
		flags.StringVar(&revision, "revision", "", "exact retained revision for get")
		flags.StringVar(&key, "idempotency-key", "", "retry-safe mutation key")
		flags.StringVar(&path, "file", "", "typed mutation JSON file")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor project sitemap get|mutate <project-id> [flags]")
	if err != nil {
		return err
	}
	switch args[0] {
	case "get":
		if path != "" || key != "" {
			return &UsageError{Message: "sitemap get accepts only --revision"}
		}
		params := &publicapi.GetProjectSitemapParams{}
		if revision != "" {
			value := publicapi.Revision(revision)
			params.Revision = &value
		}
		response, err := client.API.GetProjectSitemapWithResponse(ctx, positionals[0], params)
		if err != nil {
			return err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		return printer.Object(response.JSON200, response.JSON200.Revision)
	case "mutate":
		if path == "" || revision != "" {
			return &UsageError{Message: "sitemap mutate requires --file; the file carries expectedRevision"}
		}
		input, err := openRegularInput(path, "sitemap mutation")
		if err != nil {
			return err
		}
		defer input.Close()
		data, err := io.ReadAll(io.LimitReader(input, projectsitemap.MaxBatchBytes+1))
		if err != nil {
			return err
		}
		if len(data) > projectsitemap.MaxBatchBytes {
			return &UsageError{Message: "sitemap mutation exceeds the batch byte bound"}
		}
		generated := key == ""
		key, err = idempotencyKey(key)
		if err != nil {
			return err
		}
		if generated {
			defer func() { c.printCreateRetryHint(key, resultErr) }()
		}
		response, err := client.API.MutateProjectSitemapWithBodyWithResponse(ctx, positionals[0], &publicapi.MutateProjectSitemapParams{IdempotencyKey: key}, "application/json", bytes.NewReader(data))
		if err != nil {
			return err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		return printer.Object(response.JSON200, string(response.JSON200.Artifact.Revision))
	default:
		return &UsageError{Message: "sitemap requires get or mutate"}
	}
}
