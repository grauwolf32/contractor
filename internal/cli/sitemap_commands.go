package cli

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/httpapi/httpx"
	"github.com/grauwolf32/contractor/internal/projectsitemap"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

// projectSitemap reads through the ordinary Project Artifact API and sends
// domain commands through the owner mutation API.
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
		view, err := readProjectSitemap(ctx, client, positionals[0], revision)
		if err != nil {
			return err
		}
		return printer.Object(view, view.Revision)
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

type sitemapReadResult struct {
	Revision string                  `json:"revision"`
	Document projectsitemap.Document `json:"document"`
}

func readProjectSitemap(ctx context.Context, client *publicclient.Client, projectID, revision string) (sitemapReadResult, error) {
	params := &publicapi.DownloadProjectArtifactParams{}
	if revision != "" {
		if err := (contracts.ArtifactRef{Namespace: artifactpolicy.ProjectSitemapNamespace, Name: "map", Revision: &revision}).ValidateExact(); err != nil {
			return sitemapReadResult{}, err
		}
		value := publicapi.Revision(revision)
		params.Revision = &value
	}
	response, err := client.API.DownloadProjectArtifact(ctx, projectID, artifactpolicy.ProjectSitemapNamespace, "map", params)
	if err != nil {
		return sitemapReadResult{}, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, projectsitemap.MaxDocumentBytes+1))
	if err != nil {
		return sitemapReadResult{}, err
	}
	if response.StatusCode != http.StatusOK {
		err := publicclient.DecodeError(response.StatusCode, data)
		var failure *publicclient.APIError
		if revision != "" || !errors.As(err, &failure) || failure.Status != http.StatusNotFound || failure.Code != "not_found" {
			return sitemapReadResult{}, err
		}
		// A missing binding and an unavailable Project share the owner-safe 404.
		// Prove that an active Project exists before representing an absent map.
		project, err := client.API.GetProjectWithResponse(ctx, projectID)
		if err != nil {
			return sitemapReadResult{}, err
		}
		if err := publicclient.CheckResponse(project, http.StatusOK); err != nil {
			return sitemapReadResult{}, err
		}
		if project.JSON200 == nil || project.JSON200.ProjectId != projectID || !project.JSON200.Lifecycle.Valid() {
			return sitemapReadResult{}, errors.New("invalid Project response while resolving an absent sitemap")
		}
		if project.JSON200.Lifecycle == publicapi.ProjectLifecycleDeleting {
			return sitemapReadResult{}, &publicclient.APIError{Status: http.StatusConflict, Code: "project_deleting", Message: "Project is deleting"}
		}
		return sitemapReadResult{Document: projectsitemap.Empty()}, nil
	}
	if response.Header.Get("Content-Type") != projectsitemap.MediaType || len(data) > projectsitemap.MaxDocumentBytes {
		return sitemapReadResult{}, errors.New("invalid sitemap Artifact media type or size")
	}
	tags := response.Header.Values("ETag")
	if len(tags) != 1 {
		return sitemapReadResult{}, errors.New("sitemap Artifact response requires one exact revision ETag")
	}
	actual, err := httpx.ParseStrongETag(tags[0])
	if err != nil || (contracts.ArtifactRef{Namespace: artifactpolicy.ProjectSitemapNamespace, Name: "map", Revision: &actual}).ValidateExact() != nil || revision != "" && revision != actual {
		return sitemapReadResult{}, errors.New("sitemap Artifact ETag did not match the requested revision")
	}
	document, err := projectsitemap.Decode(data)
	if err != nil {
		return sitemapReadResult{}, fmt.Errorf("invalid sitemap Artifact: %w", err)
	}
	return sitemapReadResult{Revision: actual, Document: document}, nil
}
