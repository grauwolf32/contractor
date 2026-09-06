package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/sourcebundle"
)

type sourcePushResult struct {
	Artifact      publicapi.ExactArtifactRef `json:"artifact"`
	MediaType     publicapi.MediaType        `json:"mediaType"`
	Size          int                        `json:"size"`
	Files         int                        `json:"files"`
	ExpandedBytes int64                      `json:"expandedBytes"`
	SHA256        string                     `json:"sha256"`
}

func (c *CLI) runSource(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 || args[0] != "push" {
		return &UsageError{Message: "usage: contractor source push <directory> [--name NAME] [--project ID]"}
	}
	var projectID, namespace, name string
	var includeIgnored bool
	flags, err := parseFlags("contractor source push", c.stderr, args[1:], func(flags *flag.FlagSet) {
		flags.StringVar(&projectID, "project", "", "upload to Project scope")
		flags.StringVar(&namespace, "namespace", "projects", "Artifact namespace")
		flags.StringVar(&name, "name", "", "Artifact name (defaults to the directory name)")
		flags.BoolVar(&includeIgnored, "include-ignored", false, "include files ignored by Git")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor source push <directory> [--name NAME] [--project ID]")
	if err != nil {
		return err
	}
	if name == "" {
		absolute, resolveErr := filepath.Abs(positionals[0])
		if resolveErr != nil {
			return resolveErr
		}
		name = sourceArtifactName(filepath.Base(absolute))
	}
	if err := contracts.ValidateArtifactName(namespace); err != nil {
		return &UsageError{Message: "invalid source namespace: " + err.Error()}
	}
	if err := contracts.ValidateArtifactName(name); err != nil {
		return &UsageError{Message: "invalid source Artifact name: " + err.Error()}
	}
	bundle, err := sourcebundle.Build(positionals[0], sourcebundle.Options{IncludeIgnored: includeIgnored})
	if err != nil {
		return err
	}
	_, _ = fmt.Fprintf(c.stderr, "packed %d files, %d expanded bytes, %d ZIP bytes\n", bundle.Files, bundle.ExpandedBytes, len(bundle.Data))

	scope := artifactScope{projectID: projectID}
	create := false
	revision := ""
	metadata, metadataErr := artifactMetadata(ctx, client, scope, namespace, name, "")
	if metadataErr == nil {
		revision = metadata.Artifact.Revision
	} else {
		var apiError *publicclient.APIError
		if !errors.As(metadataErr, &apiError) || apiError.Status != http.StatusNotFound {
			return metadataErr
		}
		create = true
	}
	written, _, err := putArtifact(ctx, client, scope, namespace, name, "application/zip", bundle.Data, create, revision)
	if err != nil {
		return err
	}
	result := sourcePushResult{
		Artifact: written.Artifact, MediaType: written.MediaType, Size: written.Size,
		Files: bundle.Files, ExpandedBytes: bundle.ExpandedBytes, SHA256: bundle.SHA256,
	}
	if printer.Mode() == OutputTable {
		return printer.Table(
			[]string{"ARTIFACT", "FILES", "EXPANDED", "ZIP", "SHA256"},
			[][]string{{exactArtifactName(result.Artifact), fmt.Sprint(result.Files), fmt.Sprint(result.ExpandedBytes), fmt.Sprint(result.Size), result.SHA256}},
		)
	}
	return printer.Object(result, exactArtifactName(result.Artifact))
}

var sourceNameSeparator = regexp.MustCompile(`[^A-Za-z0-9_.-]+`)

func sourceArtifactName(value string) string {
	value = sourceNameSeparator.ReplaceAllString(strings.TrimSpace(value), "-")
	value = strings.Trim(value, "-._")
	if value == "" {
		value = "source"
	}
	if len(value) > 128 {
		value = value[:128]
		value = strings.TrimRight(value, "-._")
	}
	return value
}
