package cli

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

type artifactScope struct {
	projectID string
	runID     string
}

func (s artifactScope) validate(writable bool) error {
	if s.projectID != "" && s.runID != "" {
		return &UsageError{Message: "--project and --run are mutually exclusive"}
	}
	if writable && s.runID != "" {
		return &UsageError{Message: "Run-scoped Artifacts are read-only"}
	}
	return nil
}

func (c *CLI) runArtifact(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "artifact requires list, put, get, metadata, versions, or lineage"}
	}
	switch args[0] {
	case "list":
		return c.listArtifacts(ctx, client, printer, args[1:])
	case "put", "upload":
		return c.putArtifactCommand(ctx, client, printer, args[1:])
	case "get", "download":
		return c.downloadArtifactCommand(ctx, client, args[1:])
	case "metadata", "show":
		return c.artifactMetadataCommand(ctx, client, printer, args[1:])
	case "versions":
		return c.artifactVersionsCommand(ctx, client, printer, args[1:])
	case "lineage":
		return c.artifactLineageCommand(ctx, client, printer, args[1:])
	default:
		return &UsageError{Message: fmt.Sprintf("unknown artifact command %q", args[0])}
	}
}

func addScopeFlags(flags *flag.FlagSet, scope *artifactScope) {
	flags.StringVar(&scope.projectID, "project", "", "use a Project-scoped Artifact")
	flags.StringVar(&scope.runID, "run", "", "use a Run-scoped Artifact")
}

func (c *CLI) listArtifacts(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var scope artifactScope
	var namespace, cursor string
	var limit int
	flags, err := parseFlags("contractor artifact list", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&namespace, "namespace", "", "filter by exact namespace")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.IntVar(&limit, "limit", 50, "maximum number of Artifacts")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor artifact list [--project ID | --run ID] [flags]"); err != nil {
		return err
	}
	if err := scope.validate(false); err != nil {
		return err
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	var namespaceValue *publicapi.ArtifactName
	if namespace != "" {
		value := publicapi.ArtifactName(namespace)
		namespaceValue = &value
	}
	var cursorValue *publicapi.Cursor
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		cursorValue = &value
	}
	var page *publicapi.ArtifactPage
	switch {
	case scope.projectID != "":
		response, callErr := client.API.ListProjectArtifactsWithResponse(ctx, scope.projectID, &publicapi.ListProjectArtifactsParams{Namespace: namespaceValue, Cursor: cursorValue, Limit: &limitValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	case scope.runID != "":
		response, callErr := client.API.ListRunArtifactsWithResponse(ctx, scope.runID, &publicapi.ListRunArtifactsParams{Namespace: namespaceValue, Cursor: cursorValue, Limit: &limitValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	default:
		response, callErr := client.API.ListArtifactsWithResponse(ctx, &publicapi.ListArtifactsParams{Namespace: namespaceValue, Cursor: cursorValue, Limit: &limitValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	}
	return printArtifactPage(printer, page)
}

func (c *CLI) putArtifactCommand(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var scope artifactScope
	var input, mediaType, revision string
	var create bool
	flags, err := parseFlags("contractor artifact put", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&input, "file", "-", "input file, or - for stdin")
		flags.StringVar(&mediaType, "type", "application/octet-stream", "Artifact media type")
		flags.BoolVar(&create, "create", false, "create a new binding with If-None-Match: *")
		flags.StringVar(&revision, "if-match", "", "expected current revision")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor artifact put <namespace/name> (--create | --if-match REV) [flags]")
	if err != nil {
		return err
	}
	if err := scope.validate(true); err != nil {
		return err
	}
	if create == (revision != "") {
		return &UsageError{Message: "artifact put requires exactly one of --create or --if-match"}
	}
	namespace, name, embeddedRevision, err := parseArtifactRef(positionals[0])
	if err != nil {
		return err
	}
	if embeddedRevision != "" {
		return &UsageError{Message: "artifact put target must not contain @revision"}
	}
	if _, _, err := mime.ParseMediaType(mediaType); err != nil {
		return &UsageError{Message: "invalid media type: " + err.Error()}
	}
	payload, err := c.readArtifactInput(input)
	if err != nil {
		return err
	}
	written, _, err := putArtifact(ctx, client, scope, namespace, name, mediaType, payload, create, revision)
	if err != nil {
		return err
	}
	return printer.Object(written, exactArtifactName(written.Artifact))
}

func (c *CLI) downloadArtifactCommand(ctx context.Context, client *publicclient.Client, args []string) error {
	var scope artifactScope
	var revision, destination string
	var force bool
	flags, err := parseFlags("contractor artifact get", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&revision, "revision", "", "exact revision")
		flags.StringVar(&destination, "to", "-", "destination file, or - for stdout")
		flags.BoolVar(&force, "force", false, "replace an existing destination file")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor artifact get <namespace/name[@revision]> [--project ID | --run ID] [--to FILE]")
	if err != nil {
		return err
	}
	if err := scope.validate(false); err != nil {
		return err
	}
	namespace, name, embeddedRevision, err := parseArtifactRef(positionals[0])
	if err != nil {
		return err
	}
	if revision != "" && embeddedRevision != "" && revision != embeddedRevision {
		return &UsageError{Message: "conflicting revisions in Artifact reference and --revision"}
	}
	if revision == "" {
		revision = embeddedRevision
	}
	payload, err := downloadArtifact(ctx, client, scope, namespace, name, revision)
	if err != nil {
		return err
	}
	if destination == "-" {
		_, err = c.stdout.Write(payload)
		return err
	}
	return writeDownloadedFile(destination, payload, force)
}

func (c *CLI) artifactMetadataCommand(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var scope artifactScope
	var revision string
	flags, err := parseFlags("contractor artifact metadata", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&revision, "revision", "", "exact revision")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor artifact metadata <namespace/name[@revision]> [scope flags]")
	if err != nil {
		return err
	}
	if err := scope.validate(false); err != nil {
		return err
	}
	namespace, name, embedded, err := parseArtifactRef(positionals[0])
	if err != nil {
		return err
	}
	if revision != "" && embedded != "" && revision != embedded {
		return &UsageError{Message: "conflicting revisions in Artifact reference and --revision"}
	}
	if revision == "" {
		revision = embedded
	}
	metadata, err := artifactMetadata(ctx, client, scope, namespace, name, revision)
	if err != nil {
		return err
	}
	return printer.Object(metadata, exactArtifactName(metadata.Artifact))
}

func (c *CLI) artifactVersionsCommand(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var scope artifactScope
	var cursor string
	var limit int
	flags, err := parseFlags("contractor artifact versions", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.IntVar(&limit, "limit", 50, "maximum number of revisions")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor artifact versions <namespace/name> [scope flags]")
	if err != nil {
		return err
	}
	if err := scope.validate(false); err != nil {
		return err
	}
	namespace, name, revision, err := parseArtifactRef(positionals[0])
	if err != nil {
		return err
	}
	if revision != "" {
		return &UsageError{Message: "artifact versions target must not contain @revision"}
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	var cursorValue *publicapi.Cursor
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		cursorValue = &value
	}
	var page *publicapi.ArtifactPage
	switch {
	case scope.projectID != "":
		response, callErr := client.API.ListProjectArtifactVersionsWithResponse(ctx, scope.projectID, namespace, name, &publicapi.ListProjectArtifactVersionsParams{Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	case scope.runID != "":
		response, callErr := client.API.ListRunArtifactVersionsWithResponse(ctx, scope.runID, namespace, name, &publicapi.ListRunArtifactVersionsParams{Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	default:
		response, callErr := client.API.ListArtifactVersionsWithResponse(ctx, namespace, name, &publicapi.ListArtifactVersionsParams{Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	}
	return printArtifactPage(printer, page)
}

func (c *CLI) artifactLineageCommand(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var scope artifactScope
	var revision, cursor string
	var limit int
	flags, err := parseFlags("contractor artifact lineage", c.stderr, args, func(flags *flag.FlagSet) {
		addScopeFlags(flags, &scope)
		flags.StringVar(&revision, "revision", "", "exact revision")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.IntVar(&limit, "limit", 50, "maximum number of edges")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor artifact lineage <namespace/name[@revision]> [scope flags]")
	if err != nil {
		return err
	}
	if err := scope.validate(false); err != nil {
		return err
	}
	namespace, name, embedded, err := parseArtifactRef(positionals[0])
	if err != nil {
		return err
	}
	if revision != "" && embedded != "" && revision != embedded {
		return &UsageError{Message: "conflicting revisions in Artifact reference and --revision"}
	}
	if revision == "" {
		revision = embedded
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	var revisionValue *publicapi.Revision
	if revision != "" {
		value := publicapi.Revision(revision)
		revisionValue = &value
	}
	var cursorValue *publicapi.Cursor
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		cursorValue = &value
	}
	var page *publicapi.ArtifactLineagePage
	switch {
	case scope.projectID != "":
		response, callErr := client.API.GetProjectArtifactLineageWithResponse(ctx, scope.projectID, namespace, name, &publicapi.GetProjectArtifactLineageParams{Revision: revisionValue, Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	case scope.runID != "":
		response, callErr := client.API.GetRunArtifactLineageWithResponse(ctx, scope.runID, namespace, name, &publicapi.GetRunArtifactLineageParams{Revision: revisionValue, Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	default:
		response, callErr := client.API.GetArtifactLineageWithResponse(ctx, namespace, name, &publicapi.GetArtifactLineageParams{Revision: revisionValue, Limit: &limitValue, Cursor: cursorValue})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	}
	return printer.Object(page)
}

func putArtifact(ctx context.Context, client *publicclient.Client, scope artifactScope, namespace, name, mediaType string, payload []byte, create bool, revision string) (*publicapi.ArtifactWriteResponse, int, error) {
	if err := scope.validate(true); err != nil {
		return nil, 0, err
	}
	if len(payload) > artifacts.MaxPayloadSize {
		return nil, 0, fmt.Errorf("Artifact exceeds %d-byte upload limit", artifacts.MaxPayloadSize)
	}
	if create == (revision != "") {
		return nil, 0, errors.New("internal Artifact write requires exactly one precondition")
	}
	var ifMatch *publicapi.IfMatch
	if revision != "" {
		quoted, err := publicclient.QuoteETag(revision)
		if err != nil {
			return nil, 0, err
		}
		value := publicapi.IfMatch(quoted)
		ifMatch = &value
	}
	if scope.projectID != "" {
		params := &publicapi.PutProjectArtifactParams{IfMatch: ifMatch}
		if create {
			value := publicapi.PutProjectArtifactParamsIfNoneMatch("*")
			params.IfNoneMatch = &value
		}
		response, err := client.API.PutProjectArtifactWithBodyWithResponse(ctx, scope.projectID, namespace, name, params, mediaType, bytes.NewReader(payload))
		if err != nil {
			return nil, 0, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK, http.StatusCreated); err != nil {
			return nil, response.StatusCode(), err
		}
		if response.JSON201 != nil {
			return response.JSON201, response.StatusCode(), nil
		}
		return response.JSON200, response.StatusCode(), nil
	}
	params := &publicapi.PutArtifactParams{IfMatch: ifMatch}
	if create {
		value := publicapi.PutArtifactParamsIfNoneMatch("*")
		params.IfNoneMatch = &value
	}
	response, err := client.API.PutArtifactWithBodyWithResponse(ctx, namespace, name, params, mediaType, bytes.NewReader(payload))
	if err != nil {
		return nil, 0, err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK, http.StatusCreated); err != nil {
		return nil, response.StatusCode(), err
	}
	if response.JSON201 != nil {
		return response.JSON201, response.StatusCode(), nil
	}
	return response.JSON200, response.StatusCode(), nil
}

func artifactMetadata(ctx context.Context, client *publicclient.Client, scope artifactScope, namespace, name, revision string) (*publicapi.ArtifactMetadata, error) {
	var revisionValue *publicapi.Revision
	if revision != "" {
		value := publicapi.Revision(revision)
		revisionValue = &value
	}
	switch {
	case scope.projectID != "":
		response, err := client.API.GetProjectArtifactMetadataWithResponse(ctx, scope.projectID, namespace, name, &publicapi.GetProjectArtifactMetadataParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.JSON200, nil
	case scope.runID != "":
		response, err := client.API.GetRunArtifactMetadataWithResponse(ctx, scope.runID, namespace, name, &publicapi.GetRunArtifactMetadataParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.JSON200, nil
	default:
		response, err := client.API.GetArtifactMetadataWithResponse(ctx, namespace, name, &publicapi.GetArtifactMetadataParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.JSON200, nil
	}
}

func downloadArtifact(ctx context.Context, client *publicclient.Client, scope artifactScope, namespace, name, revision string) ([]byte, error) {
	var revisionValue *publicapi.Revision
	if revision != "" {
		value := publicapi.Revision(revision)
		revisionValue = &value
	}
	switch {
	case scope.projectID != "":
		response, err := client.API.DownloadProjectArtifactWithResponse(ctx, scope.projectID, namespace, name, &publicapi.DownloadProjectArtifactParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.Body, nil
	case scope.runID != "":
		response, err := client.API.DownloadRunArtifactWithResponse(ctx, scope.runID, namespace, name, &publicapi.DownloadRunArtifactParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.Body, nil
	default:
		response, err := client.API.DownloadArtifactWithResponse(ctx, namespace, name, &publicapi.DownloadArtifactParams{Revision: revisionValue})
		if err != nil {
			return nil, err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return nil, err
		}
		return response.Body, nil
	}
}

func (c *CLI) readArtifactInput(path string) ([]byte, error) {
	reader := c.stdin
	var file *os.File
	if path != "-" {
		info, err := os.Lstat(filepath.Clean(path))
		if err != nil {
			return nil, fmt.Errorf("inspect Artifact input: %w", err)
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("Artifact input must be a regular file")
		}
		file, err = os.Open(filepath.Clean(path))
		if err != nil {
			return nil, fmt.Errorf("open Artifact input: %w", err)
		}
		defer file.Close()
		opened, err := file.Stat()
		if err != nil || !os.SameFile(info, opened) {
			return nil, errors.New("Artifact input changed while opening it")
		}
		reader = file
	}
	payload, err := io.ReadAll(io.LimitReader(reader, artifacts.MaxPayloadSize+1))
	if err != nil {
		return nil, fmt.Errorf("read Artifact input: %w", err)
	}
	if len(payload) > artifacts.MaxPayloadSize {
		return nil, fmt.Errorf("Artifact exceeds %d-byte upload limit", artifacts.MaxPayloadSize)
	}
	return payload, nil
}

func writeDownloadedFile(path string, payload []byte, force bool) error {
	path = filepath.Clean(path)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if !force {
		output, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("%s already exists (use --force to replace it)", path)
		}
		if err != nil {
			return err
		}
		committed := false
		defer func() {
			_ = output.Close()
			if !committed {
				_ = os.Remove(path)
			}
		}()
		if _, err := output.Write(payload); err != nil {
			return err
		}
		if err := output.Sync(); err != nil {
			return err
		}
		if err := output.Close(); err != nil {
			return err
		}
		committed = true
		return nil
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".contractor-download-*")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return err
	}
	if _, err := temporary.Write(payload); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryName, path); err != nil {
		return err
	}
	return nil
}

func parseArtifactRef(value string) (string, string, string, error) {
	binding, revision, _ := strings.Cut(value, "@")
	if strings.Count(binding, "/") != 1 || strings.Contains(revision, "@") {
		return "", "", "", &UsageError{Message: "Artifact reference must be namespace/name or namespace/name@revision"}
	}
	namespace, name, _ := strings.Cut(binding, "/")
	if namespace == "" || name == "" || (strings.Contains(value, "@") && revision == "") {
		return "", "", "", &UsageError{Message: "Artifact reference must be namespace/name or namespace/name@revision"}
	}
	return namespace, name, revision, nil
}

func exactArtifactName(reference publicapi.ExactArtifactRef) string {
	return reference.Namespace + "/" + reference.Name + "@" + reference.Revision
}

func printArtifactPage(printer *Printer, page *publicapi.ArtifactPage) error {
	if printer.Mode() == OutputJSON {
		return printer.JSON(page)
	}
	if printer.Mode() == OutputName {
		names := make([]string, 0, len(page.Items))
		for _, item := range page.Items {
			names = append(names, exactArtifactName(item.Artifact))
		}
		return printer.Names(names...)
	}
	rows := make([][]string, 0, len(page.Items))
	for _, item := range page.Items {
		rows = append(rows, []string{
			exactArtifactName(item.Artifact), stringValue(item.MediaType), strconv.Itoa(item.Size),
			strconv.FormatBool(item.Current), strconv.FormatBool(item.Frozen), item.CreatedAt.Format(time.RFC3339),
		})
	}
	return printer.Table([]string{"ARTIFACT", "TYPE", "BYTES", "CURRENT", "FROZEN", "CREATED"}, rows)
}
