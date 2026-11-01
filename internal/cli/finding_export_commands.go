package cli

import (
	"context"
	"errors"
	"flag"
	"io"
	"net/http"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/findingexport"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runAudit(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) < 2 || args[0] != "findings" {
		return &UsageError{Message: "usage: contractor audit findings export|download"}
	}
	switch args[1] {
	case "export":
		return c.exportAuditFindings(ctx, client, printer, args[2:])
	case "download":
		return c.downloadAuditFindings(ctx, client, args[2:])
	default:
		return &UsageError{Message: "audit findings requires export or download"}
	}
}

func (c *CLI) exportAuditFindings(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var selection, key, destination string
	var include, force bool
	flags, err := parseFlags("contractor audit findings export", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&selection, "selection", "-", "JSON array of exact finding selections, or - for stdin")
		flags.StringVar(&key, "client-key", "", "stable publication key for safe retry")
		flags.StringVar(&destination, "to", "-", "destination SARIF file, or - for stdout")
		flags.BoolVar(&include, "include-duplicates", false, "include every selected duplicate finding")
		flags.BoolVar(&force, "force", false, "replace an existing destination file")
	})
	if err != nil || flags == nil {
		return err
	}
	positional, err := requirePositionals(flags, 1, 1, "usage: contractor audit findings export AUDIT --client-key KEY --selection FILE [--include-duplicates] [--to FILE]")
	if err != nil {
		return err
	}
	reader := c.stdin
	if selection != "-" {
		file, err := openRegularInput(selection, "finding selection")
		if err != nil {
			return err
		}
		defer file.Close()
		reader = file
	}
	data, err := io.ReadAll(io.LimitReader(reader, (1<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return &UsageError{Message: "finding selection exceeds 1 MiB"}
	}
	var findings []findingexport.Selection
	if strictjson.DecodeNonNull(data, &findings) != nil {
		return &UsageError{Message: "selection must be a strict JSON array of findingId/revision and optional presentationReceiptId"}
	}
	request := findingexport.Request{ClientKey: key, Format: "sarif", Profile: "generic", Findings: findings, IncludeDuplicates: include}
	normalized, _, err := request.Normalize(positional[0])
	if err != nil {
		return &UsageError{Message: "invalid Audit, client key or finding selection"}
	}
	members := make([]publicapi.FindingExportSelection, len(normalized.Findings))
	for i, f := range normalized.Findings {
		members[i] = publicapi.FindingExportSelection{FindingId: f.FindingID, Revision: int(f.Revision)}
		if f.PresentationReceiptID != "" {
			value := publicapi.FindingCollectionIdentity(f.PresentationReceiptID)
			members[i].PresentationReceiptId = &value
		}
	}
	response, err := client.API.PublishAuditFindingExportWithResponse(ctx, positional[0], &publicapi.PublishAuditFindingExportParams{}, publicapi.PublishFindingExportRequest{ClientKey: key, Format: publicapi.Sarif, Profile: publicapi.Generic, Findings: members, IncludeDuplicates: &include})
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusCreated, http.StatusOK); err != nil {
		return err
	}
	published := response.JSON201
	if published == nil {
		published = response.JSON200
	}
	if published == nil {
		return errors.New("export response has no publication")
	}
	payload, err := downloadFindingExport(ctx, client, positional[0], published.ExportId, "content")
	if err != nil {
		return err
	}
	if len(payload) != published.Artifact.SizeBytes || contentdigest.Bytes(payload) != published.Artifact.Digest {
		return errors.New("downloaded export does not match the published exact artifact")
	}
	if destination == "-" {
		_, err = c.stdout.Write(payload)
		return err
	}
	if err := writeDownloadedFile(destination, payload, force); err != nil {
		return err
	}
	return printer.Object(published, published.ExportId)
}

func (c *CLI) downloadAuditFindings(ctx context.Context, client *publicclient.Client, args []string) error {
	var destination, part string
	var force bool
	flags, err := parseFlags("contractor audit findings download", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&destination, "to", "-", "destination file, or - for stdout")
		flags.StringVar(&part, "part", "content", "content, snapshot or diagnostics")
		flags.BoolVar(&force, "force", false, "replace an existing destination file")
	})
	if err != nil || flags == nil {
		return err
	}
	ids, err := requirePositionals(flags, 2, 2, "usage: contractor audit findings download AUDIT EXPORT [--part content|snapshot|diagnostics] [--to FILE]")
	if err != nil {
		return err
	}
	if contracts.ValidateResourceID("audit", ids[0]) != nil {
		return &UsageError{Message: "invalid Audit ID"}
	}
	payload, err := downloadFindingExport(ctx, client, ids[0], ids[1], part)
	if err != nil {
		return err
	}
	if destination == "-" {
		_, err = c.stdout.Write(payload)
		return err
	}
	return writeDownloadedFile(destination, payload, force)
}

func downloadFindingExport(ctx context.Context, client *publicclient.Client, auditID, exportID, part string) ([]byte, error) {
	var response *http.Response
	var err error
	maximum := findingexport.MaxBytes
	switch part {
	case "content":
		response, err = client.API.DownloadAuditFindingExport(ctx, auditID, exportID)
	case "snapshot":
		response, err = client.API.DownloadAuditFindingExportSnapshot(ctx, auditID, exportID)
	case "diagnostics":
		response, err = client.API.DownloadAuditFindingExportDiagnostics(ctx, auditID, exportID)
		maximum = findingexport.MaxDiagnosticsBytes
	default:
		return nil, &UsageError{Message: "part must be content, snapshot or diagnostics"}
	}
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, int64(maximum)+1))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, publicclient.DecodeError(response.StatusCode, data)
	}
	if len(data) > maximum {
		return nil, errors.New("finding export exceeds its byte limit")
	}
	return data, nil
}
