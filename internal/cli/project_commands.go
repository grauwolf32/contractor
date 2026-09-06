package cli

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"time"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runProject(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "project requires list, create, get, update, or delete"}
	}
	switch args[0] {
	case "list":
		return c.listProjects(ctx, client, printer, args[1:])
	case "create":
		return c.createProject(ctx, client, printer, args[1:])
	case "get", "show":
		return c.getProject(ctx, client, printer, args[1:])
	case "update":
		return c.updateProject(ctx, client, printer, args[1:])
	case "delete", "remove":
		return c.deleteProject(ctx, client, printer, args[1:])
	default:
		return &UsageError{Message: fmt.Sprintf("unknown project command %q", args[0])}
	}
}

func (c *CLI) listProjects(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var limit int
	var cursor, kind string
	flags, err := parseFlags("contractor project list", c.stderr, args, func(flags *flag.FlagSet) {
		flags.IntVar(&limit, "limit", 50, "maximum number of Projects")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.StringVar(&kind, "kind", "", "filter by project or evaluation")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor project list [flags]"); err != nil {
		return err
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	params := &publicapi.ListProjectsParams{Limit: &limitValue}
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		params.Cursor = &value
	}
	if kind != "" {
		if kind != string(publicapi.ProjectKindProject) && kind != string(publicapi.ProjectKindEvaluation) {
			return &UsageError{Message: "project kind must be project or evaluation"}
		}
		value := publicapi.ProjectKind(kind)
		params.Kind = &value
	}
	response, err := client.API.ListProjectsWithResponse(ctx, params)
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	page := response.JSON200
	if printer.Mode() == OutputJSON {
		return printer.JSON(page)
	}
	if printer.Mode() == OutputName {
		names := make([]string, 0, len(page.Items))
		for _, project := range page.Items {
			names = append(names, project.ProjectId)
		}
		return printer.Names(names...)
	}
	rows := make([][]string, 0, len(page.Items))
	for _, project := range page.Items {
		rows = append(rows, projectRow(project))
	}
	return printer.Table([]string{"ID", "NAME", "KIND", "LIFECYCLE", "UPDATED"}, rows)
}

func (c *CLI) createProject(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var kind, description, key string
	flags, err := parseFlags("contractor project create", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&kind, "kind", string(publicapi.ProjectKindProject), "project kind: project or evaluation")
		flags.StringVar(&description, "description", "", "project description")
		flags.StringVar(&key, "idempotency-key", "", "retry-safe request key (generated when omitted)")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor project create <name> [flags]")
	if err != nil {
		return err
	}
	if kind != string(publicapi.ProjectKindProject) && kind != string(publicapi.ProjectKindEvaluation) {
		return &UsageError{Message: "project kind must be project or evaluation"}
	}
	key, err = idempotencyKey(key)
	if err != nil {
		return err
	}
	body := publicapi.CreateProjectRequest{Name: positionals[0], Kind: publicapi.ProjectKind(kind)}
	if description != "" {
		body.Description = &description
	}
	response, err := client.API.CreateProjectWithResponse(ctx, &publicapi.CreateProjectParams{IdempotencyKey: key}, body)
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusCreated); err != nil {
		return err
	}
	return printer.Object(response.JSON201, response.JSON201.ProjectId)
}

func (c *CLI) getProject(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	flags, err := parseFlags("contractor project get", c.stderr, args, func(*flag.FlagSet) {})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor project get <project-id>")
	if err != nil {
		return err
	}
	response, err := client.API.GetProjectWithResponse(ctx, positionals[0])
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	if printer.Mode() == OutputTable {
		return printer.Table([]string{"ID", "NAME", "KIND", "LIFECYCLE", "UPDATED"}, [][]string{projectRow(*response.JSON200)})
	}
	return printer.Object(response.JSON200, response.JSON200.ProjectId)
}

func (c *CLI) updateProject(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var name, description, revision string
	flags, err := parseFlags("contractor project update", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&name, "name", "", "new project name")
		flags.StringVar(&description, "description", "", "new project description; empty clears it")
		flags.StringVar(&revision, "if-match", "", "expected revision (fetched when omitted)")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor project update <project-id> (--name NAME | --description TEXT) [--if-match REV]")
	if err != nil {
		return err
	}
	var nameSet, descriptionSet bool
	flags.Visit(func(current *flag.Flag) {
		nameSet = nameSet || current.Name == "name"
		descriptionSet = descriptionSet || current.Name == "description"
	})
	if !nameSet && !descriptionSet {
		return &UsageError{Message: "project update requires --name or --description"}
	}
	if revision == "" {
		revision, err = currentProjectRevision(ctx, client, positionals[0])
		if err != nil {
			return err
		}
	}
	etag, err := publicclient.QuoteETag(revision)
	if err != nil {
		return err
	}
	body := publicapi.UpdateProjectRequest{}
	if nameSet {
		body.Name = &name
	}
	if descriptionSet {
		body.Description = &description
	}
	response, err := client.API.UpdateProjectWithResponse(ctx, positionals[0], &publicapi.UpdateProjectParams{IfMatch: etag}, body)
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	return printer.Object(response.JSON200, response.JSON200.ProjectId)
}

func (c *CLI) deleteProject(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var revision string
	flags, err := parseFlags("contractor project delete", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&revision, "if-match", "", "expected revision (fetched when omitted)")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor project delete <project-id> [--if-match REV]")
	if err != nil {
		return err
	}
	if revision == "" {
		revision, err = currentProjectRevision(ctx, client, positionals[0])
		if err != nil {
			return err
		}
	}
	etag, err := publicclient.QuoteETag(revision)
	if err != nil {
		return err
	}
	response, err := client.API.DeleteProjectWithResponse(ctx, positionals[0], &publicapi.DeleteProjectParams{IfMatch: etag})
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusAccepted); err != nil {
		return err
	}
	return printer.Object(response.JSON202, response.JSON202.ProjectId)
}

func currentProjectRevision(ctx context.Context, client *publicclient.Client, projectID string) (string, error) {
	response, err := client.API.GetProjectWithResponse(ctx, projectID)
	if err != nil {
		return "", err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return "", err
	}
	return response.JSON200.Revision, nil
}

func validatedLimit(value int) (publicapi.Limit, error) {
	if value < 1 || value > 200 {
		return 0, &UsageError{Message: "limit must be between 1 and 200"}
	}
	return publicapi.Limit(value), nil
}

func projectRow(project publicapi.Project) []string {
	return []string{
		project.ProjectId,
		project.Name,
		string(project.Kind),
		string(project.Lifecycle),
		project.UpdatedAt.Format(time.RFC3339),
	}
}
