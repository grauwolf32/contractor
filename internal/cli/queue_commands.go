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

func (c *CLI) runQueue(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "queue requires list, status, pause, or resume"}
	}
	switch args[0] {
	case "list":
		return c.listQueue(ctx, client, printer, args[1:])
	case "status":
		if len(args) != 1 {
			return &UsageError{Message: "usage: contractor queue status"}
		}
		control, err := ownerQueueControl(ctx, client)
		if err != nil {
			return err
		}
		return printQueueControl(printer, control)
	case "pause", "resume":
		if len(args) != 1 {
			return &UsageError{Message: "usage: contractor queue " + args[0]}
		}
		return c.setQueuePaused(ctx, client, printer, args[0] == "pause")
	default:
		return &UsageError{Message: fmt.Sprintf("unknown queue command %q", args[0])}
	}
}

func (c *CLI) listQueue(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var state, membership, cursor string
	var limit int
	flags, err := parseFlags("contractor queue list", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&state, "state", "", "filter by initializing, running, or cancelling")
		flags.StringVar(&membership, "membership", "", "filter by standalone, project, or evaluation")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.IntVar(&limit, "limit", 50, "maximum number of queue items")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor queue list [filters]"); err != nil {
		return err
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	var stateValue *publicapi.NonTerminalWorkflowRunState
	if state != "" {
		if !validRunState(state, true) {
			return &UsageError{Message: "invalid non-terminal queue state"}
		}
		value := publicapi.NonTerminalWorkflowRunState(state)
		stateValue = &value
	}
	var membershipValue *publicapi.RunQueueMembership
	if membership != "" {
		if membership != "standalone" && membership != "project" && membership != "evaluation" {
			return &UsageError{Message: "queue membership must be standalone, project, or evaluation"}
		}
		value := publicapi.RunQueueMembership(membership)
		membershipValue = &value
	}
	var cursorValue *publicapi.Cursor
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		cursorValue = &value
	}
	response, err := client.API.ListRunQueueWithResponse(ctx, &publicapi.ListRunQueueParams{
		Limit: &limitValue, Cursor: cursorValue, State: stateValue, Membership: membershipValue,
	})
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
		for _, item := range page.Items {
			names = append(names, item.RunId)
		}
		return printer.Names(names...)
	}
	rows := make([][]string, 0, len(page.Items))
	for _, item := range page.Items {
		project := "-"
		if item.Project != nil {
			project = item.Project.ProjectId
		}
		rows = append(rows, []string{item.RunId, stringValue(item.State), item.Workflow, project, item.UpdatedAt.Format(time.RFC3339)})
	}
	return printer.Table([]string{"RUN", "STATE", "WORKFLOW", "PROJECT", "UPDATED"}, rows)
}

func (c *CLI) setQueuePaused(ctx context.Context, client *publicclient.Client, printer *Printer, paused bool) error {
	current, err := ownerQueueControl(ctx, client)
	if err != nil {
		return err
	}
	if current.Paused == paused {
		return printQueueControl(printer, current)
	}
	etag, err := publicclient.QuoteETag(current.Revision)
	if err != nil {
		return err
	}
	response, err := client.API.PutOwnerQueueControlWithResponse(ctx,
		&publicapi.PutOwnerQueueControlParams{IfMatch: etag},
		publicapi.UpdateOwnerQueueControlRequest{Paused: paused},
	)
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	return printQueueControl(printer, response.JSON200)
}

func ownerQueueControl(ctx context.Context, client *publicclient.Client) (*publicapi.OwnerQueueControl, error) {
	response, err := client.API.GetOwnerQueueControlWithResponse(ctx)
	if err != nil {
		return nil, err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return nil, err
	}
	return response.JSON200, nil
}

func printQueueControl(printer *Printer, control *publicapi.OwnerQueueControl) error {
	if printer.Mode() == OutputTable {
		updated := "-"
		if control.UpdatedAt != nil {
			updated = control.UpdatedAt.Format(time.RFC3339)
		}
		return printer.Table([]string{"PAUSED", "REVISION", "UPDATED"}, [][]string{{fmt.Sprint(control.Paused), control.Revision, updated}})
	}
	return printer.Object(control, control.Revision)
}
