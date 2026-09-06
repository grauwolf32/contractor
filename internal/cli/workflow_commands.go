package cli

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runWorkflow(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "workflow requires list or get"}
	}
	switch args[0] {
	case "list":
		var limit int
		var cursor string
		flags, err := parseFlags("contractor workflow list", c.stderr, args[1:], func(flags *flag.FlagSet) {
			flags.IntVar(&limit, "limit", 50, "maximum number of Workflows")
			flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		})
		if err != nil || flags == nil {
			return err
		}
		if _, err := requirePositionals(flags, 0, 0, "usage: contractor workflow list [--limit N] [--cursor CURSOR]"); err != nil {
			return err
		}
		params, err := listWorkflowsParams(limit, cursor)
		if err != nil {
			return err
		}
		response, err := client.API.ListWorkflowsWithResponse(ctx, params)
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
			for _, workflow := range page.Items {
				names = append(names, workflow.Ref.Name+"@"+workflow.Ref.Version)
			}
			return printer.Names(names...)
		}
		rows := make([][]string, 0, len(page.Items))
		for _, workflow := range page.Items {
			rows = append(rows, []string{
				workflow.Ref.Name + "@" + workflow.Ref.Version,
				workflow.EntryStage,
				strconv.Itoa(len(workflow.Inputs)),
				strconv.Itoa(len(workflow.Parameters)),
				strconv.Itoa(len(workflow.Outputs)),
			})
		}
		return printer.Table([]string{"WORKFLOW", "ENTRY STAGE", "INPUTS", "PARAMETERS", "OUTPUTS"}, rows)
	case "get", "show":
		flags, err := parseFlags("contractor workflow get", c.stderr, args[1:], func(*flag.FlagSet) {})
		if err != nil || flags == nil {
			return err
		}
		positionals, err := requirePositionals(flags, 1, 1, "usage: contractor workflow get <name@version>")
		if err != nil {
			return err
		}
		name, version, err := exactSelector(positionals[0])
		if err != nil {
			return err
		}
		response, err := client.API.GetWorkflowWithResponse(ctx, name, version)
		if err != nil {
			return err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		return printer.Object(response.JSON200, name+"@"+version)
	default:
		return &UsageError{Message: fmt.Sprintf("unknown workflow command %q", args[0])}
	}
}

func listWorkflowsParams(limit int, cursor string) (*publicapi.ListWorkflowsParams, error) {
	if limit < 1 || limit > 200 {
		return nil, &UsageError{Message: "limit must be between 1 and 200"}
	}
	value := publicapi.Limit(limit)
	params := &publicapi.ListWorkflowsParams{Limit: &value}
	if cursor != "" {
		current := publicapi.Cursor(cursor)
		params.Cursor = &current
	}
	return params, nil
}
