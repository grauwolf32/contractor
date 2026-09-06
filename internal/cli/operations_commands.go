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

func (c *CLI) runOperations(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "ops requires snapshot, agents, principals, or allocations"}
	}
	if args[0] == "snapshot" {
		if len(args) != 1 {
			return &UsageError{Message: "usage: contractor ops snapshot"}
		}
		response, err := client.API.GetOperationsSnapshotWithResponse(ctx)
		if err != nil {
			return err
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		return printer.Object(response.JSON200, fmt.Sprint(response.JSON200.Cursor))
	}
	var limit int
	var cursor string
	flags, err := parseFlags("contractor ops "+args[0], c.stderr, args[1:], func(flags *flag.FlagSet) {
		flags.IntVar(&limit, "limit", 50, "maximum number of results")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor ops "+args[0]+" [--limit N] [--cursor CURSOR]"); err != nil {
		return err
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
	switch args[0] {
	case "agents":
		response, err := client.API.ListRuntimeAgentsWithResponse(ctx, &publicapi.ListRuntimeAgentsParams{Limit: &limitValue, Cursor: cursorValue})
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
				names = append(names, item.InstanceId)
			}
			return printer.Names(names...)
		}
		rows := make([][]string, 0, len(page.Items))
		for _, item := range page.Items {
			last := "-"
			if item.LastAcceptedHeartbeat != nil {
				last = item.LastAcceptedHeartbeat.Format(time.RFC3339)
			}
			rows = append(rows, []string{item.InstanceId, stringValue(item.ObservedState), stringValue(item.SlotState), item.SoftwareVersion, last})
		}
		return printer.Table([]string{"INSTANCE", "STATE", "SLOT", "VERSION", "LAST HEARTBEAT"}, rows)
	case "principals":
		response, err := client.API.ListRuntimeAgentPrincipalsWithResponse(ctx, &publicapi.ListRuntimeAgentPrincipalsParams{Limit: &limitValue, Cursor: cursorValue})
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
				names = append(names, item.RuntimeAgentId)
			}
			return printer.Names(names...)
		}
		rows := make([][]string, 0, len(page.Items))
		for _, item := range page.Items {
			rows = append(rows, []string{item.RuntimeAgentId, stringValue(item.Availability), fmt.Sprint(len(item.Labels)), item.Revision, item.UpdatedAt.Format(time.RFC3339)})
		}
		return printer.Table([]string{"RUNTIME AGENT", "AVAILABILITY", "LABELS", "REVISION", "UPDATED"}, rows)
	case "allocations":
		response, err := client.API.ListAllocationsWithResponse(ctx, &publicapi.ListAllocationsParams{Limit: &limitValue, Cursor: cursorValue})
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
				names = append(names, item.AllocationId)
			}
			return printer.Names(names...)
		}
		rows := make([][]string, 0, len(page.Items))
		for _, item := range page.Items {
			rows = append(rows, []string{item.AllocationId, item.RunId, item.LogicalWorker, stringValue(item.AuthoritativePhase), stringValue(item.ObservedPhase), item.RuntimeAgentInstanceId})
		}
		return printer.Table([]string{"ALLOCATION", "RUN", "WORKER", "PHASE", "OBSERVED", "RUNTIME"}, rows)
	default:
		return &UsageError{Message: fmt.Sprintf("unknown ops command %q", args[0])}
	}
}
