package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

type stringList []string

func (values *stringList) String() string { return strings.Join(*values, ",") }
func (values *stringList) Set(value string) error {
	*values = append(*values, value)
	return nil
}

func (c *CLI) runRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "run requires list, create, get, watch, cancel, delete, or output"}
	}
	switch args[0] {
	case "list":
		return c.listRuns(ctx, client, printer, args[1:])
	case "create", "start":
		return c.createRun(ctx, client, printer, args[1:])
	case "get", "show":
		return c.getRun(ctx, client, printer, args[1:])
	case "watch", "wait":
		return c.watchRun(ctx, client, printer, args[1:])
	case "cancel":
		return c.cancelRun(ctx, client, printer, args[1:])
	case "delete", "remove":
		return c.deleteRun(ctx, client, printer, args[1:])
	case "output":
		return c.downloadRunOutput(ctx, client, args[1:])
	default:
		return &UsageError{Message: fmt.Sprintf("unknown run command %q", args[0])}
	}
}

func (c *CLI) listRuns(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var projectID, state, lifecycle, cursor string
	var labels stringList
	var limit int
	flags, err := parseFlags("contractor run list", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&projectID, "project", "", "list Runs in one Project")
		flags.StringVar(&state, "state", "", "filter by exact Run state")
		flags.StringVar(&lifecycle, "lifecycle", "", "filter by active or terminal")
		flags.Var(&labels, "label", "filter by label key=value; repeatable")
		flags.StringVar(&cursor, "cursor", "", "pagination cursor")
		flags.IntVar(&limit, "limit", 50, "maximum number of Runs")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor run list [--project ID] [filters]"); err != nil {
		return err
	}
	limitValue, err := validatedLimit(limit)
	if err != nil {
		return err
	}
	if state != "" && !validRunState(state, false) {
		return &UsageError{Message: "invalid Run state"}
	}
	if lifecycle != "" && lifecycle != "active" && lifecycle != "terminal" {
		return &UsageError{Message: "Run lifecycle must be active or terminal"}
	}
	for _, label := range labels {
		if _, _, err := splitAssignment(label); err != nil {
			return &UsageError{Message: "invalid --label: " + err.Error()}
		}
	}
	var cursorValue *publicapi.Cursor
	if cursor != "" {
		value := publicapi.Cursor(cursor)
		cursorValue = &value
	}
	var stateValue *publicapi.WorkflowRunState
	if state != "" {
		value := publicapi.WorkflowRunState(state)
		stateValue = &value
	}
	var lifecycleValue *publicapi.WorkflowRunLifecycle
	if lifecycle != "" {
		value := publicapi.WorkflowRunLifecycle(lifecycle)
		lifecycleValue = &value
	}
	labelValues := []string(labels)
	var labelPointer *[]string
	if len(labelValues) != 0 {
		labelPointer = &labelValues
	}
	var page *publicapi.RunPage
	if projectID != "" {
		response, callErr := client.API.ListProjectRunsWithResponse(ctx, projectID, &publicapi.ListProjectRunsParams{
			Limit: &limitValue, Cursor: cursorValue, State: stateValue, Lifecycle: lifecycleValue, Label: labelPointer,
		})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	} else {
		response, callErr := client.API.ListRunsWithResponse(ctx, &publicapi.ListRunsParams{
			Limit: &limitValue, Cursor: cursorValue, State: stateValue, Lifecycle: lifecycleValue, Label: labelPointer,
		})
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
			return err
		}
		page = response.JSON200
	}
	return printRunPage(printer, page)
}

func (c *CLI) createRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var projectID, requestFile, key string
	var parameters, artifactValues, labels, runtimeLabels stringList
	flags, err := parseFlags("contractor run create", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&projectID, "project", "", "create the Run in one Project")
		flags.StringVar(&requestFile, "request", "", "complete JSON request file, or - for stdin")
		flags.StringVar(&key, "idempotency-key", "", "retry-safe request key (generated when omitted)")
		flags.Var(&parameters, "param", "Workflow parameter key=value; repeatable")
		flags.Var(&artifactValues, "artifact", "input slot=namespace/name[@revision]; repeatable")
		flags.Var(&labels, "label", "metadata label key=value; repeatable")
		flags.Var(&runtimeLabels, "runtime-label", "Runtime infrastructure label; repeatable")
	})
	if err != nil || flags == nil {
		return err
	}
	minimum, maximum := 1, 1
	if requestFile != "" {
		minimum, maximum = 0, 0
	}
	positionals, err := requirePositionals(flags, minimum, maximum, "usage: contractor run create <workflow@version> [flags], or contractor run create --request FILE")
	if err != nil {
		return err
	}
	var body publicapi.CreateRunRequest
	if requestFile != "" {
		if len(parameters)+len(artifactValues)+len(labels)+len(runtimeLabels) != 0 {
			return &UsageError{Message: "--request cannot be combined with --param, --artifact, --label, or --runtime-label"}
		}
		if err := c.decodeRequestFile(requestFile, &body); err != nil {
			return err
		}
	} else {
		if _, _, err := exactSelector(positionals[0]); err != nil {
			return err
		}
		body.Workflow = positionals[0]
		parsedParameters, err := assignments(parameters)
		if err != nil {
			return &UsageError{Message: "invalid --param: " + err.Error()}
		}
		if len(parsedParameters) != 0 {
			body.Parameters = &parsedParameters
		}
		parsedLabels, err := assignments(labels)
		if err != nil {
			return &UsageError{Message: "invalid --label: " + err.Error()}
		}
		if len(parsedLabels) != 0 {
			value := publicapi.RunMetadataLabels(parsedLabels)
			body.Labels = &value
		}
		if len(runtimeLabels) != 0 {
			value := []publicapi.ConfigId(runtimeLabels)
			body.RuntimeLabels = &value
		}
		if len(artifactValues) != 0 {
			parsedArtifacts := make(map[string]publicapi.ArtifactRef, len(artifactValues))
			for _, assignment := range artifactValues {
				slot, reference, assignmentErr := splitAssignment(assignment)
				if assignmentErr != nil {
					return &UsageError{Message: "invalid --artifact: " + assignmentErr.Error()}
				}
				namespace, name, revision, referenceErr := parseArtifactRef(reference)
				if referenceErr != nil {
					return referenceErr
				}
				artifact := publicapi.ArtifactRef{Namespace: namespace, Name: name}
				if revision != "" {
					artifact.Revision = &revision
				}
				if _, duplicate := parsedArtifacts[slot]; duplicate {
					return &UsageError{Message: "duplicate Artifact input slot " + slot}
				}
				parsedArtifacts[slot] = artifact
			}
			body.Artifacts = &parsedArtifacts
		}
	}
	if strings.TrimSpace(body.Workflow) == "" {
		return &UsageError{Message: "Run request requires workflow"}
	}
	key, err = idempotencyKey(key)
	if err != nil {
		return err
	}
	var created *publicapi.CreateRunResponse
	if projectID != "" {
		response, callErr := client.API.CreateProjectRunWithResponse(ctx, projectID, &publicapi.CreateProjectRunParams{IdempotencyKey: key}, body)
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusAccepted); err != nil {
			return err
		}
		created = response.JSON202
	} else {
		response, callErr := client.API.CreateRunWithResponse(ctx, &publicapi.CreateRunParams{IdempotencyKey: key}, body)
		if callErr != nil {
			return callErr
		}
		if err := publicclient.CheckResponse(response, http.StatusAccepted); err != nil {
			return err
		}
		created = response.JSON202
	}
	return printer.Object(created, created.RunId)
}

func (c *CLI) getRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	flags, err := parseFlags("contractor run get", c.stderr, args, func(*flag.FlagSet) {})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor run get <run-id>")
	if err != nil {
		return err
	}
	status, err := getRunStatus(ctx, client, positionals[0])
	if err != nil {
		return err
	}
	return printRunStatus(printer, status)
}

type RunOutcomeError struct {
	RunID string
	State string
}

func (e *RunOutcomeError) Error() string {
	return fmt.Sprintf("Run %s finished with state %s", e.RunID, e.State)
}

func (c *CLI) watchRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var interval, waitTimeout time.Duration
	flags, err := parseFlags("contractor run watch", c.stderr, args, func(flags *flag.FlagSet) {
		flags.DurationVar(&interval, "interval", time.Second, "poll interval")
		flags.DurationVar(&waitTimeout, "wait-timeout", 0, "overall wait timeout; zero waits indefinitely")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor run watch <run-id> [--interval 1s] [--wait-timeout 10m]")
	if err != nil {
		return err
	}
	if interval < 100*time.Millisecond {
		return &UsageError{Message: "watch interval must be at least 100ms"}
	}
	if waitTimeout < 0 {
		return &UsageError{Message: "wait timeout cannot be negative"}
	}
	if waitTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, waitTimeout)
		defer cancel()
	}
	runID := positionals[0]
	lastState := ""
	for {
		status, getErr := getRunStatus(ctx, client, runID)
		if getErr != nil {
			return getErr
		}
		state := stringValue(status.State)
		if state != lastState {
			_, _ = fmt.Fprintf(c.stderr, "%s\t%s\n", time.Now().Format(time.RFC3339), state)
			lastState = state
		}
		if state == "succeeded" || state == "failed" || state == "cancelled" {
			if err := printRunStatus(printer, status); err != nil {
				return err
			}
			if state != "succeeded" {
				return &RunOutcomeError{RunID: runID, State: state}
			}
			return nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func (c *CLI) cancelRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	var reason string
	flags, err := parseFlags("contractor run cancel", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&reason, "reason", "", "human-readable cancellation reason")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor run cancel <run-id> [--reason TEXT]")
	if err != nil {
		return err
	}
	body := publicapi.CancelRunRequest{}
	if reason != "" {
		body.Reason = &reason
	}
	response, err := client.API.CancelRunWithResponse(ctx, positionals[0], &publicapi.CancelRunParams{}, body)
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK, http.StatusAccepted); err != nil {
		return err
	}
	result := response.JSON200
	if result == nil {
		result = response.JSON202
	}
	return printer.Object(result, result.RunId)
}

func (c *CLI) deleteRun(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	flags, err := parseFlags("contractor run delete", c.stderr, args, func(*flag.FlagSet) {})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor run delete <run-id>")
	if err != nil {
		return err
	}
	response, err := client.API.DeleteRunWithResponse(ctx, positionals[0], &publicapi.DeleteRunParams{})
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusNoContent); err != nil {
		return err
	}
	return printer.Names(positionals[0])
}

func (c *CLI) downloadRunOutput(ctx context.Context, client *publicclient.Client, args []string) error {
	var destination string
	var force bool
	flags, err := parseFlags("contractor run output", c.stderr, args, func(flags *flag.FlagSet) {
		flags.StringVar(&destination, "to", "-", "destination file, or - for stdout")
		flags.BoolVar(&force, "force", false, "replace an existing destination file")
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 2, 2, "usage: contractor run output <run-id> <slot> [--to FILE]")
	if err != nil {
		return err
	}
	response, err := client.API.DownloadRunOutputWithResponse(ctx, positionals[0], positionals[1])
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return err
	}
	if destination == "-" {
		_, err = c.stdout.Write(response.Body)
		return err
	}
	return writeDownloadedFile(destination, response.Body, force)
}

func getRunStatus(ctx context.Context, client *publicclient.Client, runID string) (*publicapi.RunStatus, error) {
	response, err := client.API.GetRunWithResponse(ctx, runID)
	if err != nil {
		return nil, err
	}
	if err := publicclient.CheckResponse(response, http.StatusOK); err != nil {
		return nil, err
	}
	return response.JSON200, nil
}

func printRunPage(printer *Printer, page *publicapi.RunPage) error {
	if printer.Mode() == OutputJSON {
		return printer.JSON(page)
	}
	if printer.Mode() == OutputName {
		names := make([]string, 0, len(page.Items))
		for _, run := range page.Items {
			names = append(names, run.RunId)
		}
		return printer.Names(names...)
	}
	rows := make([][]string, 0, len(page.Items))
	for _, run := range page.Items {
		project := "-"
		if run.ProjectId != nil {
			project = *run.ProjectId
		}
		rows = append(rows, []string{run.RunId, stringValue(run.State), run.Workflow, project, run.UpdatedAt.Format(time.RFC3339)})
	}
	return printer.Table([]string{"RUN", "STATE", "WORKFLOW", "PROJECT", "UPDATED"}, rows)
}

func printRunStatus(printer *Printer, status *publicapi.RunStatus) error {
	if printer.Mode() == OutputTable {
		project := "-"
		if status.ProjectId != nil {
			project = *status.ProjectId
		}
		return printer.Table([]string{"RUN", "STATE", "WORKFLOW", "PROJECT", "OUTPUTS", "DELETABLE"}, [][]string{{
			status.RunId, stringValue(status.State), status.Workflow, project, fmt.Sprint(len(status.Outputs)), fmt.Sprint(status.Deletable),
		}})
	}
	return printer.Object(status, status.RunId)
}

func validRunState(value string, nonTerminal bool) bool {
	for _, state := range []string{"initializing", "running", "cancelling", "succeeded", "failed", "cancelled"} {
		if state == value {
			return !nonTerminal || state == "initializing" || state == "running" || state == "cancelling"
		}
	}
	return false
}

func splitAssignment(value string) (string, string, error) {
	key, result, found := strings.Cut(value, "=")
	if !found || strings.TrimSpace(key) == "" {
		return "", "", errors.New("expected key=value")
	}
	return key, result, nil
}

func assignments(values []string) (map[string]string, error) {
	result := make(map[string]string, len(values))
	for _, value := range values {
		key, item, err := splitAssignment(value)
		if err != nil {
			return nil, err
		}
		if _, duplicate := result[key]; duplicate {
			return nil, fmt.Errorf("duplicate key %q", key)
		}
		result[key] = item
	}
	return result, nil
}

func (c *CLI) decodeRequestFile(path string, target any) error {
	var reader io.Reader = c.stdin
	var file *os.File
	if path != "-" {
		info, err := os.Lstat(filepath.Clean(path))
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return errors.New("request must be a regular file")
		}
		file, err = os.Open(filepath.Clean(path))
		if err != nil {
			return err
		}
		defer file.Close()
		reader = file
	}
	limited := &io.LimitedReader{R: reader, N: 1024*1024 + 1}
	decoder := json.NewDecoder(limited)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("decode Run request: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("Run request contains multiple JSON values")
		}
		return fmt.Errorf("decode Run request: %w", err)
	}
	if limited.N <= 0 {
		return errors.New("Run request exceeds 1 MiB")
	}
	return nil
}
