package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/pagelimit"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/publicclient"
	api "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runAudit(ctx context.Context, client *publicclient.Client, p *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "audit requires profiles, list, create, get, start, pause, resume, cancel, delete, tasks, hypotheses, materials, findings, events, workspace, or report"}
	}
	switch args[0] {
	case "findings":
		if len(args) > 1 && (args[1] == "export" || args[1] == "download") {
			return c.runAuditFindingExport(ctx, client, p, args[1:])
		}
		return c.auditFindings(ctx, client, p, args[1:])
	case "tasks", "hypotheses", "materials":
		return c.auditResearch(ctx, client, p, args[0], args[1:])
	case "create", "start", "pause", "resume", "cancel", "delete":
		return c.auditMutation(ctx, client, p, args[0], args[1:])
	case "profiles", "list", "get", "events", "report", "workspace":
		return c.auditRead(ctx, client, p, args[0], args[1:])
	default:
		return &UsageError{Message: fmt.Sprintf("unknown audit command %q", args[0])}
	}
}

type auditReply interface {
	StatusCode() int
	GetBody() []byte
}

// Research pages retain their complete metadata and continuation in JSON.
// A caller must follow page.nextCursor even when the current page is empty.
func printAuditReply(p *Printer, r auditReply, err error, names ...string) error {
	if err != nil {
		return err
	}
	if err := publicclient.CheckResponse(r, http.StatusOK, http.StatusCreated, http.StatusAccepted); err != nil {
		return err
	}
	if !json.Valid(r.GetBody()) {
		return errors.New("Audit response has no JSON body")
	}
	return p.Object(json.RawMessage(r.GetBody()), names...)
}

func auditPageFlags(f *flag.FlagSet, limit *int, cursor *string) {
	f.IntVar(limit, "limit", 50, "maximum page rows")
	f.StringVar(cursor, "cursor", "", "opaque continuation from the previous page")
}

func auditPageArgs(limit int, cursor string) (*api.Limit, *api.Cursor, error) {
	l, err := validatedLimit(limit)
	if err != nil {
		return nil, nil, err
	}
	if len(cursor) > pagelimit.MaxCursorBytes {
		return nil, nil, &UsageError{Message: "cursor exceeds the public bound"}
	}
	if cursor == "" {
		return &l, nil, nil
	}
	return &l, &cursor, nil
}

func (c *CLI) auditRead(ctx context.Context, client *publicclient.Client, p *Printer, op string, args []string) error {
	var limit int
	var cursor, project string
	flags, err := parseFlags("contractor audit "+op, c.stderr, args, func(f *flag.FlagSet) {
		if op == "profiles" || op == "list" || op == "events" {
			auditPageFlags(f, &limit, &cursor)
		}
		if op == "list" {
			f.StringVar(&project, "project", "", "only this Project")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	n := 1
	if op == "list" || op == "profiles" {
		n = 0
	}
	pos, err := requirePositionals(flags, n, n, "usage: contractor audit "+op+" [AUDIT] [flags]")
	if err != nil {
		return err
	}
	if n == 1 && !contracts.ValidResourceID(pos[0]) {
		return &UsageError{Message: "invalid Audit ID"}
	}
	if op == "get" {
		r, e := client.API.GetAuditWithResponse(ctx, pos[0])
		return printAuditReply(p, r, e, pos[0])
	}
	if op == "workspace" {
		r, e := client.API.GetAuditWorkspaceWithResponse(ctx, pos[0])
		return printAuditReply(p, r, e, pos[0])
	}
	if op == "report" {
		r, e := client.API.GetAuditReportWithResponse(ctx, pos[0])
		return printAuditReply(p, r, e, pos[0])
	}
	l, cur, err := auditPageArgs(limit, cursor)
	if err != nil {
		return err
	}
	switch op {
	case "profiles":
		r, e := client.API.ListAuditProfilesWithResponse(ctx, &api.ListAuditProfilesParams{Limit: l, Cursor: cur})
		if e != nil {
			return e
		}
		if e = publicclient.CheckResponse(r, 200); e != nil {
			return e
		}
		if r.JSON200 == nil {
			return fmt.Errorf("Audit response has no JSON object")
		}
		names := []string{}
		for _, v := range r.JSON200.Items {
			names = append(names, v.Ref.Name+"@"+v.Ref.Version)
		}
		return printAuditReply(p, r, nil, names...)
	case "list":
		var page *api.AuditPage
		if project != "" {
			if !contracts.ValidResourceID(project) {
				return &UsageError{Message: "invalid Project ID"}
			}
			r, e := client.API.ListProjectAuditsWithResponse(ctx, project, &api.ListProjectAuditsParams{Limit: l, Cursor: cur})
			if e != nil {
				return e
			}
			if e = publicclient.CheckResponse(r, 200); e != nil {
				return e
			}
			page = r.JSON200
		} else {
			r, e := client.API.ListOwnerAuditsWithResponse(ctx, &api.ListOwnerAuditsParams{Limit: l, Cursor: cur})
			if e != nil {
				return e
			}
			if e = publicclient.CheckResponse(r, 200); e != nil {
				return e
			}
			page = r.JSON200
		}
		names := []string{}
		if page == nil {
			return errors.New("Audit response has no JSON object")
		}
		for _, v := range page.Items {
			names = append(names, v.AuditId)
		}
		return p.Object(page, names...)
	case "events":
		r, e := client.API.ListAuditEventsWithResponse(ctx, pos[0], &api.ListAuditEventsParams{Limit: l, Cursor: cur})
		return printAuditReply(p, r, e, pos[0])
	}
	return nil
}

func auditPrecondition(revision string) (string, error) {
	n, err := strconv.ParseUint(revision, 10, 64)
	if err != nil || n < 1 || n > contracts.MaxExactInteger || strconv.FormatUint(n, 10) != revision {
		return "", &UsageError{Message: "--revision must be the exact positive revision read from the resource"}
	}
	return strconv.Quote(revision), nil
}

// readAuditRequest validates duplicate keys and exact field spelling before the
// generated body is used. It shares the API's mutation metadata byte ceiling.
func (c *CLI) readAuditRequest(path string, target any) error {
	if path == "" {
		return &UsageError{Message: "--request FILE (or - for stdin) is required"}
	}
	r := c.stdin
	if path != "-" {
		f, e := openRegularInput(path, "Audit request")
		if e != nil {
			return e
		}
		defer f.Close()
		r = f
	}
	data, err := io.ReadAll(io.LimitReader(r, (1<<20)+1))
	if err != nil {
		return err
	}
	if len(data) > 1<<20 {
		return &UsageError{Message: "Audit request exceeds 1 MiB"}
	}
	if err := strictjson.DecodeNonNull(data, target); err != nil {
		return &UsageError{Message: "invalid strict Audit request JSON"}
	}
	return nil
}

func (c *CLI) auditMutation(ctx context.Context, client *publicclient.Client, p *Printer, op string, args []string) error {
	var request, key, revision string
	var deadline int
	var confirm bool
	flags, err := parseFlags("contractor audit "+op, c.stderr, args, func(f *flag.FlagSet) {
		f.StringVar(&key, "idempotency-key", "", "stable key; reuse with the same body and revision after response loss")
		if op == "create" {
			f.StringVar(&request, "request", "", "strict Audit creation JSON file or -")
		} else {
			f.StringVar(&revision, "revision", "", "exact Audit revision read before the operation")
		}
		if op == "start" || op == "resume" {
			f.IntVar(&deadline, "deadline-seconds", 0, "replace deadline allowance; zero removes it")
		}
		if op == "delete" {
			f.BoolVar(&confirm, "yes", false, "confirm deletion of this Audit")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	pos, err := requirePositionals(flags, 1, 1, "usage: contractor audit "+op+" ID --idempotency-key KEY [flags]")
	if err != nil {
		return err
	}
	if !contracts.ValidResourceID(pos[0]) || !contracts.ValidIdempotencyKey(key) {
		return &UsageError{Message: "valid ID and explicit --idempotency-key are required"}
	}
	if op == "create" {
		var body api.CreateAuditRequest
		if err := c.readAuditRequest(request, &body); err != nil {
			return err
		}
		r, e := client.API.CreateAuditWithResponse(ctx, pos[0], &api.CreateAuditParams{IdempotencyKey: key}, body)
		if e != nil {
			return e
		}
		if e = publicclient.CheckResponse(r, 201); e != nil {
			return e
		}
		if r.JSON201 != nil {
			return p.Object(r.JSON201, r.JSON201.AuditId)
		}
		return errors.New("Audit creation response has no body")
	}
	match, err := auditPrecondition(revision)
	if err != nil {
		return err
	}
	var body api.AuditTimeLimitRequest
	flags.Visit(func(f *flag.Flag) {
		if f.Name == "deadline-seconds" {
			body.DeadlineSeconds = &deadline
		}
	})
	switch op {
	case "start":
		r, e := client.API.StartAuditWithResponse(ctx, pos[0], &api.StartAuditParams{IdempotencyKey: key, IfMatch: match}, body)
		return printAuditReply(p, r, e, pos[0])
	case "resume":
		r, e := client.API.ResumeAuditWithResponse(ctx, pos[0], &api.ResumeAuditParams{IdempotencyKey: key, IfMatch: match}, body)
		return printAuditReply(p, r, e, pos[0])
	case "pause":
		r, e := client.API.PauseAuditWithResponse(ctx, pos[0], &api.PauseAuditParams{IdempotencyKey: key, IfMatch: match})
		return printAuditReply(p, r, e, pos[0])
	case "cancel":
		r, e := client.API.CancelAuditWithResponse(ctx, pos[0], &api.CancelAuditParams{IdempotencyKey: key, IfMatch: match})
		return printAuditReply(p, r, e, pos[0])
	case "delete":
		if !confirm {
			return &UsageError{Message: "Audit deletion requires --yes"}
		}
		r, e := client.API.DeleteAuditWithResponse(ctx, pos[0], &api.DeleteAuditParams{IdempotencyKey: key, IfMatch: match})
		return printAuditReply(p, r, e, pos[0])
	}
	return nil
}
