package cli

import (
	"context"
	"flag"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/publicclient"
	api "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

// Finding reads and edits always use the one Project owner, including sources
// retained after their Audit or Run has been deleted.
func (c *CLI) auditFindings(ctx context.Context, client *publicclient.Client, p *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "audit findings requires list, get, sources, source, history, edit, export, or download"}
	}
	op, args := args[0], args[1:]
	if op != "list" && op != "get" && op != "sources" && op != "source" && op != "history" && op != "edit" {
		return &UsageError{Message: "unsupported findings operation"}
	}
	page := op == "list" || op == "sources" || op == "history"
	var limit int
	var cursor, finding, revision, key, request string
	flags, err := parseFlags("contractor audit findings "+op, c.stderr, args, func(f *flag.FlagSet) {
		if page {
			auditPageFlags(f, &limit, &cursor)
		}
		if op == "sources" {
			f.StringVar(&finding, "finding", "", "only sources of this Project Finding")
		}
		if op == "edit" {
			f.StringVar(&revision, "revision", "", "exact Finding revision")
			f.StringVar(&key, "idempotency-key", "", "stable key for this edit")
			f.StringVar(&request, "request", "", "strict edit JSON file or -")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	n := 1
	if op == "get" || op == "source" || op == "history" || op == "edit" {
		n = 2
	}
	pos, err := requirePositionals(flags, n, n, "usage: contractor audit findings "+op+" PROJECT [FINDING_OR_RECEIPT] [flags]")
	if err != nil {
		return err
	}
	for _, id := range pos {
		if !contracts.ValidResourceID(id) {
			return &UsageError{Message: "invalid Project/Finding/source ID"}
		}
	}
	switch op {
	case "get":
		r, e := client.API.GetProjectFindingWithResponse(ctx, pos[0], pos[1])
		return printAuditReply(p, r, e, pos[1])
	case "source":
		r, e := client.API.GetProjectFindingSourceWithResponse(ctx, pos[0], pos[1])
		return printAuditReply(p, r, e, pos[1])
	case "edit":
		match, e := auditPrecondition(revision)
		if e != nil {
			return e
		}
		if !contracts.ValidIdempotencyKey(key) {
			return &UsageError{Message: "explicit valid --idempotency-key is required"}
		}
		var body api.EditProjectFindingRequest
		if e = c.readAuditRequest(request, &body); e != nil {
			return e
		}
		r, e := client.API.EditProjectFindingWithResponse(ctx, pos[0], pos[1], &api.EditProjectFindingParams{IdempotencyKey: key, IfMatch: match}, body)
		return printAuditReply(p, r, e, pos[1])
	}
	l, cur, e := auditPageArgs(limit, cursor)
	if e != nil {
		return e
	}
	if op != "list" && limit > 64 {
		return &UsageError{Message: "source pages allow at most 64 rows"}
	}
	switch op {
	case "list":
		r, e := client.API.ListProjectFindingsWithResponse(ctx, pos[0], &api.ListProjectFindingsParams{Limit: l, Cursor: cur})
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
			names = append(names, v.FindingId)
		}
		return p.Object(r.JSON200, names...)
	case "sources":
		params := &api.ListProjectFindingSourcesParams{Limit: l, Cursor: cur}
		if finding != "" {
			if !contracts.ValidResourceID(finding) {
				return &UsageError{Message: "invalid Finding ID"}
			}
			params.FindingId = &finding
		}
		r, e := client.API.ListProjectFindingSourcesWithResponse(ctx, pos[0], params)
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
			names = append(names, v.ReceiptId)
		}
		return p.Object(r.JSON200, names...)
	case "history":
		r, e := client.API.ListProjectFindingSourceHistoryWithResponse(ctx, pos[0], pos[1], &api.ListProjectFindingSourceHistoryParams{Limit: l, Cursor: cur})
		return printAuditReply(p, r, e, pos[1])
	}
	return nil
}
