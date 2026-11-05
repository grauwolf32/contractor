package cli

import (
	"context"
	"flag"
	"fmt"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/publicclient"
	api "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) auditResearch(ctx context.Context, client *publicclient.Client, p *Printer, group string, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: group + " requires an operation"}
	}
	op, args := args[0], args[1:]
	allowed := map[string]map[string]bool{
		"tasks":      {"list": true, "get": true, "create": true, "attempts": true, "result": true},
		"hypotheses": {"list": true, "get": true, "create": true, "edit": true},
		"materials":  {"list": true, "get": true, "publish": true},
	}
	if !allowed[group][op] {
		return &UsageError{Message: "unsupported " + group + " operation"}
	}
	mutation := op == "create" || op == "edit" || op == "publish"
	page := op == "list" || op == "attempts"
	download := group == "materials" && op == "get" || op == "result"
	var limit int
	var cursor, revision, key, request, hypothesis, state, to string
	var force bool
	flags, err := parseFlags("contractor audit "+group+" "+op, c.stderr, args, func(f *flag.FlagSet) {
		if page {
			auditPageFlags(f, &limit, &cursor)
		}
		if mutation {
			f.StringVar(&request, "request", "", "strict request JSON file or -")
			f.StringVar(&key, "idempotency-key", "", "stable key for this exact request")
		}
		if mutation && op != "publish" || group == "hypotheses" && op == "get" || download || op == "publish" {
			f.StringVar(&revision, "revision", "", "exact resource revision; omit only for create-only material publication")
		}
		if group == "tasks" && op == "list" {
			f.StringVar(&hypothesis, "hypothesis", "", "filter by hypothesis ID")
			f.StringVar(&state, "state", "", "ready, selected, or settled")
		}
		if download {
			f.StringVar(&to, "to", "-", "destination file or - for stdout")
			f.BoolVar(&force, "force", false, "replace an existing destination file")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	n := 1
	if op == "get" || op == "edit" || op == "attempts" || op == "result" {
		n = 2
	}
	pos, err := requirePositionals(flags, n, n, "usage: contractor audit "+group+" "+op+" ID [MEMBER] [flags]")
	if err != nil {
		return err
	}
	if !contracts.ValidResourceID(pos[0]) {
		return &UsageError{Message: "invalid Audit or Project ID"}
	}
	if n == 2 && !(group == "materials" && op == "get") && !contracts.ValidResourceID(pos[1]) {
		return &UsageError{Message: "invalid member ID"}
	}
	if download {
		if contracts.ValidateOpaqueID("Artifact revision", revision) != nil {
			return &UsageError{Message: "--revision must name an exact material revision"}
		}
		logical := pos[1]
		if op == "result" {
			logical = "task/" + pos[1] + "/result"
		}
		r, e := client.API.GetAuditArtifactWithResponse(ctx, pos[0], &api.GetAuditArtifactParams{Key: logical, Revision: revision})
		if e != nil {
			return e
		}
		if e = publicclient.CheckResponse(r, 200); e != nil {
			return e
		}
		if to == "-" {
			_, e = c.stdout.Write(r.Body)
			return e
		}
		return writeDownloadedFile(to, r.Body, force)
	}
	if mutation {
		if !contracts.ValidIdempotencyKey(key) {
			return &UsageError{Message: "explicit valid --idempotency-key is required"}
		}
		if op == "publish" {
			var body api.AuditMaterialPublicationRequest
			if err := c.readAuditRequest(request, &body); err != nil {
				return err
			}
			params := &api.PublishAuditMaterialParams{IdempotencyKey: key}
			if revision == "" {
				v := api.PublishAuditMaterialParamsIfNoneMatch("*")
				params.IfNoneMatch = &v
			} else {
				if contracts.ValidateOpaqueID("Artifact revision", revision) != nil {
					return &UsageError{Message: "invalid Artifact revision"}
				}
				v := strconv.Quote(revision)
				params.IfMatch = &v
			}
			r, e := client.API.PublishAuditMaterialWithResponse(ctx, pos[0], params, body)
			return printAuditReply(p, r, e, body.Target.Namespace+"/"+body.Target.Name)
		}
		match, e := auditPrecondition(revision)
		if e != nil {
			return e
		}
		if group == "tasks" {
			var body api.AuditTaskCreationRequest
			if e = c.readAuditRequest(request, &body); e != nil {
				return e
			}
			r, e := client.API.CreateAuditTasksWithResponse(ctx, pos[0], &api.CreateAuditTasksParams{IdempotencyKey: key, IfMatch: match}, body)
			if e != nil {
				return e
			}
			if e = publicclient.CheckResponse(r, 200); e != nil {
				return e
			}
			if r.JSON200 == nil {
				return fmt.Errorf("Audit response has no JSON object")
			}
			return p.Object(r.JSON200, r.JSON200.TaskIds...)
		}
		var body api.AuditHypothesisContent
		if e = c.readAuditRequest(request, &body); e != nil {
			return e
		}
		if op == "create" {
			r, e := client.API.CreateAuditHypothesisWithResponse(ctx, pos[0], &api.CreateAuditHypothesisParams{IdempotencyKey: key, IfMatch: match}, body)
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
			for _, ref := range r.JSON200.Hypotheses {
				names = append(names, ref.Id)
			}
			return printAuditReply(p, r, nil, names...)
		}
		r, e := client.API.EditAuditHypothesisWithResponse(ctx, pos[0], pos[1], &api.EditAuditHypothesisParams{IdempotencyKey: key, IfMatch: match}, body)
		return printAuditReply(p, r, e, pos[1])
	}
	if op == "get" {
		if group == "tasks" {
			r, e := client.API.GetAuditTaskWithResponse(ctx, pos[0], pos[1])
			return printAuditReply(p, r, e, pos[1])
		}
		_, e := auditPrecondition(revision)
		if e != nil {
			return e
		}
		number, _ := strconv.Atoi(revision)
		r, e := client.API.GetAuditHypothesisWithResponse(ctx, pos[0], pos[1], &api.GetAuditHypothesisParams{Revision: number})
		return printAuditReply(p, r, e, pos[1])
	}
	l, cur, e := auditPageArgs(limit, cursor)
	if e != nil {
		return e
	}
	switch {
	case op == "attempts":
		r, e := client.API.ListAuditTaskAttemptsWithResponse(ctx, pos[0], pos[1], &api.ListAuditTaskAttemptsParams{Limit: l, Cursor: cur})
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
		for _, a := range r.JSON200.Items {
			names = append(names, a.ExecutionId)
		}
		return p.Object(r.JSON200, names...)
	case group == "materials":
		r, e := client.API.ListAuditMaterialsWithResponse(ctx, pos[0], &api.ListAuditMaterialsParams{Limit: l, Cursor: cur})
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
		for _, a := range r.JSON200.Items {
			names = append(names, a.Key)
		}
		return p.Object(r.JSON200, names...)
	case group == "hypotheses":
		r, e := client.API.ListAuditHypothesesWithResponse(ctx, pos[0], &api.ListAuditHypothesesParams{Limit: l, Cursor: cur})
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
		for _, a := range r.JSON200.Items {
			names = append(names, a.Document.Id)
		}
		return p.Object(r.JSON200, names...)
	case group == "tasks":
		params := &api.ListAuditTasksParams{Limit: l, Cursor: cur}
		if hypothesis != "" {
			if !contracts.ValidResourceID(hypothesis) {
				return &UsageError{Message: "invalid hypothesis ID"}
			}
			params.HypothesisId = &hypothesis
		}
		if state != "" {
			if state != "ready" && state != "selected" && state != "settled" {
				return &UsageError{Message: "invalid Task state"}
			}
			v := api.AuditTaskState(state)
			params.State = &v
		}
		r, e := client.API.ListAuditTasksWithResponse(ctx, pos[0], params)
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
		for _, a := range r.JSON200.Items {
			names = append(names, a.Document.TaskId)
		}
		return p.Object(r.JSON200, names...)
	}
	return fmt.Errorf("unsupported Audit research command")
}
