package cli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"sort"
	"strconv"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
)

func (c *CLI) runRemoteScanSweep(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "scan-sweep remote requires create, list, show, members, cancel, or purge"}
	}
	action := args[0]
	if action == "create" {
		return c.createRemoteScanSweep(ctx, client, printer, args[1:])
	}
	var limit int
	var cursor string
	var allPages, yes bool
	flags, err := parseFlags("contractor scan-sweep remote "+action, c.stderr, args[1:], func(f *flag.FlagSet) {
		if action == "list" || action == "members" {
			f.IntVar(&limit, "limit", 50, "maximum rows per page (1..200)")
			f.StringVar(&cursor, "cursor", "", "Server pagination cursor")
		}
		if action == "members" {
			f.BoolVar(&allPages, "all-pages", false, "read every member page (at most 1000)")
		}
		if action == "purge" {
			f.BoolVar(&yes, "yes", false, "confirm permanent removal of sweep metadata")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	count := 1
	if action == "list" {
		count = 0
	}
	positional, err := requirePositionals(flags, count, count, "usage: contractor scan-sweep remote "+action+" [sweep-id] [flags]")
	if err != nil {
		return err
	}
	id := ""
	if count == 1 {
		id = positional[0]
		if !contracts.ValidResourceID(id) {
			return &UsageError{Message: "invalid sweep ID"}
		}
	}
	var pageLimit publicapi.Limit
	if action == "list" || action == "members" {
		pageLimit, err = validatedLimit(limit)
		if err != nil {
			return err
		}
		if len(cursor) > pagelimit.MaxCursorBytes {
			return &UsageError{Message: "cursor exceeds its byte limit"}
		}
	}
	cursorValue := func(value string) *publicapi.Cursor {
		if value == "" {
			return nil
		}
		v := publicapi.Cursor(value)
		return &v
	}
	switch action {
	case "list":
		response, err := client.API.ListScanSweeps(ctx, &publicapi.ListScanSweepsParams{Limit: &pageLimit, Cursor: cursorValue(cursor)})
		page, err := sweepResponse[publicapi.ScanSweepPage](response, err, http.StatusOK)
		if err != nil {
			return err
		}
		if len(page.Items) > limit {
			return errors.New("sweep API exceeded its page limit")
		}
		for _, s := range page.Items {
			if err := validateRemoteSweep(s); err != nil {
				return err
			}
		}
		if _, err := remoteSweepCursor(page.Page, len(page.Items)); err != nil {
			return err
		}
		return List(printer, page, page.Items, func(s publicapi.ScanSweep) string { return s.SweepId }, remoteSweepHeaders(), remoteSweepRow)
	case "show", "get", "cancel":
		var response *http.Response
		if action == "cancel" {
			response, err = client.API.CancelScanSweep(ctx, id, nil) // The key is optional and not bound (API-R28).
		} else {
			response, err = client.API.GetScanSweep(ctx, id)
		}
		sweep, err := sweepResponse[publicapi.ScanSweep](response, err, http.StatusOK)
		if err != nil {
			return err
		}
		if err := validateRemoteSweep(sweep); err != nil {
			return err
		}
		if sweep.SweepId != id {
			return errors.New("sweep API returned another identity")
		}
		return printRemoteSweep(printer, sweep)
	case "members":
		seenCursors := map[string]bool{}
		if cursor != "" {
			seenCursors[cursor] = true
		}
		seenProjects := map[string]bool{}
		lastOrdinal := -1
		members := []publicapi.ScanSweepMember{}
		for {
			response, err := client.API.ListScanSweepMembers(ctx, id, &publicapi.ListScanSweepMembersParams{Limit: &pageLimit, Cursor: cursorValue(cursor)})
			page, err := sweepResponse[publicapi.ScanSweepMemberPage](response, err, http.StatusOK)
			if err != nil {
				return err
			}
			if len(page.Items) > limit {
				return errors.New("sweep API exceeded its page limit")
			}
			for _, m := range page.Items {
				if !m.State.Valid() || !contracts.ValidResourceID(m.ProjectId) || m.Ordinal < 0 || m.Ordinal >= maxSweepProjects || m.Ordinal <= lastOrdinal || seenProjects[m.ProjectId] || len(m.Target) > 2048 || len(m.Reason) > 128 {
					return errors.New("sweep API returned invalid or duplicate members")
				}
				if runID, e := m.RunId.Get(); e == nil {
					if !contracts.ValidResourceID(runID) || m.State != publicapi.ScanSweepMemberAdmitted {
						return errors.New("sweep API returned invalid Run identity")
					}
				} else if m.State == publicapi.ScanSweepMemberAdmitted {
					return errors.New("admitted sweep member has no Run identity")
				}
				if state, e := m.RunState.Get(); e == nil {
					if !validRunState(string(state), false) {
						return errors.New("sweep API returned invalid Run state")
					}
				}
				lastOrdinal = m.Ordinal
				seenProjects[m.ProjectId] = true
			}
			next, err := remoteSweepCursor(page.Page, len(page.Items))
			if err != nil {
				return err
			}
			if !allPages {
				return printRemoteMembers(printer, page, page.Items)
			}
			if len(members)+len(page.Items) > maxSweepProjects {
				return errors.New("sweep API exceeded its member limit")
			}
			members = append(members, page.Items...)
			if next == "" {
				return printRemoteMembers(printer, members, members)
			}
			if len(members) == maxSweepProjects {
				return errors.New("sweep API continued beyond its member limit")
			}
			if seenCursors[next] {
				return errors.New("sweep API repeated a pagination cursor")
			}
			seenCursors[next] = true
			cursor = next
		}
	case "purge":
		if !yes {
			return &UsageError{Message: "purge requires --yes to confirm removal of sweep metadata"}
		}
		response, err := client.API.PurgeScanSweep(ctx, id)
		if _, err := sweepResponse[struct{}](response, err, http.StatusNoContent); err != nil {
			return err
		}
		return printer.Object(map[string]any{"sweepId": id, "purged": true}, id)
	default:
		return &UsageError{Message: fmt.Sprintf("unknown remote sweep command %q", action)}
	}
}

func (c *CLI) createRemoteScanSweep(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) (resultErr error) {
	var projects, parameters stringList
	var all bool
	var workflow, rulepack, key string
	flags, err := parseFlags("contractor scan-sweep remote create", c.stderr, args, func(f *flag.FlagSet) {
		f.Var(&projects, "project", "selected Project ID (repeatable)")
		f.BoolVar(&all, "all", false, "capture every owner Project on the Server (maximum 1000)")
		f.StringVar(&workflow, "workflow", "nuclei-web@1", "exact Workflow selector")
		f.Var(&parameters, "param", "rule parameter key=value; targets come from Projects")
		f.StringVar(&rulepack, "rulepack", "", "exact owner UserScope namespace/name@revision")
		f.StringVar(&key, "idempotency-key", "", "retry-safe create key (generated when omitted)")
	})
	if err != nil || flags == nil {
		return err
	}
	if _, err := requirePositionals(flags, 0, 0, "usage: contractor scan-sweep remote create (--all | --project ID...) [flags]"); err != nil {
		return err
	}
	if all == (len(projects) > 0) || len(projects) > maxSweepProjects {
		return &UsageError{Message: "select either --all or up to 1000 unique --project IDs"}
	}
	ids := append([]publicapi.ResourceId{}, projects...)
	sort.Strings(ids)
	for i, id := range ids {
		if !contracts.ValidResourceID(id) || (i > 0 && ids[i-1] == id) {
			return &UsageError{Message: "invalid or duplicate Project ID"}
		}
	}
	if _, _, err := exactSelector(workflow); err != nil {
		return err
	}
	params, err := assignments(parameters)
	if err != nil {
		return &UsageError{Message: "invalid --param assignment"}
	}
	if _, ok := params["target"]; ok {
		return &UsageError{Message: "target is captured by the Server from each Project"}
	}
	if len(params) > 32 {
		return &UsageError{Message: "too many rule parameters"}
	}
	for name, value := range params {
		if len(name) > 128 || len(value) > 2048 {
			return &UsageError{Message: "rule parameter exceeds its byte limit"}
		}
	}
	body := publicapi.CreateScanSweepRequest{Workflow: &workflow, Parameters: &params}
	if all {
		body.AllProjects = &all
	} else {
		body.ProjectIds = &ids
	}
	if rulepack != "" {
		namespace, name, revision, err := parseArtifactRef(rulepack)
		if err != nil {
			return err
		}
		ref := contracts.ArtifactRef{Namespace: namespace, Name: name, Revision: &revision}
		if revision == "" || ref.ValidateExact() != nil {
			return &UsageError{Message: "rulepack requires an exact namespace/name@revision"}
		}
		body.Rulepack = &publicapi.ExactArtifactRef{Namespace: namespace, Name: name, Revision: revision}
	}
	generated := key == ""
	key, err = idempotencyKey(key)
	if err != nil {
		return err
	}
	if !contracts.ValidResourceID(key) || len(key) > 128 {
		return &UsageError{Message: "invalid idempotency key"}
	}
	if generated {
		defer func() { c.printCreateRetryHint(key, resultErr) }()
	}
	response, err := client.API.CreateScanSweep(ctx, &publicapi.CreateScanSweepParams{IdempotencyKey: key}, body)
	sweep, err := sweepResponse[publicapi.ScanSweep](response, err, http.StatusCreated)
	if err != nil {
		return err
	}
	if err := validateRemoteSweep(sweep); err != nil {
		return err
	}
	return printRemoteSweep(printer, sweep)
}

func remoteSweepCursor(page publicapi.PageInfo, count int) (string, error) {
	if !page.HasMore {
		return "", nil
	}
	if count == 0 || page.NextCursor == nil || *page.NextCursor == "" || len(*page.NextCursor) > pagelimit.MaxCursorBytes {
		return "", errors.New("sweep API returned invalid pagination")
	}
	return *page.NextCursor, nil
}
func validateRemoteSweep(s publicapi.ScanSweep) error {
	counts := s.Counts
	if !contracts.ValidResourceID(s.SweepId) || !s.State.Valid() || !contentdigest.Valid(s.SnapshotDigest) || s.CreatedAt.IsZero() || counts.Total < 1 || counts.Total > maxSweepProjects || counts.Pending < 0 || counts.Admitted < 0 || counts.Skipped < 0 || counts.Failed < 0 || counts.Cancelled < 0 || counts.Unsettled < 0 || counts.Unsettled > counts.Total || counts.Pending+counts.Admitted+counts.Skipped+counts.Failed+counts.Cancelled != counts.Total {
		return errors.New("sweep API returned invalid state or counters")
	}
	if _, _, err := exactSelector(s.Workflow); err != nil {
		return errors.New("sweep API returned invalid Workflow identity")
	}
	terminal := s.State == publicapi.ScanSweepCompleted || s.State == publicapi.ScanSweepCancelled
	if terminal {
		if _, err := s.SettledAt.Get(); err != nil || counts.Unsettled != 0 {
			return errors.New("sweep API returned invalid settlement")
		}
	}
	return nil
}
func remoteSweepHeaders() []string {
	return []string{"ID", "STATE", "TOTAL", "PENDING", "ADMITTED", "SKIPPED", "ADMISSION ERRORS", "CANCELLED", "UNSETTLED"}
}
func remoteSweepRow(s publicapi.ScanSweep) []string {
	return []string{s.SweepId, string(s.State), strconv.Itoa(s.Counts.Total), strconv.Itoa(s.Counts.Pending), strconv.Itoa(s.Counts.Admitted), strconv.Itoa(s.Counts.Skipped), strconv.Itoa(s.Counts.Failed), strconv.Itoa(s.Counts.Cancelled), strconv.Itoa(s.Counts.Unsettled)}
}
func printRemoteSweep(p *Printer, s publicapi.ScanSweep) error {
	if p.Mode() == OutputTable {
		return p.Table(remoteSweepHeaders(), [][]string{remoteSweepRow(s)})
	}
	return p.Object(s, s.SweepId)
}
func printRemoteMembers(p *Printer, value any, members []publicapi.ScanSweepMember) error {
	return List(p, value, members, func(m publicapi.ScanSweepMember) string { return m.ProjectId }, []string{"PROJECT", "TARGET", "ADMISSION", "RUN", "RUN STATE", "REASON"}, func(m publicapi.ScanSweepMember) []string {
		run, _ := m.RunId.Get()
		state, _ := m.RunState.Get()
		return []string{m.ProjectId, m.Target, string(m.State), run, stringValue(state), m.Reason}
	})
}
