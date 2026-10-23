package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/grauwolf32/contractor/internal/publicclient"
	publicapi "github.com/grauwolf32/contractor/internal/publicclient/generated"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

const (
	maxSweepProjects      = 1000
	maxSweepManifestBytes = 8 * 1024 * 1024
	maxSweepRequestBytes  = 16 * 1024
	maxSweepResponseBytes = 4 * 1024 * 1024
)

type scanSweepMember struct {
	ProjectID     string          `json:"projectId"`
	Key           string          `json:"key,omitempty"`
	Request       json.RawMessage `json:"request,omitempty"`
	RequestDigest string          `json:"requestDigest,omitempty"`
	SkipReason    string          `json:"skipReason,omitempty"`
}
type scanSweepSnapshot struct {
	ID      string            `json:"id"`
	Server  string            `json:"server"`
	Members []scanSweepMember `json:"members"`
}
type scanSweepOutcome struct {
	ProjectID string `json:"projectId"`
	State     string `json:"state"`
	RunID     string `json:"runId,omitempty"`
	ErrorCode string `json:"errorCode,omitempty"`
}
type scanSweepManifest struct {
	SchemaVersion  int                `json:"schemaVersion"`
	Snapshot       scanSweepSnapshot  `json:"snapshot"`
	SnapshotDigest string             `json:"snapshotDigest"`
	Outcomes       []scanSweepOutcome `json:"outcomes"`
}

func (c *CLI) runScanSweep(ctx context.Context, client *publicclient.Client, printer *Printer, args []string) error {
	if len(args) == 0 {
		return &UsageError{Message: "scan-sweep requires create, resume, or status"}
	}
	action := args[0]
	if action != "create" && action != "resume" && action != "status" {
		return &UsageError{Message: "unknown scan-sweep command"}
	}
	var projects, parameters, artifactArgs stringList
	var all bool
	var workflow string
	var deadline time.Duration
	flags, err := parseFlags("contractor scan-sweep "+action, c.stderr, args[1:], func(f *flag.FlagSet) {
		f.DurationVar(&deadline, "deadline", 30*time.Minute, "total invocation deadline, independent of --timeout per API attempt")
		if action == "create" {
			f.Var(&projects, "project", "selected Project ID (repeatable)")
			f.BoolVar(&all, "all", false, "capture every Project page before creating Runs (maximum 1000)")
			f.StringVar(&workflow, "workflow", "nuclei-web@1", "exact Workflow selector")
			f.Var(&parameters, "param", "captured key=value parameter (target comes from each Project)")
			f.Var(&artifactArgs, "artifact", "slot=namespace/name[@revision], resolved once per Project")
		}
	})
	if err != nil || flags == nil {
		return err
	}
	positionals, err := requirePositionals(flags, 1, 1, "usage: contractor scan-sweep "+action+" MANIFEST [flags]")
	if err != nil {
		return err
	}
	if deadline <= 0 || deadline > 24*time.Hour {
		return &UsageError{Message: "--deadline must be positive and at most 24h"}
	}
	path, err := filepath.Abs(positionals[0])
	if err != nil {
		return err
	}
	if action == "create" {
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return err
		}
	}
	unlock, err := lockSweepManifest(path)
	if err != nil {
		return err
	}
	defer unlock()
	ctx, cancel := context.WithTimeout(ctx, deadline)
	defer cancel()
	var manifest scanSweepManifest
	if action == "create" {
		if _, err := os.Lstat(path); !errors.Is(err, os.ErrNotExist) {
			return errors.New("sweep manifest already exists or is inaccessible; use resume")
		}
		if all == (len(projects) > 0) {
			return &UsageError{Message: "select either --project IDs or --all"}
		}
		if _, _, err := exactSelector(workflow); err != nil {
			return err
		}
		params, err := assignments(parameters)
		if err != nil {
			return &UsageError{Message: "invalid --param assignment"}
		}
		if _, supplied := params["target"]; supplied {
			return &UsageError{Message: "target is captured from each Project; do not supply --param target"}
		}
		refs, err := sweepArtifactArguments(artifactArgs)
		if err != nil {
			return err
		}
		manifest, err = c.captureScanSweep(ctx, client, projects, all, workflow, params, refs)
		if err != nil {
			return err
		}
		// Publish the complete immutable request snapshot before the first POST.
		if err := saveScanSweep(path, &manifest, true); err != nil {
			return err
		}
	} else {
		manifest, err = loadScanSweep(path)
		if err != nil {
			return err
		}
	}
	if manifest.Snapshot.Server != client.Origin() {
		return errors.New("sweep manifest belongs to another Server")
	}
	if action != "status" {
		if err := dispatchScanSweep(ctx, client, path, &manifest); err != nil {
			return err
		}
	}
	rows := make([][]string, 0, len(manifest.Outcomes))
	names := make([]string, 0, len(manifest.Outcomes))
	for _, outcome := range manifest.Outcomes {
		rows = append(rows, []string{outcome.ProjectID, outcome.State, outcome.RunID, outcome.ErrorCode})
		if outcome.RunID != "" {
			names = append(names, outcome.RunID)
		}
	}
	var outputErr error
	switch printer.Mode() {
	case OutputJSON:
		outputErr = printer.JSON(manifest.Outcomes)
	case OutputName:
		outputErr = printer.Names(names...)
	default:
		outputErr = printer.Table([]string{"PROJECT", "STATE", "RUN", "REASON"}, rows)
	}
	if outputErr != nil {
		return outputErr
	}
	if action != "status" {
		for _, outcome := range manifest.Outcomes {
			if outcome.State == "unknown" || outcome.State == "failed" {
				return errors.New("sweep has unresolved members; inspect reasons and resume unchanged requests, or create a new sweep for changed inputs")
			}
		}
	}
	return nil
}

func sweepArtifactArguments(values []string) (map[string]publicapi.ArtifactRef, error) {
	refs := map[string]publicapi.ArtifactRef{}
	if len(values) > 32 {
		return nil, &UsageError{Message: "too many artifact inputs"}
	}
	for _, value := range values {
		slot, reference, err := splitAssignment(value)
		if err != nil {
			return nil, err
		}
		if contracts.ValidateArtifactName(slot) != nil {
			return nil, &UsageError{Message: "invalid artifact slot"}
		}
		if _, exists := refs[slot]; exists {
			return nil, &UsageError{Message: "duplicate artifact slot"}
		}
		namespace, name, revision, err := parseArtifactRef(reference)
		if err != nil {
			return nil, err
		}
		if contracts.ValidateArtifactName(namespace) != nil || contracts.ValidateArtifactName(name) != nil || (revision != "" && contracts.ValidateResourceID("revision", revision) != nil) {
			return nil, &UsageError{Message: "invalid artifact reference"}
		}
		ref := publicapi.ArtifactRef{Namespace: namespace, Name: name}
		if revision != "" {
			ref.Revision.Set(revision)
		}
		refs[slot] = ref
	}
	return refs, nil
}

func sweepResponse[T any](response *http.Response, callErr error, expected int) (T, error) {
	var result T
	if callErr != nil {
		return result, callErr
	}
	if response == nil || response.Body == nil {
		return result, errors.New("sweep API returned no response")
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxSweepResponseBytes+1))
	if err != nil {
		return result, err
	}
	if len(data) > maxSweepResponseBytes {
		return result, errors.New("sweep API response exceeds its byte limit")
	}
	if response.StatusCode != expected {
		return result, publicclient.DecodeError(response.StatusCode, data)
	}
	if err := strictjson.Decode(data, &result); err != nil {
		return result, errors.New("sweep API returned invalid JSON")
	}
	return result, nil
}

func (c *CLI) captureScanSweep(ctx context.Context, client *publicclient.Client, ids []string, all bool, workflow string, parameters map[string]string, refs map[string]publicapi.ArtifactRef) (scanSweepManifest, error) {
	manifest := scanSweepManifest{SchemaVersion: 1, Snapshot: scanSweepSnapshot{Server: client.Origin(), Members: []scanSweepMember{}}, Outcomes: []scanSweepOutcome{}}
	id, err := publicclient.NewIdempotencyKey()
	if err != nil {
		return manifest, err
	}
	manifest.Snapshot.ID = "scan-sweep-" + id
	projects := map[string]publicapi.Project{}
	unavailable := map[string]bool{}
	if all {
		limit := publicapi.Limit(pagelimit.MaxRows)
		params := &publicapi.ListProjectsParams{Limit: &limit}
		seenCursors := map[string]bool{}
		for {
			response, callErr := client.API.ListProjects(ctx, params)
			page, err := sweepResponse[publicapi.ProjectPage](response, callErr, http.StatusOK)
			if err != nil {
				return manifest, err
			}
			if len(page.Items) > int(limit) || (page.Page.HasMore && len(page.Items) == 0) {
				return manifest, errors.New("invalid Project page")
			}
			for _, project := range page.Items {
				if !contracts.ValidResourceID(project.ProjectId) {
					return manifest, errors.New("invalid Project identity")
				}
				if _, found := projects[project.ProjectId]; found {
					return manifest, errors.New("duplicate Project in pagination")
				}
				projects[project.ProjectId] = project
			}
			if len(projects) > maxSweepProjects {
				return manifest, errors.New("sweep exceeds 1000 Projects")
			}
			if !page.Page.HasMore {
				break
			}
			if page.Page.NextCursor == nil || *page.Page.NextCursor == "" || len(*page.Page.NextCursor) > pagelimit.MaxCursorBytes || seenCursors[*page.Page.NextCursor] {
				return manifest, errors.New("invalid Project pagination cursor")
			}
			seenCursors[*page.Page.NextCursor] = true
			params.Cursor = page.Page.NextCursor
		}
	} else {
		if len(ids) > maxSweepProjects {
			return manifest, errors.New("sweep exceeds 1000 Projects")
		}
		for _, id := range ids {
			if !contracts.ValidResourceID(id) {
				return manifest, &UsageError{Message: "invalid Project identity"}
			}
			if _, exists := projects[id]; exists {
				return manifest, &UsageError{Message: "duplicate Project identity"}
			}
			response, callErr := client.API.GetProject(ctx, id)
			project, err := sweepResponse[publicapi.Project](response, callErr, http.StatusOK)
			if err != nil {
				var apiErr *publicclient.APIError
				if errors.As(err, &apiErr) && apiErr.Status == http.StatusNotFound {
					projects[id] = publicapi.Project{ProjectId: id}
					unavailable[id] = true
					continue
				}
				return manifest, err
			}
			if project.ProjectId != id {
				return manifest, errors.New("Project response identity mismatch")
			}
			projects[id] = project
		}
	}
	if len(projects) == 0 {
		return manifest, errors.New("sweep selected no Projects")
	}
	sorted := make([]string, 0, len(projects))
	for id := range projects {
		sorted = append(sorted, id)
	}
	sort.Strings(sorted)
	for _, id := range sorted {
		if err := ctx.Err(); err != nil {
			return manifest, err
		}
		project := projects[id]
		member := scanSweepMember{ProjectID: id}
		outcome := scanSweepOutcome{ProjectID: id, State: "pending"}
		switch {
		case unavailable[id]:
			member.SkipReason = "project_unavailable"
		case project.Lifecycle == publicapi.Deleting:
			member.SkipReason = "project_deleting"
		case project.Lifecycle != publicapi.Active:
			return manifest, errors.New("invalid Project lifecycle")
		case project.HttpTarget == nil:
			member.SkipReason = "http_target_missing"
		}
		if member.SkipReason != "" {
			outcome.State = "skipped"
			outcome.ErrorCode = member.SkipReason
		} else {
			body := publicapi.CreateRunRequest{Workflow: workflow}
			values := map[string]string{}
			for k, v := range parameters {
				values[k] = v
			}
			values["target"] = project.HttpTarget.Url
			body.Parameters.Set(values)
			labels := publicapi.RunMetadataLabels{"scan-sweep": manifest.Snapshot.ID}
			body.Labels = &labels
			exacts := map[string]publicapi.ArtifactRef{}
			for slot, ref := range refs {
				params := &publicapi.GetProjectArtifactMetadataParams{}
				if ref.Revision.IsSpecified() {
					revision, err := ref.Revision.Get()
					if err != nil {
						return manifest, err
					}
					params.Revision = &revision
				}
				response, callErr := client.API.GetProjectArtifactMetadata(ctx, id, ref.Namespace, ref.Name, params)
				metadata, err := sweepResponse[publicapi.ArtifactMetadata](response, callErr, http.StatusOK)
				if err != nil {
					var apiErr *publicclient.APIError
					if errors.As(err, &apiErr) && apiErr.Status >= 400 && apiErr.Status < 500 && !apiErr.Retryable {
						member.SkipReason = "input_artifact_unavailable"
						break
					}
					return manifest, err
				}
				exact := metadata.Artifact
				if exact.Namespace != ref.Namespace || exact.Name != ref.Name || contracts.ValidateResourceID("revision", exact.Revision) != nil || (params.Revision != nil && exact.Revision != *params.Revision) {
					return manifest, errors.New("artifact metadata identity mismatch")
				}
				pinned := publicapi.ArtifactRef{Namespace: exact.Namespace, Name: exact.Name}
				pinned.Revision.Set(exact.Revision)
				exacts[slot] = pinned
			}
			if member.SkipReason != "" {
				outcome.State = "skipped"
				outcome.ErrorCode = member.SkipReason
			} else {
				body.Artifacts.Set(exacts)
				request, err := strictjson.Canonical(body)
				if err != nil {
					return manifest, err
				}
				if len(request) > maxSweepRequestBytes {
					return manifest, errors.New("sweep child request exceeds 16 KiB")
				}
				member.Request = request
				member.RequestDigest = contentdigest.Bytes(request)
				member.Key = sweepMemberKey(manifest.Snapshot.ID, id, member.RequestDigest)
			}
		}

		manifest.Snapshot.Members = append(manifest.Snapshot.Members, member)
		manifest.Outcomes = append(manifest.Outcomes, outcome)
	}
	canonical, err := strictjson.Canonical(manifest.Snapshot)
	if err != nil {
		return manifest, err
	}
	manifest.SnapshotDigest = contentdigest.Bytes(canonical)
	return manifest, validateScanSweep(&manifest)
}

func sweepMemberKey(id, project, digest string) string {
	return "scan-sweep-" + strings.TrimPrefix(contentdigest.Bytes([]byte(id+"\n"+project+"\n"+digest)), "sha256:")
}

func validateScanSweep(manifest *scanSweepManifest) error {
	invalid := func() error {
		return errors.New("invalid or changed sweep manifest; changed inputs require a new sweep")
	}
	if manifest.SchemaVersion != 1 || !contracts.ValidResourceID(manifest.Snapshot.ID) || len(manifest.Snapshot.Members) == 0 || len(manifest.Snapshot.Members) > maxSweepProjects || len(manifest.Outcomes) != len(manifest.Snapshot.Members) {
		return invalid()
	}
	origin, err := publicclient.NormalizeServer(manifest.Snapshot.Server, true)
	if err != nil || origin != manifest.Snapshot.Server {
		return invalid()
	}
	canonical, err := strictjson.Canonical(manifest.Snapshot)
	if err != nil || contentdigest.Bytes(canonical) != manifest.SnapshotDigest {
		return invalid()
	}
	previous := ""
	for i, member := range manifest.Snapshot.Members {
		outcome := manifest.Outcomes[i]
		if !contracts.ValidResourceID(member.ProjectID) || member.ProjectID <= previous || member.ProjectID != outcome.ProjectID {
			return invalid()
		}
		previous = member.ProjectID
		if member.SkipReason != "" {
			if member.SkipReason != "project_deleting" && member.SkipReason != "http_target_missing" && member.SkipReason != "project_unavailable" && member.SkipReason != "input_artifact_unavailable" {
				return invalid()
			}
			if len(member.Request) != 0 || member.Key != "" || member.RequestDigest != "" || outcome.State != "skipped" || outcome.ErrorCode != member.SkipReason || outcome.RunID != "" {
				return invalid()
			}
			continue
		}
		if len(member.Request) == 0 || len(member.Request) > maxSweepRequestBytes || contentdigest.Bytes(member.Request) != member.RequestDigest || member.Key != sweepMemberKey(manifest.Snapshot.ID, member.ProjectID, member.RequestDigest) {
			return invalid()
		}
		var request publicapi.CreateRunRequest
		if strictjson.Decode(member.Request, &request) != nil {
			return invalid()
		}
		normalized, err := strictjson.Canonical(request)
		if err != nil || !bytes.Equal(normalized, member.Request) {
			return invalid()
		}
		if _, _, err := exactSelector(request.Workflow); err != nil {
			return invalid()
		}
		artifacts, err := request.Artifacts.Get()
		if err != nil || len(artifacts) > 32 {
			return invalid()
		}
		for _, ref := range artifacts {
			revision, err := ref.Revision.Get()
			if err != nil || (contracts.ArtifactRef{Namespace: ref.Namespace, Name: ref.Name, Revision: &revision}).ValidateExact() != nil {
				return invalid()
			}
		}
		parameters, err := request.Parameters.Get()
		if err != nil || parameters["target"] == "" {
			return invalid()
		}
		switch outcome.State {
		case "created":
			if !contracts.ValidResourceID(outcome.RunID) || outcome.ErrorCode != "" {
				return invalid()
			}
		case "pending":
			if outcome.RunID != "" || outcome.ErrorCode != "" {
				return invalid()
			}
		case "unknown", "failed":
			if outcome.RunID != "" || !contracts.ValidResourceID(outcome.ErrorCode) {
				return invalid()
			}
		default:
			return invalid()
		}
	}
	return nil
}

func loadScanSweep(path string) (scanSweepManifest, error) {
	var manifest scanSweepManifest
	info, err := os.Lstat(path)
	if err != nil {
		return manifest, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxSweepManifestBytes {
		return manifest, errors.New("sweep manifest must be a bounded regular file")
	}
	file, err := os.Open(path)
	if err != nil {
		return manifest, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxSweepManifestBytes+1))
	if err != nil {
		return manifest, err
	}
	if len(data) > maxSweepManifestBytes {
		return manifest, errors.New("sweep manifest exceeds 8 MiB")
	}
	if err := strictjson.Decode(data, &manifest); err != nil {
		return manifest, errors.New("invalid sweep manifest JSON")
	}
	return manifest, validateScanSweep(&manifest)
}

func saveScanSweep(path string, manifest *scanSweepManifest, create bool) error {
	if err := validateScanSweep(manifest); err != nil {
		return err
	}
	data, err := strictjson.Canonical(manifest)
	if err != nil {
		return err
	}
	if len(data) > maxSweepManifestBytes {
		return errors.New("sweep manifest exceeds 8 MiB")
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".scan-sweep-*")
	if err != nil {
		return err
	}
	defer os.Remove(temporary.Name())
	if err := temporary.Chmod(0600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if create {
		err = os.Link(temporary.Name(), path)
	} else {
		info, statErr := os.Lstat(path)
		if statErr != nil {
			return statErr
		}
		if !info.Mode().IsRegular() {
			return errors.New("sweep manifest must be a regular file")
		}
		err = os.Rename(temporary.Name(), path)
	}
	if err != nil {
		return err
	}
	directory, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func dispatchScanSweep(ctx context.Context, client *publicclient.Client, path string, manifest *scanSweepManifest) error {
	for i, member := range manifest.Snapshot.Members {
		outcome := &manifest.Outcomes[i]
		if outcome.State != "pending" && outcome.State != "unknown" {
			continue
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		response, callErr := client.API.CreateProjectRunWithBody(ctx, member.ProjectID, &publicapi.CreateProjectRunParams{IdempotencyKey: member.Key}, "application/json", bytes.NewReader(member.Request))
		created, err := sweepResponse[publicapi.CreateRunResponse](response, callErr, http.StatusAccepted)
		if err == nil && (!contracts.ValidResourceID(created.RunId) || created.ProjectId == nil || *created.ProjectId != member.ProjectID) {
			err = errors.New("invalid child Run response")
		}
		if err == nil {
			outcome.State = "created"
			outcome.RunID = created.RunId
			outcome.ErrorCode = ""
		} else {
			outcome.State = "unknown"
			outcome.ErrorCode = "create_outcome_unknown"
			var apiErr *publicclient.APIError
			if errors.As(err, &apiErr) && apiErr.Status >= 400 && !apiErr.Retryable {
				outcome.State = "failed"
				outcome.ErrorCode = apiErr.Code
				if !contracts.ValidResourceID(outcome.ErrorCode) {
					outcome.ErrorCode = "permanent_create_failure"
				}
			}
		}
		// Even cancellation or lost acknowledgement preserves the original child
		// key/request. A failed local replacement leaves the previous valid file.
		if err := saveScanSweep(path, manifest, false); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}
