package auditservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/ownerstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig/runtimeconfigtest"
	"github.com/jackc/pgx/v5"
)

func TestAuditRulepackMetadataAdmissionPostgres(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, dsn)
	catalog := &switchableProfileCatalog{snapshot: aliasedRulepackCatalog(t), available: true}
	service, err := New(Options{Pool: pool, Profiles: catalog,
		TransactionLLMCredentials: runtimeconfigtest.TransactionLLMCredentialLookupFactoryFunc(func(pgx.Tx) (resolved.CredentialLookup, error) {
			return &switchableCredentialLookup{available: true}, nil
		}), CredentialGuard: &countingCredentialGuard{}})
	if err != nil {
		t.Fatal(err)
	}
	profile, err := catalog.AuditProfile("aliased-web-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name, media string
		size        int
		invalid     bool
	}{
		{"yaml-limit", "application/yaml", scan.MaxNucleiRulepackFileBytes, false},
		{"yaml-over", "application/yaml", scan.MaxNucleiRulepackFileBytes + 1, true},
		{"zip-limit", scan.NucleiRulepackMediaType, scan.MaxNucleiRulepackZIPBytes, false},
		{"zip-over", scan.NucleiRulepackMediaType, scan.MaxNucleiRulepackZIPBytes + 1, true},
		{"wrong-media", "application/json", 8, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			project, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{
				ProjectID: "project-" + test.name, OwnerID: "owner-rulepack", Kind: projectstore.KindProject, Name: "Rulepack admission",
				IdempotencyKey: test.name, RequestDigest: serviceTestDigest(test.name)})
			if err != nil {
				t.Fatal(err)
			}
			store, err := artifacts.NewService(artifacts.NewPostgresRepository(pool)).Project(project.ProjectID)
			if err != nil {
				t.Fatal(err)
			}
			refs := map[string]contracts.ArtifactRef{}
			exacts := map[string]contracts.ExactArtifact{}
			write := func(name, media string, data []byte) {
				t.Helper()
				written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, artifacts.Payload{MediaType: media, Data: data}, nil)
				if err != nil {
					t.Fatal(err)
				}
				metadata, err := store.Metadata(ctx, written.Ref)
				if err != nil {
					t.Fatal(err)
				}
				refs[name], exacts[name] = written.Ref, metadata.Exact()
			}
			write("target", "application/json", []byte(`{"schema":"contractor.audit.web-target.v1","url":"https://target.test/"}`))
			write("settings", "application/json", []byte(`{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"","tags":"","severity":"","rateLimit":10}`))
			// These are opaque payloads: this boundary owns metadata admission;
			// Runtime separately validates YAML and ZIP content.
			write("selected_rules", test.media, bytes.Repeat([]byte("x"), test.size))
			draft, _, err := service.CreateDraft(ctx, CreateDraftParams{AuditID: "audit-" + test.name, OwnerID: project.OwnerID, ProjectID: project.ProjectID,
				Profile: ProfileSelector{Name: "aliased-web-scan", Version: "1"}, Inputs: refs,
				IdempotencyKey: "create-" + test.name, RequestDigest: serviceTestDigest("create-" + test.name)})
			if test.invalid {
				if !errors.Is(err, ErrInvalid) {
					t.Errorf("CreateDraft = %v, want invalid rulepack", err)
				}
				// Persist a draft as accepted by the old service. Start must remain
				// authoritative even when CreateDraft was bypassed or ran earlier.
				draft = storedRulepackDraft(t, ctx, service, profile, project.ProjectID, project.OwnerID, "stored-"+test.name, exacts)
			} else if err != nil {
				t.Fatal(err)
			}
			params := StartParams{OwnerID: project.OwnerID, AuditID: draft.AuditID, ExpectedRevision: draft.Revision,
				IdempotencyKey: "start-" + test.name, RequestDigest: serviceTestDigest("start-" + test.name)}
			started, err := service.Start(ctx, params)
			if test.invalid {
				if !errors.Is(err, ErrInvalid) {
					t.Errorf("Start = %v, want invalid rulepack", err)
				}
				current, err := service.Get(ctx, project.OwnerID, draft.AuditID)
				if err != nil || current.State != auditstore.AuditDraft || current.Revision != draft.Revision || len(current.BaselineSnapshot) != 0 {
					t.Errorf("refused draft changed: state=%s revision=%d baseline=%s error=%v", current.State, current.Revision, current.BaselineSnapshot, err)
				}
				var count int
				if err := pool.QueryRow(ctx, `SELECT count(*) FROM workflow_runs WHERE project_id = $1`, project.ProjectID).Scan(&count); err != nil || count != 0 {
					t.Fatalf("child Runs=%d error=%v", count, err)
				}
				return
			}
			if err != nil || started.Audit.State != auditstore.AuditActive {
				t.Fatalf("Start = %v", err)
			}
			// Advance current content beyond the limit and make the catalog
			// unavailable: response-loss replay must use the accepted receipt.
			pack := exacts["selected_rules"]
			if _, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: "selected_rules"}, artifacts.Payload{MediaType: test.media, Data: bytes.Repeat([]byte("x"), test.size+1)}, pack.Ref.Revision); err != nil {
				t.Fatal(err)
			}
			catalog.setAvailable(false)
			replay, err := service.Start(ctx, params)
			catalog.setAvailable(true)
			if err != nil || !replay.Replayed || replay.Audit.AuditID != started.Audit.AuditID {
				t.Fatalf("replay = %+v, %v", replay, err)
			}
			baseline, err := DecodeBaseline(replay.Audit.BaselineSnapshot)
			if err != nil || baseline.Inputs["selected_rules"].Verify(pack) != nil {
				t.Fatalf("replay changed exact input: %v", err)
			}
		})
	}
}

func storedRulepackDraft(t *testing.T, ctx context.Context, service *Service, profile auditprofile.ResolvedAuditProfile, projectID, ownerID, auditID string, inputs map[string]contracts.ExactArtifact) auditstore.Audit {
	t.Helper()
	snapshot, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	selection, err := encodeDraftSelection(DraftSelection{Schema: DraftSelectionSchema, Inputs: inputs, RuntimeLabels: []string{}, Labels: map[string]string{}, RequiredPlacementLabels: []string{}})
	if err != nil {
		t.Fatal(err)
	}
	draft, _, err := ownerstore.New(service.pool).CreateDraft(ctx, auditstore.CreateDraftParams{
		AuditID: auditID, OwnerID: ownerID, ProjectID: projectID,
		Profile:         auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest},
		ProfileSnapshot: snapshot, InputSelection: selection,
		Limits:         auditstore.Limits{MaxRounds: 1, BatchSize: 1, MaxItemsPerRound: 1, MaxItemsTotal: 1, MaxSubmittedRuns: 3, MaxItemRunAttempts: 3, MaxEvidenceBytes: 67108864},
		IdempotencyKey: auditID, RequestDigest: serviceTestDigest(auditID),
	})
	if err != nil {
		t.Fatal(err)
	}
	return draft
}

// Exercise the real catalog with different Audit, role and Workflow input names.
// The closed scan executor still uses its declared rulepack context alias.
func aliasedRulepackCatalog(t *testing.T) *config.Snapshot {
	t.Helper()
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS(auditServiceCatalogFixture)); err != nil {
		t.Fatal(err)
	}
	edits := map[string][][2]string{
		"audit-profiles/web_scan.yaml": {
			{"name: web-scan,", "name: aliased-web-scan,"},
			{"itemWorkflowRole: scan", "itemWorkflowRole: arbitrary-role"},
			{"    scan:\n", "    arbitrary-role:\n"},
			{"    rulepack: {required:", "    selected_rules: {required:"},
			{"        rulepack: {source: audit-input, name: rulepack}", "        workflow_rules: {source: audit-input, name: selected_rules}"},
		},
		"workflows/audit_web_nuclei_scan.yaml": {
			{"    rulepack: {required:", "    workflow_rules: {required:"},
			{"rulepack: {namespace: inputs, name: rulepack,", "rulepack: {namespace: inputs, name: workflow_rules,"},
		},
	}
	for path, replacements := range edits {
		full := filepath.Join(root, path)
		data, err := os.ReadFile(full)
		if err != nil {
			t.Fatal(err)
		}
		value := string(data)
		for _, replacement := range replacements {
			if !strings.Contains(value, replacement[0]) {
				t.Fatal("fixture replacement absent:", replacement[0])
			}
			value = strings.ReplaceAll(value, replacement[0], replacement[1])
		}
		if err := os.WriteFile(full, []byte(value), 0600); err != nil {
			t.Fatal(err)
		}
	}
	catalog, _, err := configload.LoadUnionReadOnly(root, filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	return catalog
}

type rulepackReadProbe struct {
	artifacts.Repository
	reads       int
	describeErr error
	readErr     error
}

func (p *rulepackReadProbe) Read(ctx context.Context, scope artifacts.Scope, ref contracts.ArtifactRef) (artifacts.ReadResult, error) {
	p.reads++
	if p.readErr != nil {
		return artifacts.ReadResult{}, p.readErr
	}
	return p.Repository.Read(ctx, scope, ref)
}
func (p *rulepackReadProbe) DescribeExact(ctx context.Context, scope artifacts.Scope, refs []contracts.ArtifactRef) ([]contracts.ExactArtifact, error) {
	if p.describeErr != nil {
		return nil, p.describeErr
	}
	return p.Repository.DescribeExact(ctx, scope, refs)
}

func TestAuditRulepackMetadataPrecedesPayloadReadsPostgres(t *testing.T) {
	dsn := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	pool := isolatedAuditServicePool(t, ctx, dsn)
	_, _, err := projectstore.NewPostgresStore(pool).Create(ctx, projectstore.CreateParams{ProjectID: "metadata-project", OwnerID: "owner-metadata", Kind: projectstore.KindProject, Name: "Metadata admission", IdempotencyKey: "metadata-project", RequestDigest: serviceTestDigest("metadata-project")})
	if err != nil {
		t.Fatal(err)
	}
	repo := artifacts.NewPostgresRepository(pool)
	store, _ := artifacts.NewService(repo).Project("metadata-project")
	profile, err := aliasedRulepackCatalog(t).AuditProfile("aliased-web-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string]contracts.ExactArtifact{}
	payloads := map[string]artifacts.ReadResult{}
	for name, payload := range map[string]artifacts.Payload{
		"target":         {MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.web-target.v1","url":"https://target.test/"}`)},
		"settings":       {MediaType: "application/json", Data: []byte(`{"schema":"contractor.audit.web-scan-settings.v1","templateIds":"","tags":"","severity":"","rateLimit":10}`)},
		"selected_rules": {MediaType: "application/yaml", Data: bytes.Repeat([]byte("x"), scan.MaxNucleiRulepackFileBytes+1)},
	} {
		written, err := store.Write(ctx, contracts.ArtifactRef{Namespace: "inputs", Name: name}, payload, nil)
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = contracts.ExactContent(written.Ref, payload.MediaType, payload.Data)
		payloads[name] = artifacts.ReadResult{Ref: written.Ref, Payload: payload}
	}
	if err := ValidateInputPreview(profile, Scope{}, payloads, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("preview accepted aliased oversize: %v", err)
	}
	for _, test := range []struct {
		name                       string
		edit                       func(map[string]contracts.ExactArtifact)
		describeErr, readErr, want error
		reads                      int
	}{
		{name: "oversized", want: ErrInvalid},
		{name: "forged-size", edit: func(in map[string]contracts.ExactArtifact) {
			v := in["selected_rules"]
			v.SizeBytes = 1
			in["selected_rules"] = v
		}, want: ErrInvalid},
		{name: "unavailable-exact", edit: func(in map[string]contracts.ExactArtifact) {
			v := in["target"]
			revision := "absent-revision"
			v.Ref.Revision = &revision
			in["target"] = v
		}, want: artifacts.ErrArtifactNotFound},
		{name: "cancelled-metadata", describeErr: context.Canceled, want: context.Canceled},
		{name: "failed-metadata", describeErr: artifacts.ErrArtifactNotFound, want: artifacts.ErrArtifactNotFound},
		{name: "cancelled-payload", edit: func(in map[string]contracts.ExactArtifact) { delete(in, "selected_rules") }, readErr: context.Canceled, want: context.Canceled, reads: 1},
		{name: "failed-payload", edit: func(in map[string]contracts.ExactArtifact) { delete(in, "selected_rules") }, readErr: artifacts.ErrArtifactNotFound, want: artifacts.ErrArtifactNotFound, reads: 1},
		{name: "installed-mode", edit: func(in map[string]contracts.ExactArtifact) { delete(in, "selected_rules") }, reads: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			selected := maps.Clone(inputs)
			if test.edit != nil {
				test.edit(selected)
			}
			probe := &rulepackReadProbe{Repository: repo, describeErr: test.describeErr, readErr: test.readErr}
			_, err := readAndVerifyInputs(ctx, artifacts.NewService(probe), "metadata-project", profile, DraftSelection{Inputs: selected})
			if !errors.Is(err, test.want) || probe.reads != test.reads {
				t.Fatalf("read admission error=%v reads=%d, want %v/%d", err, probe.reads, test.want, test.reads)
			}
		})
	}
	// Rejections never delete or mutate independently retained source materials.
	for name, exact := range inputs {
		metadata, err := store.Metadata(ctx, exact.Ref)
		if err != nil || exact.Verify(metadata.Exact()) != nil {
			t.Fatalf("input %s changed: %v", name, err)
		}
	}
}
