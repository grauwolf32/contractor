package auditstandards

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
)

func TestPackageDirectoryIsCanonicalAndStrict(t *testing.T) {
	first := t.TempDir()
	second := t.TempDir()
	document := validDocument("example-standard", "1.0")
	writeDocument(t, first, document, true)
	writeDocument(t, second, document, false)

	left, leftPackage, err := PackageDirectory(first)
	if err != nil {
		t.Fatal(err)
	}
	right, rightPackage, err := PackageDirectory(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(left, right) || leftPackage.Digest != rightPackage.Digest {
		t.Fatal("equivalent source JSON did not produce one canonical package")
	}
	if _, err := Validate(left, Reference{Scheme: "other", Version: "1.0"}); ErrorCode(err) != CodeIdentityMismatch {
		t.Fatalf("identity mismatch error = %v", err)
	}

	unknown := append([]byte(nil), mustJSON(t, document)...)
	unknown = bytes.Replace(unknown, []byte(`"schema":`), []byte(`"unknown":true,"schema":`), 1)
	writeRaw(t, filepath.Join(first, ManifestPath), unknown)
	if _, _, err := PackageDirectory(first); ErrorCode(err) != CodeManifestInvalid {
		t.Fatalf("unknown field error = %v", err)
	}

	pathAttack := rawArchive(t, "../standard.json", mustJSON(t, document))
	if _, err := Validate(pathAttack, Reference{}); ErrorCode(err) != CodePathInvalid {
		t.Fatalf("path attack error = %v", err)
	}

	outside := filepath.Join(t.TempDir(), "outside.json")
	writeRaw(t, outside, mustJSON(t, document))
	linked := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(linked, ManifestPath)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := PackageDirectory(linked); ErrorCode(err) != CodeMemberForbidden {
		t.Fatalf("source symlink error = %v", err)
	}
}

func TestValidationRejectsInvalidLicenseDuplicatesAndDanglingMappings(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Document)
		code string
	}{
		{name: "license", edit: func(value *Document) { value.Standard.License.ID = "unknown" }, code: CodeLicenseInvalid},
		{name: "duplicate entry", edit: func(value *Document) { value.Entries = append(value.Entries, value.Entries[0]) }, code: CodeManifestInvalid},
		{name: "dangling entry", edit: func(value *Document) { value.Mappings[0].EntryIDs[0] = "missing" }, code: CodeDanglingMapping},
		{name: "dangling contract", edit: func(value *Document) { value.Entries[0].EvidenceContract.ID = "missing" }, code: CodeDanglingMapping},
		{name: "versionless contract", edit: func(value *Document) { value.EvidenceContracts[0].Version = "" }, code: CodeManifestInvalid},
		{name: "too many mappings", edit: func(value *Document) { value.Mappings = make([]Mapping, MaximumMappings+1) }, code: CodeLimitExceeded},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			document := validDocument("example-standard", "1")
			test.edit(&document)
			root := t.TempDir()
			writeDocument(t, root, document, false)
			_, _, err := PackageDirectory(root)
			if ErrorCode(err) != test.code {
				t.Fatalf("error = %v (%q), want %q", err, ErrorCode(err), test.code)
			}
		})
	}
}

func TestDiscoveryIsAllOrNothingAndRejectsDuplicateIdentity(t *testing.T) {
	root := t.TempDir()
	if plan, err := DiscoverBundled(root); err != nil || len(plan.Packages()) != 0 {
		t.Fatalf("missing catalog = (%+v, %v)", plan, err)
	}
	if err := os.MkdirAll(filepath.Join(root, CatalogNamespace), 0o755); err != nil {
		t.Fatal(err)
	}
	writeRaw(t, filepath.Join(root, CatalogNamespace, ".gitkeep"), []byte("\n"))
	if plan, err := DiscoverBundled(root); err != nil || len(plan.Packages()) != 0 {
		t.Fatalf("empty catalog marker = (%+v, %v)", plan, err)
	}
	writePackageSource(t, root, "one", validDocument("example", "1"))
	writePackageSource(t, root, "two", validDocument("example", "1"))
	if plan, err := DiscoverBundled(root); ErrorCode(err) != CodeIdentityMismatch || plan != nil {
		t.Fatalf("duplicate catalog = (%+v, %v)", plan, err)
	}
	if _, err := os.Stat(filepath.Join(root, CatalogNamespace, "one", ManifestPath)); err != nil {
		t.Fatalf("discovery mutated source tree: %v", err)
	}
}

func TestCatalogCreateOnlySeedResolvePinAndDisclosure(t *testing.T) {
	root := t.TempDir()
	document := validDocument("example", "1")
	writePackageSource(t, root, "example-v1", document)
	plan, err := DiscoverBundled(root)
	if err != nil {
		t.Fatal(err)
	}
	repository := newMemoryRepository()
	service := artifacts.NewService(repository)
	catalog, _ := NewCatalog(service)

	first, err := catalog.Initialize(context.Background(), "owner", plan)
	if err != nil || len(first) != 1 || first[0].Status != SeedCreated {
		t.Fatalf("first seed = (%+v, %v)", first, err)
	}
	second, err := catalog.Initialize(context.Background(), "owner", plan)
	if err != nil || len(second) != 1 || second[0].Status != SeedInSync || repository.writes != 1 {
		t.Fatalf("second seed = (%+v, %v), writes=%d", second, err, repository.writes)
	}
	ref := Reference{Scheme: "example", Version: "1"}
	resolved, err := catalog.Resolve(context.Background(), "owner", ref)
	if err != nil || resolved.Source.Artifact.Revision == nil {
		t.Fatalf("resolve = (%+v, %v)", resolved, err)
	}
	listed, err := catalog.List(context.Background(), "owner")
	if err != nil || len(listed) != 1 || listed[0].Package.Digest != resolved.Package.Digest {
		t.Fatalf("list = (%+v, %v)", listed, err)
	}
	pinned, err := catalog.Pin(context.Background(), "owner", "project", "audit-fixed", []Reference{ref})
	if err != nil || len(pinned) != 1 || pinned[0].Retained.Artifact.Revision == nil ||
		pinned[0].Retained.Digest != pinned[0].Catalog.Digest {
		t.Fatalf("pin = (%+v, %v)", pinned, err)
	}
	project, _ := service.Project("project")
	retained, err := project.Read(context.Background(), pinned[0].Retained.Artifact)
	if err != nil || !bytes.Equal(retained.Payload.Data, resolved.Package.Payload()) {
		t.Fatalf("retained package = (%+v, %v)", retained, err)
	}

	full := Projection(resolved, true)
	if len(full.Entries) != 1 || len(full.EvidenceContracts) != 1 || len(full.Mappings) != 1 {
		t.Fatalf("full projection = %+v", full)
	}
	resolved.Package.Document.Standard.License = License{
		ID: "LicenseRef-Proprietary", URL: "https://example.invalid/license",
		Attribution: "Example owner", Disclosure: DisclosureMetadata,
	}
	metadataOnly := Projection(resolved, true)
	if metadataOnly.Entries != nil || metadataOnly.EvidenceContracts != nil || metadataOnly.Mappings != nil {
		t.Fatalf("metadata-only projection exposed package body: %+v", metadataOnly)
	}

	user, _ := service.User("owner")
	if _, err := user.Write(context.Background(), artifacts.ArtifactRef{
		Namespace: CatalogNamespace, Name: "forbidden",
	}, artifacts.Payload{MediaType: "application/json", Data: []byte("{}")}, nil); !errors.Is(err, artifacts.ErrReservedNamespace) {
		t.Fatalf("generic catalog write error = %v", err)
	}
}

func TestCatalogRejectsDriftWithoutAdvancingBinding(t *testing.T) {
	root := t.TempDir()
	writePackageSource(t, root, "example-v1", validDocument("example", "1"))
	plan, _ := DiscoverBundled(root)
	repository := newMemoryRepository()
	service := artifacts.NewService(repository)
	catalog, _ := NewCatalog(service)
	if _, err := catalog.Initialize(context.Background(), "owner", plan); err != nil {
		t.Fatal(err)
	}
	ref := artifacts.ArtifactRef{Namespace: CatalogNamespace, Name: ArtifactName(Reference{Scheme: "example", Version: "1"})}
	current, _ := service.User("owner")
	read, _ := current.Read(context.Background(), ref)
	if _, err := service.WriteAuditStandardPackage(context.Background(), "owner", ref,
		artifacts.Payload{MediaType: "application/zip", Data: []byte("tampered")}, read.Ref.Revision); err != nil {
		t.Fatal(err)
	}
	writes := repository.writes
	if _, err := catalog.Initialize(context.Background(), "owner", plan); !errors.Is(err, ErrDrift) {
		t.Fatalf("drift error = %v", err)
	}
	if repository.writes != writes {
		t.Fatalf("drift initialization advanced binding: %d -> %d", writes, repository.writes)
	}
}

func validDocument(scheme, version string) Document {
	contract := EvidenceContractRef{ID: "source-evidence", Version: "1"}
	return Document{
		Schema: Schema,
		Standard: Metadata{
			Scheme: scheme, Version: version, Title: "Example standard",
			Description: "A bounded example standard.",
			Source:      Source{Name: "Example source", URL: "https://example.invalid/standard", Revision: version},
			License: License{ID: "CC-BY-4.0", URL: "https://creativecommons.org/licenses/by/4.0/",
				Attribution: "Example authors", Disclosure: DisclosureFull},
		},
		EvidenceContracts: []EvidenceContract{{
			ID: contract.ID, Version: contract.Version,
			Assessments:     []string{"violated", "satisfied", "inconclusive"},
			EvidenceKinds:   []string{"artifact", "observation"},
			MinimumEvidence: 1, MaximumEvidence: 8, HumanReview: "never", RationaleRequired: true,
		}},
		Entries: []Entry{{
			ID: "EX-1", Kind: "requirement", Title: "Example requirement",
			Statement: "The example must be reviewed.", Applicability: Applicability{Mode: "always"},
			AllowedMethods: []string{"source-analysis"}, EvidenceContract: contract,
		}},
		Mappings: []Mapping{{
			Key: "check-example", EntryIDs: []string{"EX-1"}, WorkflowRole: "check",
			Method: "source-analysis", EvidenceContract: contract,
			Title: "Review example", Objective: "Determine whether the example requirement is satisfied.",
		}},
	}
}

func writePackageSource(t *testing.T, root, directory string, document Document) {
	t.Helper()
	path := filepath.Join(root, CatalogNamespace, directory)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	writeDocument(t, path, document, true)
}

func writeDocument(t *testing.T, root string, document Document, indented bool) {
	t.Helper()
	var data []byte
	var err error
	if indented {
		data, err = json.MarshalIndent(document, "", "  ")
	} else {
		data, err = json.Marshal(document)
	}
	if err != nil {
		t.Fatal(err)
	}
	writeRaw(t, filepath.Join(root, ManifestPath), data)
}

func writeRaw(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func rawArchive(t *testing.T, name string, data []byte) []byte {
	t.Helper()
	var buffer bytes.Buffer
	writer := zip.NewWriter(&buffer)
	entry, err := writer.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := entry.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}

type memoryRepository struct {
	mu       sync.Mutex
	bindings map[string]artifacts.ReadResult
	writes   int
}

func newMemoryRepository() *memoryRepository {
	return &memoryRepository{bindings: map[string]artifacts.ReadResult{}}
}

func (r *memoryRepository) Write(_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef,
	payload artifacts.Payload, expected *string) (artifacts.WriteResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := artifactKey(scope, ref)
	current, exists := r.bindings[key]
	if expected == nil && exists || expected != nil && (!exists || current.Ref.Revision == nil || *current.Ref.Revision != *expected) {
		return artifacts.WriteResult{}, artifacts.ErrArtifactConflict
	}
	r.writes++
	revision := fmt.Sprintf("revision-%d", r.writes)
	now := time.Unix(int64(r.writes), 0).UTC()
	exact := ref
	exact.Revision = &revision
	r.bindings[key] = artifacts.ReadResult{Ref: exact,
		Payload:          artifacts.Payload{MediaType: payload.MediaType, Data: append([]byte(nil), payload.Data...)},
		BindingCreatedAt: now, RevisionCreatedAt: now}
	return artifacts.WriteResult{Ref: exact, MediaType: payload.MediaType, Size: int64(len(payload.Data)),
		BindingCreatedAt: now, RevisionCreatedAt: now}, nil
}

func (r *memoryRepository) WriteAuditArtifact(ctx context.Context, scope artifacts.Scope,
	ref artifacts.ArtifactRef, payload artifacts.Payload) (artifacts.WriteResult, error) {
	return r.Write(ctx, scope, ref, payload, nil)
}

func (r *memoryRepository) Read(_ context.Context, scope artifacts.Scope, ref artifacts.ArtifactRef) (artifacts.ReadResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	result, exists := r.bindings[artifactKey(scope, ref)]
	if !exists || ref.Revision != nil && (result.Ref.Revision == nil || *ref.Revision != *result.Ref.Revision) {
		return artifacts.ReadResult{}, artifacts.ErrArtifactNotFound
	}
	result.Payload.Data = append([]byte(nil), result.Payload.Data...)
	return result, nil
}

func (r *memoryRepository) List(_ context.Context, scope artifacts.Scope, namespace *string) ([]artifacts.ArtifactRef, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	prefix := string(scope.Kind()) + "/" + scope.ID() + "/"
	result := make([]artifacts.ArtifactRef, 0)
	for key, value := range r.bindings {
		if len(key) < len(prefix) || key[:len(prefix)] != prefix || namespace != nil && value.Ref.Namespace != *namespace {
			continue
		}
		result = append(result, value.Ref)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].Namespace == result[j].Namespace {
			return result[i].Name < result[j].Name
		}
		return result[i].Namespace < result[j].Namespace
	})
	return result, nil
}

func (r *memoryRepository) ForkInput(context.Context, artifacts.Scope, artifacts.ArtifactRef,
	artifacts.Scope, string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, artifacts.ErrQueryUnsupported
}

func (r *memoryRepository) BindOutputExact(context.Context, artifacts.Scope, string,
	artifacts.ArtifactRef, *string) (artifacts.ForkResult, error) {
	return artifacts.ForkResult{}, artifacts.ErrQueryUnsupported
}

func (r *memoryRepository) PinExact(context.Context, string, artifacts.Scope,
	artifacts.ArtifactRef, artifacts.PinKind, string) error {
	return artifacts.ErrQueryUnsupported
}

func (r *memoryRepository) FreezeOutputs(context.Context, artifacts.Scope) error {
	return artifacts.ErrQueryUnsupported
}

func artifactKey(scope artifacts.Scope, ref artifacts.ArtifactRef) string {
	return string(scope.Kind()) + "/" + scope.ID() + "/" + ref.Namespace + "/" + ref.Name
}

func TestCanonicalNormalizationSortsSemanticSets(t *testing.T) {
	document := validDocument("example", "1")
	document.EvidenceContracts[0].Assessments = []string{"violated", "inconclusive", "satisfied"}
	document.EvidenceContracts[0].EvidenceKinds = []string{"observation", "artifact"}
	root := t.TempDir()
	writeDocument(t, root, document, false)
	_, pkg, err := PackageDirectory(root)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pkg.Document.EvidenceContracts[0].Assessments,
		[]string{"inconclusive", "satisfied", "violated"}) {
		t.Fatalf("normalized assessments = %v", pkg.Document.EvidenceContracts[0].Assessments)
	}
}
