package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/auditstore/querystore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/pentestcontract"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/credentials"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// This is a bounded local validation deadline, independent of target budgets.
const preflightTimeout = 30 * time.Second

// PreflightParams identifies the authenticated owner and requested draft revision.
type PreflightParams struct {
	OwnerID             string
	AuditID             string
	ExpectedRevision    uint64
	OperationsPrincipal bool
}

// PreflightResult describes local metadata compatibility, never target reachability.
type PreflightResult struct {
	AuditID                string                `json:"auditId"`
	Revision               uint64                `json:"revision"`
	BaselineInputDigest    string                `json:"baselineInputDigest"`
	CompatibleCapabilities []string              `json:"compatibleCapabilities"`
	MissingCapabilities    []string              `json:"missingCapabilities"`
	Reasons                []CompatibilityReason `json:"reasons"`
	ServerCompatible       bool                  `json:"serverCompatible"`
}

type engagementSelection struct {
	artifact           auditstore.ExactArtifact
	document           pentestcontract.Document
	inputs             []auditstore.ArtifactLink
	credentialIDs      []string
	unclassifiedInputs bool
}

// Preflight revalidates one owner-visible draft without mutating it or invoking
// DNS, target transport, credential decryption or a Runtime.
func (s *Service) Preflight(ctx context.Context, params PreflightParams) (PreflightResult, error) {
	ownerID, auditID, expectedRevision := params.OwnerID, params.AuditID, params.ExpectedRevision
	if ownerID == "" || auditID == "" || expectedRevision == 0 {
		return PreflightResult{}, ErrInvalid
	}
	ctx, cancel := context.WithTimeout(ctx, preflightTimeout)
	defer cancel()
	var result PreflightResult
	err := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{IsoLevel: pgx.RepeatableRead, AccessMode: pgx.ReadOnly}, func(tx pgx.Tx) error {
		audit, err := querystore.New(tx).Get(ctx, ownerID, auditID)
		if err != nil {
			return err
		}
		if audit.Revision != expectedRevision {
			return auditstore.ErrPrecondition
		}
		if audit.State != auditstore.AuditDraft {
			return auditstore.ErrConflict
		}
		project, err := projectstore.NewPostgresStore(tx).Get(ctx, ownerID, audit.ProjectID)
		if err != nil {
			return err
		}
		if project.Lifecycle != projectstore.LifecycleActive {
			return projectstore.ErrDeleting
		}
		profile, err := auditprofile.DecodeResolvedAuditProfileSnapshot(audit.ProfileSnapshot)
		if err != nil {
			return err
		}
		selection, err := DecodeDraftSelection(audit.InputSelection)
		if err != nil {
			return err
		}
		artifactService := artifacts.NewService(artifacts.NewPostgresRepository(tx))
		store, err := artifactService.Project(audit.ProjectID)
		if err != nil {
			return err
		}
		if _, err := verifyInputMetadata(ctx, store, profile, selection); err != nil {
			return err
		}
		selected, err := readEngagement(ctx, artifactService, audit.ProjectID, profile, selection)
		if err != nil {
			return err
		}
		if selected == nil {
			return fmt.Errorf("%w: Audit has no declared pentest engagement", ErrInvalid)
		}
		if err := validateEngagementCredentials(ctx, tx, credentials.RuntimeCredentialUser{UserID: ownerID, Operations: params.OperationsPrincipal}, selected); err != nil {
			return err
		}
		compatible, missing, reasons := engagementCompatibility(profile, project.ActiveChecks, selected)
		profileCompatibility := ProfileCompatibility(profile)
		reasonSet := map[CompatibilityReason]struct{}{}
		for _, reason := range append(reasons, profileCompatibility.Reasons...) {
			reasonSet[reason] = struct{}{}
		}
		reasons = orderedReasons(reasonSet)
		digest, err := contentdigest.CanonicalJSON(struct {
			Profile      string                       `json:"profile"`
			Selection    DraftSelection               `json:"selection"`
			Engagement   auditstore.ExactArtifact     `json:"engagement"`
			Inputs       []auditstore.ArtifactLink    `json:"inputs"`
			ActiveChecks contracts.ActiveChecksPolicy `json:"activeChecks"`
		}{audit.Profile.Digest, selection, selected.artifact, selected.inputs, project.ActiveChecks})
		if err != nil {
			return err
		}
		result = PreflightResult{AuditID: auditID, Revision: audit.Revision, BaselineInputDigest: digest, CompatibleCapabilities: compatible, MissingCapabilities: missing, Reasons: reasons, ServerCompatible: len(reasons) == 0}
		return nil
	})
	return result, err
}

func readEngagement(ctx context.Context, service *artifacts.Service, projectID string, profile auditprofile.ResolvedAuditProfile, selection DraftSelection) (*engagementSelection, error) {
	var selected *engagementSelection
	for name, exact := range selection.Inputs {
		if exact.MediaType != pentestcontract.MediaType {
			continue
		}
		slot, ok := profile.Inputs[name]
		if !ok || !slices.Contains(slot.MediaTypes, pentestcontract.MediaType) || len(slot.MediaTypes) != 1 || selected != nil {
			return nil, fmt.Errorf("%w: engagement needs exactly one explicit typed input slot", ErrInvalid)
		}
		selected = &engagementSelection{artifact: exact, inputs: []auditstore.ArtifactLink{}, credentialIDs: []string{}}
	}
	if selected == nil {
		return nil, nil
	}
	store, err := service.Project(projectID)
	if err != nil {
		return nil, err
	}
	// Check metadata before loading bytes; an oversized engagement is rejected
	// even when selected by an earlier draft.
	if selected.artifact.SizeBytes > pentestcontract.MaxDocumentBytes {
		return nil, fmt.Errorf("%w: engagement exceeds its byte limit", ErrInvalid)
	}
	read, err := store.ReadExact(ctx, selected.artifact)
	if err != nil {
		return nil, err
	}
	document, err := pentestcontract.Decode(read.Payload.Data)
	if err != nil {
		return nil, fmt.Errorf("%w: pentest engagement failed validation", ErrInvalid)
	}
	selected.document = document
	if len(document.Inputs) > auditprofile.MaxAuditProfileInputs {
		return nil, fmt.Errorf("%w: engagement inputs exceed Audit input limit", ErrInvalid)
	}
	for _, exact := range selection.Inputs {
		if exact.MediaType == pentestcontract.MediaType {
			continue
		}
		classified := false
		for _, input := range document.Inputs {
			if exact.Ref.SameExact(input.Artifact) && exact.Digest == input.ContentDigest {
				classified = true
				break
			}
		}
		selected.unclassifiedInputs = selected.unclassifiedInputs || !classified
	}
	refs := make([]contracts.ArtifactRef, 0, len(document.Inputs)+len(document.Identities))
	expected := make([]contracts.ExactArtifact, 0, cap(refs))
	keys := make([]string, 0, cap(refs))
	for _, input := range document.Inputs {
		refs = append(refs, input.Artifact)
		expected = append(expected, contracts.ExactArtifact{Ref: input.Artifact, Digest: input.ContentDigest})
		keys = append(keys, "pentest-input/"+input.ID)
	}
	for _, identity := range document.Identities {
		if identity.Credential != nil {
			selected.credentialIDs = append(selected.credentialIDs, identity.Credential.CredentialID)
		}
		if identity.LoginRecipe != nil {
			refs = append(refs, identity.LoginRecipe.Ref)
			expected = append(expected, *identity.LoginRecipe)
			keys = append(keys, "pentest-login/"+identity.ID)
		}
	}
	if len(refs) > 0 {
		metadata, err := store.DescribeExact(ctx, refs)
		if err != nil {
			return nil, err
		}
		for i, exact := range metadata {
			if expected[i].Verify(exact) != nil {
				return nil, fmt.Errorf("%w: engagement input digest differs from exact revision", ErrInvalid)
			}
			provenance, err := json.Marshal(struct {
				Schema     string                   `json:"schema"`
				Engagement auditstore.ExactArtifact `json:"engagement"`
				Input      string                   `json:"input"`
			}{"contractor.audit.pentest-input.v1", selected.artifact, keys[i]})
			if err != nil {
				return nil, err
			}
			selected.inputs = append(selected.inputs, auditstore.ArtifactLink{LogicalKey: keys[i], Artifact: exact, SourceProvenance: provenance})
		}
	}
	selected.credentialIDs = mergeIDs(selected.credentialIDs)
	return selected, nil
}

func validateEngagementCredentials(ctx context.Context, tx pgx.Tx, user credentials.RuntimeCredentialUser, selected *engagementSelection) error {
	if selected == nil {
		return nil
	}
	repository := credentials.NewRuntimeCredentialRepository(tx)
	for _, identity := range selected.document.Identities {
		if identity.Credential == nil {
			continue
		}
		if err := repository.ValidateRuntimeCredentialUse(ctx, user, identity.Credential.CredentialID, string(identity.Credential.Kind)); err != nil {
			// Wrong kind and foreign IDs share the safe missing-credential outcome.
			if errors.Is(err, credentials.ErrRuntimeCredentialNotFound) || errors.Is(err, credentials.ErrRuntimeCredentialInvalid) {
				return credentials.ErrRuntimeCredentialNotFound
			}
			return err
		}
	}
	return nil
}

func engagementCompatibility(profile auditprofile.ResolvedAuditProfile, activeChecks contracts.ActiveChecksPolicy, selected *engagementSelection) ([]string, []string, []CompatibilityReason) {
	compatible, missing, reasons := []string{}, []string{}, []CompatibilityReason{}
	if selected == nil {
		return compatible, missing, reasons
	}
	for _, capability := range append(slices.Clone(selected.document.Policy.RequiredCapabilities), selected.document.Policy.OptionalCapabilities...) {
		// Controlled anonymous preflight has a placed Runtime consumer. Other
		// execution capabilities still need their own production consumer.
		if capability == pentestcontract.Capability || capability == string(runtimesettings.RuntimeAdapterPentestTargetHTTP) {
			compatible = append(compatible, capability)
		} else {
			missing = append(missing, capability)
		}
	}
	sort.Strings(compatible)
	sort.Strings(missing)
	network := false
	for _, binding := range profile.Workflows {
		for _, stage := range binding.Workflow.Stages {
			for _, agent := range stage.Agents {
				network = network || resolved.TemplateRequiresNetwork(agent.Template)
			}
		}
	}
	if len(missing) > 0 || network {
		reasons = append(reasons, ReasonPentestCapabilityUnsupported)
	}
	if activeChecks != contracts.ActiveChecksAutomatic || profile.Interaction.ActiveChecks != contracts.ActiveChecksAutomatic {
		reasons = append(reasons, ReasonPentestActiveChecksUnsupported)
	}
	if selected.document.AccessMode == "blackbox" && !blackboxClosure(profile, selected) {
		reasons = append(reasons, ReasonPentestSourceClosureUnsupported)
	}
	return compatible, missing, reasons
}

func blackboxClosure(profile auditprofile.ResolvedAuditProfile, selected *engagementSelection) bool {
	if selected.unclassifiedInputs {
		return false
	}
	for _, input := range selected.document.Inputs {
		if input.Role == "source" {
			return false
		}
	}
	for _, binding := range profile.Workflows {
		for _, stage := range binding.Workflow.Stages {
			if stage.Context.Workspace != nil && (len(stage.Context.Workspace.Sources) > 0 || stage.Context.Workspace.State != nil) {
				return false
			}
			for _, agent := range stage.Agents {
				// Shared sitemap metadata does not grant source-file access (SMT-R14).
				// Skill packages have no trusted source-provenance classification. Until
				// they do, a selected package cannot establish blackbox isolation.
				if len(agent.Template.Skills) > 0 {
					return false
				}
				for _, toolset := range agent.Template.Toolsets {
					switch toolset.Ref.ToolsetID {
					case "code-analysis", "source-analysis", "taint-annotations", "security-findings-code":
						return false
					}
				}
			}
		}
	}
	// The declared document roles are authoritative for ordinary exact inputs;
	// opaque inherited archives cannot acquire a non-source role implicitly.
	for _, input := range selected.inputs {
		if strings.HasPrefix(input.LogicalKey, "pentest-login/") {
			continue
		}
		if strings.Contains(input.Artifact.MediaType, "zip") || strings.Contains(input.Artifact.MediaType, "tar") {
			return false
		}
	}
	return true
}

func validateEngagementBaseline(baseline BaselineSnapshot) error {
	engagement := baseline.Engagement
	if engagement == nil {
		return nil
	}
	matched := false
	for _, input := range baseline.Inputs {
		matched = matched || (input.Verify(engagement.Artifact) == nil && input.MediaType == pentestcontract.MediaType)
	}
	if !matched {
		return fmt.Errorf("%w: engagement is not an exact baseline input", ErrInvalid)
	}
	if engagement.Schema != auditbaseline.EngagementSchema || engagement.Artifact.Validate() != nil || !engagement.Artifact.HasMetadata() || engagement.Artifact.MediaType != pentestcontract.MediaType || engagement.Document.Validate() != nil || engagement.AuthorizedBy == "" || engagement.AuthorizedAt.IsZero() {
		return fmt.Errorf("%w: engagement baseline is invalid", ErrInvalid)
	}
	digest, err := engagement.Document.Policy.Digest()
	if err != nil || engagement.PolicyDigest != digest {
		return fmt.Errorf("%w: engagement policy digest differs", ErrInvalid)
	}
	for _, identity := range engagement.Document.Identities {
		if identity.Credential != nil && !slices.Contains(baseline.RuntimeCredentialIDs, identity.Credential.CredentialID) {
			return fmt.Errorf("%w: engagement credential is not held", ErrInvalid)
		}
	}
	return nil
}

func cloneEngagement(source *auditbaseline.EngagementSnapshot) *auditbaseline.EngagementSnapshot {
	if source == nil {
		return nil
	}
	result := *source
	result.Artifact = source.Artifact.Clone()
	d := &result.Document
	d.Target.Origins = slices.Clone(d.Target.Origins)
	d.Target.SeedURLs = slices.Clone(d.Target.SeedURLs)
	if d.Target.Deployment != nil {
		value := *d.Target.Deployment
		d.Target.Deployment = &value
	}
	d.Scope.Routes = cloneEngagementRoutes(d.Scope.Routes)
	d.Scope.Exclusions = cloneEngagementRoutes(d.Scope.Exclusions)
	d.Scope.ActionClasses = slices.Clone(d.Scope.ActionClasses)
	d.Scope.CallbackDestinations = slices.Clone(d.Scope.CallbackDestinations)
	d.Identities = slices.Clone(d.Identities)
	for i := range d.Identities {
		identity := &d.Identities[i]
		identity.Origins = slices.Clone(identity.Origins)
		if identity.Role != nil {
			value := *identity.Role
			identity.Role = &value
		}
		if identity.Tenant != nil {
			value := *identity.Tenant
			identity.Tenant = &value
		}
		if identity.Credential != nil {
			value := *identity.Credential
			identity.Credential = &value
		}
		if identity.LoginRecipe != nil {
			value := identity.LoginRecipe.Clone()
			identity.LoginRecipe = &value
		}
	}
	d.Inputs = slices.Clone(d.Inputs)
	for i := range d.Inputs {
		d.Inputs[i].Artifact = d.Inputs[i].Artifact.Clone()
	}
	d.Policy.RequiredCapabilities = slices.Clone(d.Policy.RequiredCapabilities)
	d.Policy.OptionalCapabilities = slices.Clone(d.Policy.OptionalCapabilities)
	d.TestData = slices.Clone(d.TestData)
	for i := range d.TestData {
		d.TestData[i].IdentityIDs = slices.Clone(d.TestData[i].IdentityIDs)
	}
	return &result
}

func cloneEngagementRoutes(source []pentestcontract.Route) []pentestcontract.Route {
	result := slices.Clone(source)
	for i := range result {
		route := &result[i]
		route.PathPrefixes = slices.Clone(route.PathPrefixes)
		route.Methods = slices.Clone(route.Methods)
		route.ActionClasses = slices.Clone(route.ActionClasses)
		route.AddressRanges = slices.Clone(route.AddressRanges)
	}
	return result
}
