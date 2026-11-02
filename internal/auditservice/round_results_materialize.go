package auditservice

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifactpolicy"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runservice"
	"github.com/grauwolf32/contractor/internal/runstore"
)

const roundResultsMaterializationSchema = "contractor.audit.round-results-materialization.v1"

type roundResultsCheckpoint struct {
	Schema         string                            `json:"schema" strictjson:"required"`
	NextPage       int                               `json:"nextPage" strictjson:"required"`
	NextDependency int                               `json:"nextDependency" strictjson:"required"`
	Revisions      []string                          `json:"revisions" strictjson:"required"`
	MetadataBytes  int64                             `json:"metadataBytes" strictjson:"required"`
	Pages          []auditdomain.RoundResultsPageRef `json:"pages" strictjson:"required"`
}

// RoundResultsInputMaterializer constructs the Audit-specific, exact-page
// initializer. Generic Run creation only controls its durable initialization
// fence; this function alone interprets the accepted snapshot contracts.
func RoundResultsInputMaterializer(roots map[string]auditstore.ExactArtifact) (string, runservice.AuditInputMaterializer, error) {
	if len(roots) != 1 {
		return "", nil, fmt.Errorf("%w: exactly one round-results mapping is required", ErrInvalid)
	}
	var slot string
	var root auditstore.ExactArtifact
	for name, value := range roots {
		if contracts.ValidateArtifactName(name) != nil || value.Validate() != nil || !value.HasMetadata() {
			return "", nil, fmt.Errorf("%w: invalid round-results input", ErrInvalid)
		}
		slot, root = name, value.Clone()
	}
	identity, err := contentdigest.JSON(struct {
		Schema string                  `json:"schema"`
		Slot   string                  `json:"slot"`
		Source contracts.ExactArtifact `json:"source"`
	}{roundResultsMaterializationSchema, slot, root})
	if err != nil {
		return "", nil, err
	}
	return identity, func(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, run runstore.WorkflowRun, state runstore.AuditInputMaterialization) (runstore.AuditInputMaterialization, error) {
		return materializeRoundResultsSlice(ctx, service, intent, run, state, slot, root, identity)
	}, nil
}

func materializeRoundResultsSlice(ctx context.Context, service *artifacts.Service, intent auditstore.RunCreationIntent, run runstore.WorkflowRun, state runstore.AuditInputMaterialization, slot string, root contracts.ExactArtifact, identity string) (runstore.AuditInputMaterialization, error) {
	if state.Identity != identity || intent.Execution.Role != auditstore.ExecutionAssessment || intent.Execution.RoundID == nil ||
		run.AuditExecutionID == nil || *run.AuditExecutionID != intent.Execution.ExecutionID ||
		root.Ref.Namespace != auditdomain.ArtifactNamespace(intent.Execution.AuditID) {
		return state, fmt.Errorf("%w: round-results materialization authority is invalid", runservice.ErrInvalid)
	}
	project, err := service.Project(intent.ProjectID)
	if err != nil {
		return state, err
	}
	rootBytes, err := project.ReadExact(ctx, root)
	if err != nil {
		return state, err
	}
	manifest, err := auditdomain.DecodeRoundResultsManifest(rootBytes.Payload.Data)
	if err != nil || manifest.AuditID != intent.Execution.AuditID || manifest.RoundID != *intent.Execution.RoundID {
		return state, fmt.Errorf("%w: accepted round-results root does not match its authority", runservice.ErrInvalid)
	}
	checkpoint := roundResultsCheckpoint{Schema: roundResultsMaterializationSchema, Pages: []auditdomain.RoundResultsPageRef{}, Revisions: []string{}, MetadataBytes: manifest.MetadataBytes + int64(len(rootBytes.Payload.Data))}
	if len(state.Checkpoint) != 0 && string(state.Checkpoint) != "{}" {
		if strictjson.Decode(state.Checkpoint, &checkpoint) != nil || checkpoint.Schema != roundResultsMaterializationSchema ||
			checkpoint.Pages == nil || checkpoint.NextPage != len(checkpoint.Pages) || checkpoint.NextPage > len(manifest.Pages) || checkpoint.NextDependency < 0 ||
			checkpoint.Revisions == nil || len(checkpoint.Revisions) != checkpoint.NextDependency {
			return state, fmt.Errorf("%w: invalid round-results materialization progress", runservice.ErrInvalid)
		}
	}
	projectScope, _ := artifacts.ProjectScope(intent.ProjectID)
	runScope, _ := artifacts.RunScope(run.RunID)
	prefix := "round-results-" + strings.TrimPrefix(identity, "sha256:") + "-"
	metadataBytes := manifest.MetadataBytes + int64(len(rootBytes.Payload.Data))
	heldRefs := make(map[string]bool, len(state.Refs))
	for _, ref := range state.Refs {
		heldRefs[ref.Key()] = true
	}
	for index, page := range checkpoint.Pages {
		if page.Ordinal != index || page.ItemCount != manifest.Pages[index].ItemCount ||
			page.Artifact.Validate() != nil || page.Artifact.MediaType != "application/json" || page.Artifact.SizeBytes > auditdomain.MaximumRoundResultsDocumentBytes ||
			page.Artifact.Ref.Namespace != artifactpolicy.AuditContextNamespace || page.Artifact.Ref.Name != fmt.Sprintf("%spage-%03d", prefix, index) {
			return state, fmt.Errorf("%w: recovered round-results page is invalid", runservice.ErrInvalid)
		}
		if !heldRefs[page.Artifact.Ref.Key()] {
			return state, fmt.Errorf("%w: recovered round-results page has no closure hold", runservice.ErrInvalid)
		}
		metadataBytes += page.Artifact.SizeBytes
	}
	if checkpoint.MetadataBytes != metadataBytes || metadataBytes > auditdomain.MaximumRoundResultsBytes {
		return state, fmt.Errorf("%w: recovered round-results metadata budget is invalid", runservice.ErrInvalid)
	}
	if checkpoint.NextPage < len(manifest.Pages) {
		pageRef := manifest.Pages[checkpoint.NextPage]
		if pageRef.Artifact.Ref.Namespace != root.Ref.Namespace {
			return state, fmt.Errorf("%w: round-results page is outside its Audit", runservice.ErrInvalid)
		}
		pageBytes, err := project.ReadExact(ctx, pageRef.Artifact)
		if err != nil {
			return state, err
		}
		page, err := auditdomain.DecodeRoundResultsPage(pageBytes.Payload.Data)
		if err != nil || page.AuditID != manifest.AuditID || page.RoundID != manifest.RoundID ||
			page.SourceGeneration != manifest.SourceGeneration || page.Ordinal != checkpoint.NextPage || len(page.Items) != pageRef.ItemCount || len(page.Mappings) != 0 {
			return state, fmt.Errorf("%w: round-results page identity is inconsistent", runservice.ErrInvalid)
		}
		dependencies, err := roundResultsRequiredDependencies(page, root.Ref.Namespace)
		if err != nil || checkpoint.NextDependency > len(dependencies) {
			return state, fmt.Errorf("%w: invalid round-results dependency closure", runservice.ErrInvalid)
		}
		projectedDescriptors := len(page.Items) + 2*len(dependencies)
		for _, item := range page.Items {
			projectedDescriptors += len(item.Attachments) + len(item.Findings)
			if item.Result != nil {
				projectedDescriptors++
			}
			if item.SelectedAttempt != nil {
				projectedDescriptors++
			}
		}
		if projectedDescriptors > auditdomain.MaximumRoundResultsSliceDescriptors {
			return state, fmt.Errorf("%w: projected round-results page exceeds its descriptor budget", runservice.ErrInvalid)
		}
		stateBytes, err := json.Marshal(state)
		if err != nil {
			return state, err
		}
		workBytes := len(rootBytes.Payload.Data) + len(pageBytes.Payload.Data) + len(stateBytes)
		var payloadBytes int64
		processed := 0
		for checkpoint.NextDependency < len(dependencies) && processed < 32 {
			source := dependencies[checkpoint.NextDependency]
			metadata, _ := json.Marshal(source)
			if workBytes+2*len(metadata) > auditdomain.MaximumRoundResultsSliceBytes ||
				processed > 0 && payloadBytes+source.SizeBytes > artifacts.MaxPayloadSize {
				break
			}
			if _, err := project.ReadExact(ctx, source); err != nil {
				return state, err
			}
			fork, err := forkRoundResultsDependency(ctx, service, projectScope, runScope, source, prefix)
			if err != nil {
				return state, err
			}
			if !fork.SourceRef.SameExact(source.Ref) || fork.MediaType != source.MediaType || fork.Size != source.SizeBytes || fork.TargetRef.ValidateExact() != nil {
				return state, fmt.Errorf("%w: round-results fork differs from its exact source", runservice.ErrInvalid)
			}
			// The exact target revision and the fork commit with this checkpoint.
			// Keeping only revision IDs avoids rescanning all prior bindings or
			// issuing one metadata read per dependency on the final slice.
			checkpoint.Revisions = append(checkpoint.Revisions, *fork.TargetRef.Revision)
			workBytes += 2 * len(metadata)
			payloadBytes += source.SizeBytes
			checkpoint.NextDependency++
			processed++
		}
		if checkpoint.NextDependency == len(dependencies) {
			mappings := make(map[string]contracts.ExactArtifact, len(dependencies))
			page.Mappings = make([]auditdomain.RoundResultsMapping, 0, len(dependencies))
			for index, source := range dependencies {
				ref := roundResultsDependencyTarget(source, prefix)
				revision := checkpoint.Revisions[index]
				ref.Revision = &revision
				consumer := source.Clone()
				consumer.Ref = ref
				if consumer.Validate() != nil {
					return state, fmt.Errorf("%w: recovered round-results dependency revision is invalid", runservice.ErrInvalid)
				}
				mappings[source.Ref.Key()] = consumer
				page.Mappings = append(page.Mappings, auditdomain.RoundResultsMapping{Source: source, Consumer: consumer})
			}
			for index := range page.Items {
				item := &page.Items[index]
				if item.Result != nil {
					value := mappings[item.Result.Ref.Key()]
					item.Result = &value
				}
				for attachment := range item.Attachments {
					item.Attachments[attachment].Artifact = mappings[item.Attachments[attachment].Artifact.Ref.Key()]
				}
			}
			data, err := auditdomain.EncodeRoundResultsPage(page)
			if err != nil {
				return state, fmt.Errorf("%w: projected round-results page exceeds its contract", runservice.ErrInvalid)
			}
			if workBytes+len(data) <= auditdomain.MaximumRoundResultsSliceBytes {
				if checkpoint.MetadataBytes+int64(len(data)) > auditdomain.MaximumRoundResultsBytes {
					return state, fmt.Errorf("%w: complete round-results metadata exceeds its budget", runservice.ErrInvalid)
				}
				written, err := service.CreateFrozen(ctx, runScope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: fmt.Sprintf("%spage-%03d", prefix, checkpoint.NextPage)}, artifacts.Payload{MediaType: "application/json", Data: data})
				if err != nil {
					return state, err
				}
				consumer := contracts.ExactContent(written.Ref, written.MediaType, data)
				checkpoint.Pages = append(checkpoint.Pages, auditdomain.RoundResultsPageRef{Ordinal: page.Ordinal, ItemCount: len(page.Items), Artifact: consumer})
				checkpoint.MetadataBytes += int64(len(data))
				checkpoint.NextPage++
				checkpoint.NextDependency = 0
				checkpoint.Revisions = []string{}
				state.Refs = append(state.Refs, consumer.Ref)
			}
		}
	}
	if checkpoint.NextPage == len(manifest.Pages) {
		consumer := auditdomain.RoundResultsConsumer{Schema: auditdomain.RoundResultsConsumerSchema, Source: root, AuditID: manifest.AuditID,
			RoundID: manifest.RoundID, SourceGeneration: manifest.SourceGeneration, ItemCount: manifest.ItemCount,
			Pages: checkpoint.Pages, Mappings: []auditdomain.RoundResultsMapping{}}
		for index, page := range checkpoint.Pages {
			consumer.Mappings = append(consumer.Mappings, auditdomain.RoundResultsMapping{Source: manifest.Pages[index].Artifact, Consumer: page.Artifact})
		}
		data, err := auditdomain.EncodeRoundResultsConsumer(consumer)
		if err != nil || checkpoint.MetadataBytes+int64(len(data)) > auditdomain.MaximumRoundResultsBytes {
			return state, fmt.Errorf("%w: round-results consumer descriptor exceeds its budget", runservice.ErrInvalid)
		}
		written, err := service.CreateFrozen(ctx, runScope, contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: prefix + "descriptor"}, artifacts.Payload{MediaType: "application/json", Data: data})
		if err != nil {
			return state, err
		}
		fork, err := service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageInputFork, Source: runScope, SourceRef: written.Ref,
			Target: runScope, TargetRef: contracts.ArtifactRef{Namespace: "inputs", Name: slot}, Frozen: true, Replay: true})
		if err != nil {
			return state, err
		}
		state.Refs = append(state.Refs, written.Ref, fork.TargetRef)
		state.Complete = true
	}
	state.Checkpoint, err = json.Marshal(checkpoint)
	return state, err
}

func roundResultsRequiredDependencies(page auditdomain.RoundResultsPage, namespace string) ([]contracts.ExactArtifact, error) {
	values := map[string]contracts.ExactArtifact{}
	add := func(value contracts.ExactArtifact) error {
		if value.Ref.Namespace != namespace || value.Validate() != nil || !value.HasMetadata() {
			return runservice.ErrInvalid
		}
		if previous, found := values[value.Ref.Key()]; found && previous.Verify(value) != nil {
			return runservice.ErrInvalid
		}
		values[value.Ref.Key()] = value
		return nil
	}
	for _, item := range page.Items {
		if item.Result != nil {
			if err := add(*item.Result); err != nil {
				return nil, err
			}
		}
		for _, attachment := range item.Attachments {
			if err := add(attachment.Artifact); err != nil {
				return nil, err
			}
		}
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]contracts.ExactArtifact, 0, len(keys))
	for _, key := range keys {
		result = append(result, values[key])
	}
	return result, nil
}

func roundResultsDependencyTarget(source contracts.ExactArtifact, prefix string) contracts.ArtifactRef {
	key := auditdomain.DigestBytes([]byte(prefix + "\n" + source.Ref.Key() + "\n" + source.Digest))
	return contracts.ArtifactRef{Namespace: artifactpolicy.AuditContextNamespace, Name: "round-results-content-" + strings.TrimPrefix(key, "sha256:")}
}

func forkRoundResultsDependency(ctx context.Context, service *artifacts.Service, project, run artifacts.Scope, source contracts.ExactArtifact, prefix string) (artifacts.ForkResult, error) {
	return service.Fork(ctx, artifacts.Fork{Kind: artifacts.LineageInputFork, Source: project, SourceRef: source.Ref,
		Target: run, TargetRef: roundResultsDependencyTarget(source, prefix), Frozen: true, Replay: true})
}
