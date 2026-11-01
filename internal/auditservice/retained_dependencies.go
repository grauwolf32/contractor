package auditservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/findingintake"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// dependencyConsumerPins binds configuration/Skills to their exact content,
// never a mutable catalog name/version. The baseline already owns their holds.
func dependencyConsumerPins(binding auditprofile.ResolvedAuditWorkflowBinding, baseline BaselineSnapshot) (string, string, error) {
	workflow, skills, err := auditbaseline.ConsumerContentPins(binding, baseline)
	if err != nil {
		return "", "", fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return workflow, skills, nil
}

func hasRetainedDependencyMapping(binding auditprofile.ResolvedAuditWorkflowBinding) bool {
	for _, mapping := range binding.Inputs {
		if mapping.Source == auditprofile.AuditInputFromRetainedDependency {
			return true
		}
	}
	return false
}

// buildAcceptedDependencies is the existing finding receipt adapter's trusted
// manifest producer. A Worker supplies no manifest or origin grant. Only
// exact baseline inputs, accepted prepare outputs, and the receipt's exact
// accepted producer attempt and held evidence are candidates.
func (s *Service) buildAcceptedDependencies(ctx context.Context, writer auditArtifactWriter, audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot, inventory auditdomain.Inventory, receipts map[string]findingintake.Receipt) (map[string]auditdomain.RetainedDependencyManifest, []auditstore.ArtifactLink, int, error) {
	return buildAcceptedDependenciesWithReader(ctx, writer, s.reads(), audit, profile, baseline, inventory, receipts)
}

// buildAcceptedDependenciesWithReader also serves one bounded typed admission
// candidate. Its reader and writer join that candidate's existing claim-bound
// transaction, so accepted source checks, exact retention and staging commit
// together without a second pool acquisition.
func buildAcceptedDependenciesWithReader(ctx context.Context, writer auditArtifactWriter, reads dependencySourceReader, audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot, inventory auditdomain.Inventory, receipts map[string]findingintake.Receipt) (map[string]auditdomain.RetainedDependencyManifest, []auditstore.ArtifactLink, int, error) {
	manifests := map[string]auditdomain.RetainedDependencyManifest{}
	retainedLinks := map[string]auditstore.ArtifactLink{}
	retainedBaseline := map[string]auditstore.ExactArtifact{}
	acceptedPrepare := map[string]auditstore.AcceptedPreparationOutput{}
	acceptedResults := map[string]auditstore.DependencyResultSource{}
	descriptorCount, metadataBytes, admitted := 0, 0, 0
	var newlyRetainedBytes int64
	for index, task := range inventory.Tasks {
		if len(manifests) > 0 && (len(manifests) >= 128 || descriptorCount > 512-auditdomain.MaximumRetainedDependencies || metadataBytes > (1<<20)-auditdomain.MaximumRetainedDependencyBytes) {
			break
		}
		admitted = index + 1
		binding := profile.Workflows[task.Document.WorkflowRole]
		if !hasRetainedDependencyMapping(binding) {
			continue
		}
		if task.Document.Finding == nil {
			return nil, nil, 0, fmt.Errorf("%w: retained_dependency_origin_unsupported", ErrInvalid)
		}
		finding := task.Document.Finding
		receipt, exists := receipts[finding.ReceiptID]
		if !exists {
			return nil, nil, 0, fmt.Errorf("%w: retained_dependency_receipt_unavailable", ErrInvalid)
		}
		hold, exists := exactAuditHold(receipt, audit.AuditID, audit.ProjectID)
		if !exists {
			return nil, nil, 0, fmt.Errorf("%w: retained_dependency_receipt_unavailable", ErrInvalid)
		}
		workflowDigest, skillsDigest, err := dependencyConsumerPins(binding, baseline)
		if err != nil {
			return nil, nil, 0, err
		}
		manifest := auditdomain.RetainedDependencyManifest{Schema: auditdomain.RetainedDependencySchema, OwnerID: audit.OwnerID, ProjectID: audit.ProjectID, AuditID: audit.AuditID, ItemKey: task.Document.ItemKey, BaselineDigest: auditdomain.DigestBytes(audit.BaselineSnapshot), ProfileDigest: audit.Profile.Digest, WorkflowDigest: workflowDigest, SkillsDigest: skillsDigest, Dependencies: []auditdomain.RetainedDependency{}}
		ordinal := finding.ProposedCheckOrdinal
		findingOrigin := auditdomain.RetainedDependencyOrigin{Kind: "finding-receipt", ReceiptID: receipt.ReceiptID, ProposedCheckOrdinal: &ordinal, RunID: receipt.Origin.RunID}
		sourceProposal := auditstore.ExactArtifact{Ref: receipt.Proposal.Ref, Digest: receipt.Proposal.Digest, MediaType: receipt.Proposal.MediaType, SizeBytes: receipt.Proposal.SizeBytes}
		proposal := auditdomain.RetainedDependency{Name: "proposal", Kind: findingOrigin.Kind, Projection: "exact", Source: sourceProposal, Retained: hold.Proposal, Origin: findingOrigin, Requires: []string{}}
		if len(receipt.Evidence) != len(hold.Evidence) || len(hold.Evidence) >= auditdomain.MaximumRetainedDependencies {
			return nil, nil, 0, fmt.Errorf("%w: retained_dependency_evidence_incomplete", ErrInvalid)
		}
		for index, evidence := range hold.Evidence {
			name := fmt.Sprintf("proposal-evidence-%03d", index)
			original := receipt.Evidence[index]
			manifest.Dependencies = append(manifest.Dependencies, auditdomain.RetainedDependency{Name: name, Kind: findingOrigin.Kind, Projection: "exact", Source: original, Retained: evidence, Origin: findingOrigin, Requires: []string{}})
			proposal.Requires = append(proposal.Requires, name)
		}
		manifest.Dependencies = append(manifest.Dependencies, proposal)
		requiredNames := map[string]bool{}
		for _, mapping := range binding.Inputs {
			if mapping.Source == auditprofile.AuditInputFromRetainedDependency {
				requiredNames[mapping.Name] = true
			}
		}

		// Preparation input mappings are an explicit trusted graph. Expand only
		// this graph, never refs encountered in an output's arbitrary JSON.
		for changed := true; changed; {
			changed = false
			for _, name := range slices.Sorted(maps.Keys(requiredNames)) {
				if !strings.HasPrefix(name, "prepare.") {
					continue
				}
				for role, producer := range profile.Workflows {
					if producer.Kind != auditprofile.AuditWorkflowPrepare {
						continue
					}
					for logical := range producer.Outputs {
						if "prepare."+role+"."+logical != name {
							continue
						}

						output, exists := acceptedPrepare[name]
						if !exists {
							var err error
							output, err = reads.GetPreparationOutput(ctx, audit.AuditID, role, logical)
							if err != nil {
								return nil, nil, 0, err
							}
							acceptedPrepare[name] = output
						}
						for slot, input := range producer.Inputs {
							if _, present := output.Inputs[slot]; !present {
								if producer.Workflow.Inputs[slot].Required {
									return nil, nil, 0, fmt.Errorf("%w: retained_dependency_prepare_input_unavailable", ErrInvalid)
								}
								continue
							}
							required := ""
							switch input.Source {
							case auditprofile.AuditInputFromAudit:
								required = "baseline." + input.Name
							case auditprofile.AuditInputFromPreparation:
								required = "prepare." + input.Role + "." + input.Name
							}
							if required != "" && !requiredNames[required] {
								requiredNames[required] = true
								changed = true
							}
						}
					}
				}
			}
			if len(requiredNames) > auditdomain.MaximumRetainedDependencies {
				return nil, nil, 0, fmt.Errorf("%w: retained_dependency_closure_limit", ErrInvalid)
			}
		}
		for _, name := range slices.Sorted(maps.Keys(requiredNames)) {
			if name == "proposal" || strings.HasPrefix(name, "proposal-evidence-") {
				continue
			}
			if strings.HasPrefix(name, "baseline.") {
				source, exists := baseline.Inputs[strings.TrimPrefix(name, "baseline.")]
				if !exists {
					return nil, nil, 0, fmt.Errorf("%w: retained_dependency_baseline_unavailable", ErrInvalid)
				}
				retained, cached := retainedBaseline[source.Ref.Key()]
				if !cached {
					err = writer.withArtifactService(ctx, auditdomain.ArtifactNamespace(audit.AuditID), func(service *artifacts.Service) error {
						scope, _ := service.Project(audit.ProjectID)
						metadata, err := scope.Metadata(ctx, source.Ref)
						if err != nil {
							return err
						}
						if err := source.Verify(metadata.Exact()); err != nil {
							return err
						}
						fork, err := service.RetainAuditDependency(ctx, audit.ProjectID, auditdomain.ArtifactNamespace(audit.AuditID), source.Ref)
						if err != nil {
							return err
						}
						retained = source.Clone()
						retained.Ref = fork.TargetRef
						return nil
					})
					if err != nil {
						return nil, nil, 0, err
					}
					retainedBaseline[source.Ref.Key()] = retained
				}
				dependency := auditdomain.RetainedDependency{Name: name, Kind: "baseline", Projection: "exact", Source: source, Retained: retained, Origin: auditdomain.RetainedDependencyOrigin{Kind: "baseline", BaselineInput: strings.TrimPrefix(name, "baseline.")}, Requires: []string{}}
				manifest.Dependencies = append(manifest.Dependencies, dependency)
				key := "dependency/" + auditdomain.DeterministicID("revision", retained.Ref.Key())
				provenance, _ := json.Marshal(dependency)

				if _, planned := retainedLinks[key]; !planned {
					existing, err := reads.GetArtifactLink(ctx, audit.AuditID, key)
					if errors.Is(err, auditstore.ErrNotFound) {
						newlyRetainedBytes += retained.SizeBytes
						if newlyRetainedBytes > audit.Limits.MaxEvidenceBytes-audit.RetainedEvidenceBytes {
							return nil, nil, 0, auditstore.ErrEvidenceBudgetExhausted
						}
					} else if err != nil {
						return nil, nil, 0, err
					} else if retained.Verify(existing.Artifact) != nil {
						return nil, nil, 0, fmt.Errorf("%w: retained_dependency_receipt_conflict", ErrInvalid)
					}
				}
				retainedLinks[key] = auditstore.ArtifactLink{LogicalKey: key, Artifact: retained, SourceProvenance: provenance}
			} else if strings.HasPrefix(name, "prepare.") {
				var found *auditstore.AcceptedPreparationOutput
				for role, producer := range profile.Workflows {
					if producer.Kind != auditprofile.AuditWorkflowPrepare {
						continue
					}
					for logical := range producer.Outputs {
						if "prepare."+role+"."+logical != name {
							continue
						}
						if found != nil {
							return nil, nil, 0, fmt.Errorf("%w: retained_dependency_prepare_ambiguous", ErrInvalid)
						}

						output, exists := acceptedPrepare[name]
						if !exists {
							var err error
							output, err = reads.GetPreparationOutput(ctx, audit.AuditID, role, logical)
							if err != nil {
								return nil, nil, 0, err
							}
							acceptedPrepare[name] = output
						}
						found = &output
					}
				}
				if found == nil {
					return nil, nil, 0, fmt.Errorf("%w: retained_dependency_prepare_unavailable", ErrInvalid)
				}

				requires := []string{}
				for slot, input := range profile.Workflows[found.WorkflowRole].Inputs {
					if _, present := found.Inputs[slot]; !present {
						continue
					}
					switch input.Source {
					case auditprofile.AuditInputFromAudit:
						requires = append(requires, "baseline."+input.Name)
					case auditprofile.AuditInputFromPreparation:
						requires = append(requires, "prepare."+input.Role+"."+input.Name)
					}
				}
				sort.Strings(requires)
				requires = slices.Compact(requires)
				manifest.Dependencies = append(manifest.Dependencies, auditdomain.RetainedDependency{Name: name, Kind: "prepare-output", Projection: "exact", Source: found.Output.Source, Retained: found.Output.Retained, Origin: auditdomain.RetainedDependencyOrigin{Kind: "prepare-output", WorkflowRole: found.WorkflowRole, LogicalOutput: found.Output.LogicalName, ExecutionID: found.ExecutionID, Attempt: found.RoleAttempt, RunID: found.RunID}, Requires: requires})

			} else if name == "source-result" {
				if receipt.Origin.Audit == nil || receipt.Origin.Audit.AuditID != audit.AuditID {
					return nil, nil, 0, fmt.Errorf("%w: retained_dependency_result_foreign", ErrInvalid)
				}

				resultKey := receipt.Origin.Audit.ExecutionID + "\x00" + task.Document.SubjectKey
				result, cached := acceptedResults[resultKey]
				if !cached {
					var err error
					result, err = reads.DependencyResultSource(ctx, audit.AuditID, receipt.Origin.Audit.ExecutionID, task.Document.SubjectKey)
					if err != nil {
						return nil, nil, 0, err
					}
					acceptedResults[resultKey] = result
				}

				origin := auditdomain.RetainedDependencyOrigin{Kind: "accepted-item-result", RoundID: result.RoundID, ItemID: result.ItemID, ExecutionID: result.ExecutionID, Attempt: result.ItemAttempt, RunID: result.RunID}
				root := auditdomain.RetainedDependency{Name: name, Kind: origin.Kind, Projection: "exact", Source: result.Source, Retained: result.Retained, Origin: origin, Requires: []string{}}
				seen := map[string]bool{result.Retained.Ref.Key(): true}
				for index, link := range result.Evidence {
					if seen[link.Artifact.Ref.Key()] {
						continue
					}
					seen[link.Artifact.Ref.Key()] = true
					var provenance struct {
						Source   auditstore.ResultSourceProvenance `json:"source"`
						Evidence auditdomain.Evidence              `json:"evidence"`
					}

					if strictjson.Decode(link.SourceProvenance, &provenance) != nil || provenance.Evidence.Artifact == nil {
						return nil, nil, 0, fmt.Errorf("%w: retained_dependency_evidence_origin_invalid", ErrInvalid)
					}
					original := link.Artifact.Clone()
					original.Ref = provenance.Evidence.Artifact.Clone()
					evidenceName := fmt.Sprintf("source-evidence-%03d", index)
					root.Requires = append(root.Requires, evidenceName)
					manifest.Dependencies = append(manifest.Dependencies, auditdomain.RetainedDependency{Name: evidenceName, Kind: origin.Kind, Projection: "exact", Source: original, Retained: link.Artifact, Origin: origin, Requires: []string{}})
				}
				sort.Strings(root.Requires)
				manifest.Dependencies = append(manifest.Dependencies, root)
			} else {
				return nil, nil, 0, fmt.Errorf("%w: retained_dependency_name_unsupported", ErrInvalid)
			}
		}
		sort.Slice(manifest.Dependencies, func(i, j int) bool { return manifest.Dependencies[i].Name < manifest.Dependencies[j].Name })
		encoded, err := auditdomain.EncodeRetainedDependencies(manifest)
		if err != nil {
			return nil, nil, 0, err
		}
		for _, mapping := range binding.Inputs {
			if mapping.Source != auditprofile.AuditInputFromRetainedDependency {
				continue
			}
			present := false
			for _, d := range manifest.Dependencies {
				if d.Name == mapping.Name {
					present = true
					break
				}
			}
			if !present {
				return nil, nil, 0, fmt.Errorf("%w: retained_dependency_closure_incomplete", ErrInvalid)
			}
		}
		descriptorCount += len(manifest.Dependencies)
		metadataBytes += len(encoded)
		if len(manifests) >= 128 || descriptorCount > 512 || metadataBytes > 1<<20 {
			return nil, nil, 0, fmt.Errorf("%w: retained_dependency_slice_limit", ErrInvalid)
		}
		manifests[task.Document.ItemKey] = manifest
	}
	links := make([]auditstore.ArtifactLink, 0, len(retainedLinks))
	for _, key := range slices.Sorted(maps.Keys(retainedLinks)) {
		links = append(links, retainedLinks[key])
	}
	return manifests, links, admitted, nil
}

// retainedDependencyPrefix preserves original receipt order and ordinals;
// omitted checks have no accepted item and remain eligible for the next Round.
func retainedDependencyPrefix(proposals []auditdomain.FindingInventoryProposal, tasks []auditdomain.GeneratedTask) []auditdomain.FindingInventoryProposal {
	selected := map[string]map[int]bool{}
	for _, task := range tasks {
		finding := task.Document.Finding
		if finding == nil {
			continue
		}
		if selected[finding.ReceiptID] == nil {
			selected[finding.ReceiptID] = map[int]bool{}
		}
		selected[finding.ReceiptID][finding.ProposedCheckOrdinal] = true
	}
	result := []auditdomain.FindingInventoryProposal{}
	for _, proposal := range proposals {
		ordinals := []int{}
		for _, ordinal := range proposal.SelectedCheckOrdinals {
			if selected[proposal.ReceiptID][ordinal] {
				ordinals = append(ordinals, ordinal)
			}
		}
		if len(ordinals) > 0 {
			proposal.SelectedCheckOrdinals = ordinals
			result = append(result, proposal)
		}
	}
	return result
}
