//go:build e2e

package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func iterativeProducerStage() domainGatewayStage {
	return domainGatewayStage{name: "iterative-seed-check", tools: []string{"read_audit_task", "submit_check_result", "finding", "read_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("finding", fixedArguments(map[string]any{"title": "Joined fixture observation", "description": "Independently inspect this exact accepted observation.", "proposed_checks": []map[string]string{{"objective": "Verify the exact joined fixture observation.", "method": "verify"}}})),
		toolGatewayStep("submit_check_result", findingResultArguments(map[string]any{"assessment": "violated", "summary": "A bounded fixture observation requires follow-up.", "completed": []string{"observation"}, "gaps": []string{}, "evidence": []map[string]string{{"kind": "observation", "summary": "Controlled fixture data, without external target work."}}})),
		finalGatewayStep("Seed observation retained", nil),
	}}
}

func (j *iterativeProcessJournal) assessmentStage(round int) domainGatewayStage {
	observation := iterativeObservation{}
	return domainGatewayStage{name: fmt.Sprintf("iterative-round-%d-assessment", round), tools: []string{"read_artifact", "write_artifact", "open_source_archive", "read_source"}, steps: []domainGatewayStep{
		toolGatewayStep("read_artifact", fixedArguments(map[string]any{"namespace": "inputs", "name": "snapshot", "revision": nil})),
		toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			if strictjson.Decode(data, &observation.Consumer) != nil {
				return nil, fmt.Errorf("invalid actual assessment consumer")
			}
			if _, err := auditdomain.EncodeRoundResultsConsumer(observation.Consumer); err != nil {
				return nil, err
			}
			if observation.Consumer.ItemCount != 1 || len(observation.Consumer.Pages) != 1 || len(observation.Consumer.Mappings) != 1 {
				return nil, fmt.Errorf("assessment omitted accepted membership")
			}
			return iterativeReadArguments(observation.Consumer.Pages[0].Artifact)
		}),
		toolGatewayStep("open_source_archive", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			observation.Page, err = auditdomain.DecodeRoundResultsPage(data)
			if err != nil {
				return nil, err
			}
			page := observation.Page
			if auditdomain.DigestBytes(data) != observation.Consumer.Pages[0].Artifact.Digest || page.AuditID != observation.Consumer.AuditID || page.RoundID != observation.Consumer.RoundID || page.SourceGeneration != observation.Consumer.SourceGeneration || len(page.Items) != 1 {
				return nil, fmt.Errorf("assessment page lost its immutable root identity")
			}
			item := page.Items[0]
			if item.FinalDisposition != "accepted-result" || item.Result == nil || item.SelectedAttempt == nil || item.SelectedAttempt.Attempt != 1 || item.SelectedAttempt.TerminalOutcome != "succeeded" || item.SelectedAttempt.CollectionDisposition != "accepted-result" || item.SelectedAttempt.SourceResult == nil || item.SelectedAttempt.RunID == nil {
				return nil, fmt.Errorf("assessment inferred success without an accepted attempt")
			}
			j.mu.Lock()
			defer j.mu.Unlock()
			if round == 1 && (*item.SelectedAttempt.RunID != j.producer || item.SelectedAttempt.SourceResult.Verify(j.producerResult.Exact) != nil) {
				return nil, fmt.Errorf("first assessment changed deleted producer attribution")
			}
			if round == 2 && (item.ItemKey != j.typedTask.ItemKey || *item.SelectedAttempt.RunID == j.producer) {
				return nil, fmt.Errorf("second assessment does not describe the actual typed child")
			}
			return iterativeReadArguments(*item.Result)
		}),
		toolGatewayStep("read_source", fixedArguments(map[string]any{"path": "check-results.json", "start_line": 1, "max_lines": 100})),
		toolGatewayStep("write_artifact", func(request map[string]any) (map[string]any, error) {
			set, data, err := iterativeReadResultMember(request)
			if err != nil {
				return nil, err
			}
			item := observation.Page.Items[0]
			result := set.Results[0]
			if result.ItemKey != item.ItemKey || result.SubjectKey != item.SubjectKey {
				return nil, fmt.Errorf("assessment result membership differs: %v", err)
			}
			expected := "violated"
			if round == 2 {
				expected = "supported"
			}
			if result.Assessment != expected {
				return nil, fmt.Errorf("assessment read the wrong ordinary check result")
			}
			observation.ResultMember = bytes.Clone(data)
			j.mu.Lock()
			j.snapshots[round] = observation
			j.mu.Unlock()
			return map[string]any{"namespace": "reader", "name": "report", "media_type": "text/plain", "data_base64": base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("Inspected the exact accepted Round %d result and attribution.", round))), "expected_revision": nil}, nil
		}),
		finalGatewayStep("Exact settled result inspected", map[string]domainArtifactBinding{"report": {namespace: "reader", name: "report"}}),
	}}
}

func (j *iterativeProcessJournal) followUpStage() domainGatewayStage {
	return domainGatewayStage{name: "iterative-typed-follow-up", tools: []string{"read_audit_task", "submit_check_result", "finding", "read_artifact", "open_source_archive", "read_source"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "read_audit_task")
			if len(responses) != 1 {
				return nil, fmt.Errorf("typed follow-up has no exact assigned task")
			}
			data, err := json.Marshal(responses[0]["task"])
			if err != nil {
				return nil, err
			}
			task, err := auditdomain.DecodeItemTask(data)
			if err != nil || task.AcceptedPlan == nil || task.Dependencies == nil {
				return nil, fmt.Errorf("typed follow-up lost accepted structure: %v", err)
			}
			j.mu.Lock()
			defer j.mu.Unlock()
			plan := task.AcceptedPlan
			if task.WorkflowRole != "ordinary-follow-up" || plan.Route.WorkflowRole != task.WorkflowRole || plan.Plan.Method != "verify" || plan.Source.ReceiptID != j.receiptID || plan.Source.RunID != j.producer || plan.Source.ProposedCheckOrdinal == nil || *plan.Source.ProposedCheckOrdinal != 0 || plan.Source.Proposal.Verify(j.proposal.Exact) != nil {
				return nil, fmt.Errorf("typed follow-up changed source, route or ordinal")
			}
			j.typedTask = task
			return map[string]any{"namespace": "inputs", "name": "proposal", "revision": nil}, nil
		}),
		toolGatewayStep("read_artifact", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			j.mu.Lock()
			defer j.mu.Unlock()
			if !bytes.Equal(data, j.proposal.Data) {
				return nil, fmt.Errorf("typed Runtime did not read the original proposal bytes")
			}
			return map[string]any{"namespace": "audit-context", "name": "retained-dependencies", "revision": nil}, nil
		}),
		toolGatewayStep("open_source_archive", func(request map[string]any) (map[string]any, error) {
			data, err := roundResultsToolBytes(request)
			if err != nil {
				return nil, err
			}
			var context struct {
				Mappings []struct {
					Name                       string
					Source, Retained, Consumer contracts.ExactArtifact
				}
			}
			if json.Unmarshal(data, &context) != nil || len(context.Mappings) < 2 {
				return nil, fmt.Errorf("typed Runtime has no complete exact input mapping")
			}
			found := map[string]bool{}
			var archive contracts.ExactArtifact
			j.mu.Lock()
			defer j.mu.Unlock()
			for _, mapping := range context.Mappings {
				expected := j.proposal.Exact
				if mapping.Name == "source-result" {
					expected = j.producerResult.Exact
					archive = mapping.Consumer
				}
				if mapping.Name == "proposal" || mapping.Name == "source-result" {
					if mapping.Source.Verify(expected) != nil || mapping.Retained.Digest != expected.Digest || mapping.Consumer.Digest != expected.Digest || mapping.Consumer.Ref.Namespace != "audit-context" || mapping.Consumer.Ref.Revision == nil {
						return nil, fmt.Errorf("typed original, retained and consumer descriptors differ")
					}
					found[mapping.Name] = true
				}
			}
			if !found["proposal"] || !found["source-result"] {
				return nil, fmt.Errorf("typed input mapping omitted a required source")
			}
			return iterativeReadArguments(archive)
		}),
		toolGatewayStep("read_source", fixedArguments(map[string]any{"path": "check-results.json", "start_line": 1, "max_lines": 100})),
		toolGatewayStep("submit_check_result", func(request map[string]any) (map[string]any, error) {
			set, document, err := iterativeReadResultMember(request)
			if err != nil {
				return nil, err
			}
			j.mu.Lock()
			defer j.mu.Unlock()
			original, err := auditdomain.ValidatePackage(j.producerResult.Data)
			if err != nil {
				return nil, err
			}
			member, found := original.MemberByID(auditdomain.CheckResultsMemberID)
			if !found || !bytes.Equal(bytes.TrimSpace(member.Data()), bytes.TrimSpace(document)) || set.Results[0].Assessment != "violated" {
				return nil, fmt.Errorf("typed Runtime did not inspect the original accepted result member")
			}
			return map[string]any{"assessment": "supported", "summary": "The exact retained fixture observation was checked.", "completed": []string{"verify"}, "gaps": []string{}, "evidence": []map[string]string{{"kind": "observation", "summary": "The supplied proposal and accepted result match the original bytes."}}}, nil
		}),
		finalGatewayStep("Typed observation recorded", nil),
	}}
}

func iterativeDefaultStage() domainGatewayStage {
	return domainGatewayStage{name: "deterministic-inventory-default-check", tools: []string{"read_audit_task", "submit_check_result", "read_artifact"}, steps: []domainGatewayStep{
		toolGatewayStep("read_audit_task", fixedArguments(map[string]any{})),
		toolGatewayStep("submit_check_result", func(request map[string]any) (map[string]any, error) {
			responses := findingsToolResponses(request, "read_audit_task")
			if len(responses) != 1 {
				return nil, fmt.Errorf("default check has no assigned task")
			}
			data, _ := json.Marshal(responses[0]["task"])
			task, err := auditdomain.DecodeItemTask(data)
			if err != nil || task.ItemKey != "default-seed" || task.AcceptedPlan != nil || task.Dependencies != nil {
				return nil, fmt.Errorf("default inventory acquired optional typed context: %v", err)
			}
			return map[string]any{"assessment": "satisfied", "summary": "The unchanged deterministic inventory check completed.", "completed": []string{"observation"}, "gaps": []string{}, "evidence": []map[string]string{{"kind": "observation", "summary": "A bounded default-route observation."}}}, nil
		}),
		finalGatewayStep("Default check recorded", nil),
	}}
}

func iterativeReadArguments(exact contracts.ExactArtifact) (map[string]any, error) {
	if exact.Validate() != nil || exact.Ref.Namespace != "audit-context" || exact.Ref.Revision == nil {
		return nil, fmt.Errorf("assessment attempted an unprojected exact read")
	}
	return map[string]any{"namespace": exact.Ref.Namespace, "name": exact.Ref.Name, "revision": *exact.Ref.Revision}, nil
}

func iterativeReadResultMember(request map[string]any) (auditdomain.CheckResultSet, []byte, error) {
	responses := findingsToolResponses(request, "read_source")
	if len(responses) != 1 {
		return auditdomain.CheckResultSet{}, nil, fmt.Errorf("exact archive result member was not read")
	}
	value, ok := responses[0]["text"].(string)
	if !ok || responses[0]["truncated"] != false || responses[0]["partialLine"] != false {
		return auditdomain.CheckResultSet{}, nil, fmt.Errorf("archive result member read is incomplete")
	}
	data := []byte(value)
	set, err := auditdomain.DecodeCheckResultSet(data)
	if err != nil || len(set.Results) != 1 {
		return set, nil, fmt.Errorf("accepted result member has incorrect membership: %v", err)
	}
	return set, data, nil
}
