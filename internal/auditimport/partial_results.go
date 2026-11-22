package auditimport

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type partialTaskResults struct {
	common   bool
	run      *runstore.WorkflowRun
	accepted map[string]auditcontract.AcceptedJudgment
	links    map[string]auditstore.ArtifactLink
	evidence map[string][]auditcontract.Material
	reason   auditcontract.ResultStop
}

func (i *Importer) readPartialResults(ctx context.Context, snapshot auditstore.ReconcileSnapshot, execution auditstore.Execution, tasks []auditstore.Task) (partialTaskResults, error) {
	result := partialTaskResults{accepted: map[string]auditcontract.AcceptedJudgment{}, links: map[string]auditstore.ArtifactLink{}, evidence: map[string][]auditcontract.Material{}}
	if execution.RunID == nil {
		return result, nil
	}
	run, err := i.runs.GetRun(ctx, *execution.RunID)
	if err != nil {
		return result, err
	}
	result.run = &run
	if run.AuditCompletion == nil {
		return result, nil
	}
	if run.OwnerID != snapshot.Audit.OwnerID || run.ProjectID == nil || *run.ProjectID != snapshot.Audit.ProjectID || run.PublicationMode != runstore.PublicationAuditManaged || run.AuditExecutionID == nil || *run.AuditExecutionID != execution.ExecutionID || run.AuditSubmissionKey == nil || *run.AuditSubmissionKey != execution.SubmissionKey {
		return result, fmt.Errorf("%w: partial result origin", ErrPermanent)
	}
	result.common = true
	result.reason = auditcontract.ResultStop{Source: "run", Code: run.StateReason.Code, Message: run.StateReason.Message}
	if result.reason.Code == "" {
		result.reason = auditcontract.ResultStop{Source: "unknown", Code: "unknown"}
	}
	if len(result.reason.Message) > 4096 {
		result.reason.Message = result.reason.Message[:4096]
		for !utf8.ValidString(result.reason.Message) {
			result.reason.Message = result.reason.Message[:len(result.reason.Message)-1]
		}
	}
	links, err := i.reads.ListExecutionJudgments(ctx, execution.AuditID, execution.ExecutionID)
	if err != nil {
		return result, err
	}
	records := make([]auditcontract.StoredJudgment, 0, len(links))
	var total int64
	for _, link := range links {
		total += link.Artifact.SizeBytes
		if total > auditcontract.MaximumDocumentBytes {
			return result, fmt.Errorf("%w: judgment bytes", ErrPermanent)
		}
		data, err := i.artifacts.ReadAuditExact(ctx, execution.AuditID, link.Artifact)
		if err != nil {
			return result, err
		}
		material := auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: link.Artifact.Ref, Digest: link.Artifact.Digest, MediaType: link.Artifact.MediaType, SizeBytes: link.Artifact.SizeBytes}
		records = append(records, auditcontract.StoredJudgment{Material: material, Provenance: link.SourceProvenance, Data: data})
		key, _ := strictjson.Canonical(material)
		result.links[string(key)] = link
	}
	latest, err := auditcontract.LatestJudgments(records, execution.AuditID, execution.ExecutionID, *execution.RunID)
	if err != nil {
		return result, fmt.Errorf("%w: retained judgment", ErrPermanent)
	}
	members := map[string]auditstore.Task{}
	for index, task := range tasks {
		members[auditcontract.ExecutionItemKey(index)] = task
	}
	for key, accepted := range latest {
		task, ok := members[key]
		if !ok || accepted.Generation.TaskID != task.Document.TaskID || !reflect.DeepEqual(accepted.Item.Origin.Hypothesis, task.Document.Hypothesis) {
			return result, fmt.Errorf("%w: retained judgment member", ErrPermanent)
		}
		result.accepted[task.Document.TaskID] = accepted
	}
	notes, err := i.reads.ListExecutionEvidence(ctx, execution.AuditID, execution.ExecutionID)
	if err != nil {
		return result, err
	}
	total = 0
	for _, link := range notes {
		total += link.Artifact.SizeBytes
		if total > auditcontract.MaximumDocumentBytes {
			return result, fmt.Errorf("%w: retained note bytes", ErrPermanent)
		}
		data, err := i.artifacts.ReadAuditExact(ctx, execution.AuditID, link.Artifact)
		if err != nil {
			return result, err
		}
		var record auditcontract.EvidenceRecord
		if strictjson.Decode(data, &record) != nil || record.Validate() != nil || record.Origin.AuditID != execution.AuditID || record.Origin.ExecutionID != execution.ExecutionID || record.Origin.RunID != *execution.RunID {
			return result, fmt.Errorf("%w: retained note", ErrPermanent)
		}
		memberFound := false
		for _, task := range tasks {
			memberFound = memberFound || task.Document.TaskID == record.Origin.TaskID && reflect.DeepEqual(task.Document.Hypothesis, record.Origin.Hypothesis)
		}
		if !memberFound {
			return result, fmt.Errorf("%w: retained note member", ErrPermanent)
		}
		material := auditcontract.Material{ScopeKind: "audit", ScopeID: execution.AuditID, Ref: link.Artifact.Ref, Digest: link.Artifact.Digest, MediaType: link.Artifact.MediaType, SizeBytes: link.Artifact.SizeBytes}
		selected := append([]auditcontract.Material{material}, record.Sources...)
		for _, source := range selected {
			values := result.evidence[record.Origin.TaskID]
			if !slices.ContainsFunc(values, func(prior auditcontract.Material) bool { return reflect.DeepEqual(prior, source) }) {
				result.evidence[record.Origin.TaskID] = append(values, source)
			}
		}
	}
	return result, nil
}

func (p partialTaskResults) selectTask(execution auditstore.Execution, task auditstore.Task, collection *auditstore.TaskCollection, retained *[]auditstore.ArtifactLink) error {
	if !p.common {
		return nil
	}
	add := func(link auditstore.ArtifactLink) {
		for _, prior := range *retained {
			if prior.LogicalKey == link.LogicalKey {
				return
			}
		}
		*retained = append(*retained, link)
	}
	if accepted, ok := p.accepted[task.Document.TaskID]; ok {
		// A valid final package adds trusted publication metadata. On any gap,
		// retain the already acknowledged per-item value instead of discarding it.
		if collection.Result == nil || collection.GapCode != nil {
			material := accepted.Material
			collection.Result = &material
			key, _ := strictjson.Canonical(material)
			link, found := p.links[string(key)]
			if !found {
				return fmt.Errorf("%w: missing judgment receipt", ErrPermanent)
			}
			add(link)
		}
		collection.Retryable = false
		return nil
	}
	collection.Result = nil
	stop := &auditcontract.TaskStop{Author: "system", Meaning: "no_accepted_conclusion", Reason: p.reason, Evidence: []auditcontract.Material{}}
	if task.Document.Hypothesis != nil {
		value := "inconclusive"
		stop.Verdict = &value
	}
	for _, source := range p.evidence[task.Document.TaskID] {
		stop.Evidence = append(stop.Evidence, source)
		key, _ := strictjson.Canonical(source)
		name := strings.TrimPrefix(auditdomain.DigestBytes(key), "sha256:")
		add(auditstore.ArtifactLink{Scope: artifacts.ScopeAudit, LogicalKey: "execution/" + execution.ExecutionID + "/stop-source/" + name, Artifact: source.Exact(), SourceProvenance: []byte(`{"kind":"retained-stop-source"}`)})
	}
	if stop.Validate(execution.AuditID, task.Document.Hypothesis != nil) != nil {
		return fmt.Errorf("%w: invalid stop metadata", ErrPermanent)
	}
	collection.Stop = stop
	return nil
}

func (p partialTaskResults) verifyAcceptedFinal(tasks []auditstore.Task, raw []byte) error {
	if !p.common {
		return nil
	}
	var document struct {
		Schema string                     `json:"schema"`
		Items  []auditcontract.Conclusion `json:"items"`
	}
	if strictjson.Decode(raw, &document) != nil || document.Schema != "contractor.audit.conclusions.v1" || len(document.Items) != len(tasks) || len(p.accepted) != len(tasks) {
		return fmt.Errorf("%w: unaccepted final result", ErrPermanent)
	}
	members := map[string]string{}
	for index, task := range tasks {
		members[auditcontract.ExecutionItemKey(index)] = task.Document.TaskID
	}
	seen := map[string]bool{}
	for _, item := range document.Items {
		accepted, ok := p.accepted[members[item.ItemKey]]
		if seen[item.ItemKey] || !ok || !reflect.DeepEqual(accepted.Item.Conclusion, item) {
			return fmt.Errorf("%w: stale final result", ErrPermanent)
		}
		seen[item.ItemKey] = true
	}
	return nil
}
