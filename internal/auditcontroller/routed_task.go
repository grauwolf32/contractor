package auditcontroller

import (
	"bytes"
	"maps"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

func validateRoutedTaskContent(pkg *auditdomain.Package, manifest auditdomain.ExecutionItem, item auditstore.Item, scope map[string]string) error {
	member, ok := pkg.MemberByID("task-document")
	if !ok || len(pkg.Members()) != 1 || member.Metadata().Path != "task.json" || member.Metadata().MediaType != auditdomain.JSONMediaType {
		return invalidSubmission("typed task document is absent")
	}
	task, err := auditdomain.DecodeItemTask(member.Data())
	if err != nil {
		return err
	}
	if task.AcceptedPlan == nil || manifest.AcceptedPlan == nil || !sameDependencyManifest(task.Dependencies, manifest.Dependencies) || task.ItemKey != item.ItemKey || task.SubjectKey != item.SubjectKey || task.WorkflowRole != item.WorkflowRole || !maps.Equal(task.Scope, scope) || item.Origin.SourceRef == nil || !task.SourceRef.SameExact(*item.Origin.SourceRef) || task.SourceContentDigest != item.Origin.SourceContentDigest || task.SourceMediaType != item.Origin.SourceMediaType || task.CanonicalInventoryDigest != item.Origin.CanonicalInventoryDigest {
		return invalidSubmission("typed task differs from accepted source, scope or dependency closure")
	}
	actual, err := auditdomain.EncodeAcceptedPlan(*task.AcceptedPlan, task.Dependencies)
	if err != nil {
		return err
	}
	expected, err := auditdomain.EncodeAcceptedPlan(*manifest.AcceptedPlan, manifest.Dependencies)
	if err != nil {
		return err
	}
	if !bytes.Equal(actual, expected) {
		return invalidSubmission("task and execution accepted plans differ")
	}
	return nil
}
