package auditcontroller

import (
	"maps"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// validateTaskContent checks the task body already pinned by exact package
// identity. Plan and dependency bytes are transported only in this document;
// SQL item dependencies remain the accepted dispatch authority.
func validateTaskContent(task auditdomain.ItemTask, item auditstore.Item, scope map[string]string, typed bool) error {
	if (task.AcceptedPlan != nil) != typed || !sameDependencyManifest(task.Dependencies, item.Dependencies) || task.ItemKey != item.ItemKey || task.Kind != item.Kind || task.SubjectKey != item.SubjectKey || task.WorkflowRole != item.WorkflowRole || !maps.Equal(task.Scope, scope) || item.Origin.SourceRef == nil || !task.SourceRef.SameExact(*item.Origin.SourceRef) || task.SourceContentDigest != item.Origin.SourceContentDigest || task.SourceMediaType != item.Origin.SourceMediaType || task.CanonicalInventoryDigest != item.Origin.CanonicalInventoryDigest {
		return invalidSubmission("task differs from accepted source, scope, role or dependency closure")
	}
	return nil
}
