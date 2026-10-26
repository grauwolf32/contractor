package auditservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditbaseline"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

type routedCheckBinding struct {
	route         auditdomain.AcceptedPlanRoute
	authorization auditdomain.AcceptedPlanAuthorization
}

// RoutedCheckContext preserves the service error contract around shared exact
// baseline/configuration pins also revalidated by the acceptance store.
type RoutedCheckContext struct{ shared *auditbaseline.CheckRoutes }

func NewRoutedCheckContext(audit auditstore.Audit, profile auditprofile.ResolvedAuditProfile, baseline BaselineSnapshot) (*RoutedCheckContext, error) {
	shared, err := auditbaseline.NewCheckRoutes(audit, profile, baseline)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return &RoutedCheckContext{shared: shared}, nil
}
func (c *RoutedCheckContext) selectBinding(plan auditdomain.CheckPlan) (routedCheckBinding, bool) {
	pins, ok := c.shared.Select(plan)
	return routedCheckBinding{route: pins.Route, authorization: pins.Authorization}, ok
}
func (c *RoutedCheckContext) build(source auditdomain.AcceptedPlanSource, plan auditdomain.CheckPlan, deps *auditdomain.RetainedDependencyManifest) (auditdomain.AcceptedPlan, error) {
	accepted, err := c.shared.Build(source, plan, deps)
	if err != nil {
		return auditdomain.AcceptedPlan{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return accepted, nil
}
func (c *RoutedCheckContext) Validate(plan auditdomain.AcceptedPlan, deps *auditdomain.RetainedDependencyManifest, workflowRole string) error {
	if err := c.shared.Validate(plan, deps, workflowRole); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	return nil
}
