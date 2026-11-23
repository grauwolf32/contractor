package auditservice

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstandards"
)

func (s *Service) Standards(ctx context.Context, ownerID string) ([]auditstandards.PackageProjection, error) {
	packages, err := s.standards.List(ctx, ownerID)
	if err != nil {
		return nil, err
	}
	result := make([]auditstandards.PackageProjection, len(packages))
	for index, pkg := range packages {
		result[index] = auditstandards.Projection(pkg, false)
	}
	return result, nil
}

func (s *Service) Standard(
	ctx context.Context, ownerID string, ref auditstandards.Reference,
) (auditstandards.PackageProjection, error) {
	pkg, err := s.standards.Resolve(ctx, ownerID, ref)
	if err != nil {
		return auditstandards.PackageProjection{}, err
	}
	return auditstandards.Projection(pkg, true), nil
}
