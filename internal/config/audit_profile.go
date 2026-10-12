package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

type auditProfileDocument struct {
	APIVersion string             `yaml:"apiVersion"`
	Kind       string             `yaml:"kind"`
	Metadata   *metadataSource    `yaml:"metadata"`
	Spec       *auditprofile.Spec `yaml:"spec"`
}

func (l *loader) loadAuditProfiles() error {
	files, err := l.discover("audit-profiles")
	if err != nil {
		return err
	}
	for _, file := range files {
		if file.source != ConfigurationSourceOperator {
			return fmt.Errorf("%s: AuditProfile must be operator-authored", file.relative)
		}
		document, decodeErr := decodeOne[auditProfileDocument](file)
		if decodeErr != nil {
			return decodeErr
		}
		selector, resolveErr := validateEnvelope(
			document.APIVersion, document.Kind, auditprofile.Kind, document.Metadata,
		)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		if _, exists := l.auditProfiles[selector.String()]; exists {
			return fmt.Errorf("%s: duplicate AuditProfile identity %s", file.relative, selector)
		}
		profile, resolveErr := auditprofile.Resolve(
			selector, document.Spec,
			func(ref resolved.Selector) (resolved.ResolvedWorkflow, bool) {
				workflow, ok := l.workflows[ref.String()]
				return workflow, ok
			},
		)
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.auditProfiles[selector.String()] = profile
	}
	return nil
}
