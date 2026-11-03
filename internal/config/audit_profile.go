package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

type auditProfileDocument struct {
	APIVersion string          `yaml:"apiVersion"`
	Kind       string          `yaml:"kind"`
	Metadata   *metadataSource `yaml:"metadata"`
	Spec       map[string]any  `yaml:"spec"`
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
		data, resolveErr := strictjson.Canonical(document.Spec)
		if resolveErr != nil {
			return fmt.Errorf("%s: Audit program cannot be encoded", file.relative)
		}
		profile, resolveErr := auditprofile.PinProgram(selector, data, func(ref string) (resolved.ResolvedWorkflow, error) {
			workflow, exists := l.workflows[ref]
			if !exists {
				return resolved.ResolvedWorkflow{}, fmt.Errorf("selected Workflow is unavailable")
			}
			return workflow, nil
		})
		if resolveErr != nil {
			return fmt.Errorf("%s: %w", file.relative, resolveErr)
		}
		l.auditProfiles[selector.String()] = profile
	}
	return nil
}
