package auditimport

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// This is the Runtime security-findings catalog. The parity test requires the
// two embedded copies to stay byte-identical when its version changes.
//
//go:embed cwe_catalog.json
var bundledCWECatalogJSON []byte

type cweCatalog struct {
	scheme      string
	version     string
	weaknessIDs map[string]struct{}
}

var bundledCWECatalog = sync.OnceValues(func() (cweCatalog, error) {
	var document struct {
		Scheme      string   `json:"scheme"`
		Version     string   `json:"version"`
		WeaknessIDs []string `json:"weakness_ids"`
	}
	if err := json.Unmarshal(bundledCWECatalogJSON, &document); err != nil {
		return cweCatalog{}, fmt.Errorf("decode bundled CWE catalog: %w", err)
	}
	if document.Scheme != "CWE" || document.Version == "" || len(document.WeaknessIDs) == 0 {
		return cweCatalog{}, errors.New("bundled CWE catalog identity is invalid")
	}
	result := cweCatalog{
		scheme: document.Scheme, version: document.Version,
		weaknessIDs: make(map[string]struct{}, len(document.WeaknessIDs)),
	}
	for _, id := range document.WeaknessIDs {
		if id == "" {
			return cweCatalog{}, errors.New("bundled CWE catalog slices.Contains an empty ID")
		}
		if _, exists := result.weaknessIDs[id]; exists {
			return cweCatalog{}, errors.New("bundled CWE catalog slices.Contains a duplicate ID")
		}
		result.weaknessIDs[id] = struct{}{}
	}
	return result, nil
})

func validateCWEClassification(reference auditdomain.StandardReference) error {
	catalog, err := bundledCWECatalog()
	if err != nil {
		return err
	}
	if reference.Scheme != catalog.scheme || reference.Version != catalog.version {
		return errors.New("proposal names an unsupported CWE catalog version")
	}
	if _, exists := catalog.weaknessIDs[reference.RequirementID]; !exists {
		return errors.New("proposal names an unknown CWE weakness")
	}
	return nil
}
