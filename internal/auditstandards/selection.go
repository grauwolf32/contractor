package auditstandards

// SelectEntries validates an explicit finite selection, preserving the exact
// package source order. An absent selection means every packaged entry;
// missing mappings never remove a requirement from this denominator.
func SelectEntries(pkg Package, ids []string, maximum int) ([]Entry, error) {
	if maximum < 1 || maximum > MaximumEntries || len(ids) > maximum {
		return nil, validationError(CodeLimitExceeded, ManifestPath)
	}
	wanted := map[string]bool{}
	for _, id := range ids {
		if wanted[id] {
			return nil, validationError(CodeManifestInvalid, ManifestPath)
		}
		wanted[id] = true
	}
	selected := []Entry{}
	for _, entry := range pkg.Document.Entries {
		if len(wanted) > 0 && !wanted[entry.ID] {
			continue
		}
		if len(selected) == maximum {
			return nil, validationError(CodeLimitExceeded, ManifestPath)
		}
		selected = append(selected, entry)
	}
	if len(selected) == 0 || len(wanted) > 0 && len(selected) != len(wanted) {
		return nil, validationError(CodeManifestInvalid, ManifestPath)
	}
	return selected, nil
}
