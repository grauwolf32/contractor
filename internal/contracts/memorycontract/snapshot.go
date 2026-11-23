// Package memorycontract owns the pure Memory name and explicit export format.
package memorycontract

import (
	"regexp"
	"slices"
	"strings"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/contracts"
)

const (
	MaximumNoteNameBytes = 121
	maximumSnapshotNotes = 16
	MaximumSnapshotBytes = 512 << 10
)

var memoryNoteNamePattern = regexp.MustCompile(`^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`)

// ValidNoteName is the shared stored-note and explicit-selection grammar.
func ValidNoteName(name string) bool {
	return len(name) > 0 && len(name) <= MaximumNoteNameBytes && memoryNoteNamePattern.MatchString(name)
}

type SnapshotSelection struct {
	Namespace string
	Names     []string
}

// SnapshotOptions interprets the two existing string argument bindings.
// It never accepts a Run, Project, physical binding or revision selector.
func SnapshotOptions(arguments map[string]any) (SnapshotSelection, error) {
	namespace, ok := arguments["namespace"].(string)
	notes, namesOK := arguments["notes"].(string)
	if len(arguments) != 2 || !ok || !namesOK || !contracts.ValidName(namespace) || artifactpolicy.IsPurposeReservedNamespace(namespace) || len(notes) > maximumSnapshotNotes*(MaximumNoteNameBytes+1) {
		return SnapshotSelection{}, contracts.Invalidf("invalid Memory snapshot selection")
	}
	names := strings.Split(notes, ",")
	if len(names) > maximumSnapshotNotes {
		return SnapshotSelection{}, contracts.Invalidf("too many Memory snapshot notes")
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if !ValidNoteName(name) || seen[name] {
			return SnapshotSelection{}, contracts.Invalidf("invalid Memory snapshot note names")
		}
		seen[name] = true
	}
	slices.Sort(names)
	return SnapshotSelection{Namespace: namespace, Names: names}, nil
}
