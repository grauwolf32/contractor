package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Instructions returns the exact text dependency pinned by a normalized ref,
// so conformance tests can verify packaged instruction sources; production
// execution reads the text pinned into resolved Workflows.
func (s *Snapshot) Instructions(raw string) (contracts.ResolvedInstructions, error) {
	normalized, err := validateInstructionRef(raw)
	if err != nil {
		return contracts.ResolvedInstructions{}, err
	}
	instructions, ok := s.instructions[normalized]
	if !ok {
		return contracts.ResolvedInstructions{}, fmt.Errorf("unknown resolved instruction %q", normalized)
	}
	return instructions, nil
}
