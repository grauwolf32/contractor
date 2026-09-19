package auditservice

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
)

// ValidateInputPreview reuses the ordinary Audit inventory and compatibility
// validators without creating an Audit, retaining artifacts, or running a target.
// The caller must authenticate and verify exact input bytes before invoking it.
func ValidateInputPreview(profile config.ResolvedAuditProfile, scope Scope, inputs map[string]artifacts.ReadResult, standards []auditstandards.ResolvedPackage) error {
	if !ProfileCompatibility(profile).ServerCompatible {
		return unsupported(ProfileCompatibility(profile).Reasons)
	}
	for name, slot := range profile.Inputs {
		input, ok := inputs[name]
		if slot.Required && !ok {
			return fmt.Errorf("%w: required Audit input is missing", ErrInvalid)
		}
		if ok && !acceptsMediaType(slot.MediaTypes, input.Payload.MediaType) {
			return fmt.Errorf("%w: Audit input media type is incompatible", ErrInvalid)
		}
	}
	for name := range inputs {
		if _, ok := profile.Inputs[name]; !ok {
			return fmt.Errorf("%w: unknown Audit input", ErrInvalid)
		}
	}
	inventory, err := buildInventory(profile, DraftSelection{Scope: scope}, inputs, standards)
	if err != nil {
		return err
	}
	if reasons := InventoryCompatibility(inventory); len(reasons) != 0 {
		return unsupported(reasons)
	}
	if len(inventory.Worklist.Items) > profile.Execution.MaxItemsPerRound || len(inventory.Worklist.Items) > profile.Execution.MaxItemsTotal {
		return fmt.Errorf("%w: Audit inventory exceeds profile limits", ErrInvalid)
	}
	return nil
}
