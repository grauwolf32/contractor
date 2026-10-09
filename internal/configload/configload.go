// Package configload is the only way the product loads an operator
// configuration root. It composes the catalog loader in internal/config with
// the bundled skill and Audit standard packages that live beside the catalog
// subtrees, so every load site rejects the same invalid roots while
// internal/config stays ignorant of those formats.
//
// The functions mirror the internal/config entry points they wrap. Calling
// those directly skips the bundle checks; a boundary test in this package
// keeps every caller outside internal/config on this path.
package configload

import (
	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
)

// OperatorRootChecks returns the bundle validators every load applies to an
// operator root, in the order their failures are reported. Each check
// discovers and fully validates the bundled packages without writing them.
func OperatorRootChecks() []config.OperatorRootCheck {
	return []config.OperatorRootCheck{
		{Subject: "bundled skills", Check: func(root string) error {
			_, err := agentskills.DiscoverBundled(root)
			return err
		}},
		{Subject: "bundled Audit standards", Check: func(root string) error {
			_, err := auditstandards.DiscoverBundled(root)
			return err
		}},
	}
}

// Load validates one operator root: its bundles, then its catalog.
func Load(root string, descriptors config.Descriptors) (*config.Snapshot, error) {
	return config.Load(root, descriptors, OperatorRootChecks()...)
}

// LoadUnionReadOnly is offline validation of the Server's operator and managed
// roots; see config.LoadUnionReadOnly.
func LoadUnionReadOnly(operatorRoot, managedRoot string, descriptors config.Descriptors) (*config.Snapshot, error) {
	return config.LoadUnionReadOnly(operatorRoot, managedRoot, descriptors, OperatorRootChecks()...)
}

// NewManager starts the Server's configuration manager. The bundle checks run
// at startup and again on every reload before a managed publication, ahead of
// any checks the caller adds.
func NewManager(options config.ManagerOptions) (*config.Manager, error) {
	options.OperatorRootChecks = append(OperatorRootChecks(), options.OperatorRootChecks...)
	return config.NewManager(options)
}
