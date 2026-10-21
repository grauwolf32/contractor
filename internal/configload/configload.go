// Package configload is the only way the product loads an operator
// configuration root. It composes the catalog loader in internal/config with
// the bundled skill and Audit standard packages that live beside the catalog
// subtrees, so every load site rejects the same invalid roots while
// internal/config stays ignorant of those formats.
//
// The functions mirror the internal/config entry points they wrap. Calling
// those directly skips the bundle checks; a boundary test in this package
// keeps every caller outside internal/config on this path, and keeps bundle
// discovery here, so the Server walks its operator root once at startup.
package configload

import (
	"sync"

	"github.com/grauwolf32/contractor/internal/agentskills"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/config"
)

// Bundles are the bundled skill and Audit standard packages one load
// discovered in the operator root and fully validated, ready to seed.
type Bundles struct {
	Skills    *agentskills.SeedPlan
	Standards *auditstandards.SeedPlan
}

// bundleChecks are the bundle validators a load applies to the operator root,
// in the order their failures are reported. Each check discovers and fully
// validates the bundled packages without writing them. Until take is called
// the discovered plans are kept, so a load hands back what it validated rather
// than its caller walking the root again; later reloads only validate.
type bundleChecks struct {
	mu        sync.Mutex
	recording bool
	bundles   Bundles
}

func newBundleChecks() *bundleChecks { return &bundleChecks{recording: true} }

func (c *bundleChecks) operatorRootChecks() []config.OperatorRootCheck {
	return []config.OperatorRootCheck{
		{Subject: "bundled skills", Check: func(root string) error {
			plan, err := agentskills.DiscoverBundled(root)
			c.keep(func(bundles *Bundles) { bundles.Skills = plan })
			return err
		}},
		{Subject: "bundled Audit standards", Check: func(root string) error {
			plan, err := auditstandards.DiscoverBundled(root)
			c.keep(func(bundles *Bundles) { bundles.Standards = plan })
			return err
		}},
	}
}

func (c *bundleChecks) keep(set func(*Bundles)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.recording {
		set(&c.bundles)
	}
}

// take returns the kept bundles and stops keeping, so the checks of a
// long-lived Manager hold no package bytes after startup.
func (c *bundleChecks) take() Bundles {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.recording = false
	bundles := c.bundles
	c.bundles = Bundles{}
	return bundles
}

// Load validates one operator root: its bundles, then its catalog.
func Load(root string) (*config.Snapshot, error) {
	snapshot, err := config.Load(root, newBundleChecks().operatorRootChecks()...)
	if err != nil {
		return nil, err
	}
	if err := config.ValidateModelChoiceCatalog(root, snapshot); err != nil {
		return nil, err
	}
	return snapshot, nil
}

// LoadUnionReadOnly is offline validation of the Server's operator and managed
// roots; see config.LoadUnionReadOnly. It also returns the bundles it
// validated.
func LoadUnionReadOnly(operatorRoot, managedRoot string) (*config.Snapshot, Bundles, error) {
	checks := newBundleChecks()
	snapshot, err := config.LoadUnionReadOnly(operatorRoot, managedRoot, checks.operatorRootChecks()...)
	if err != nil {
		return nil, Bundles{}, err
	}
	if err := config.ValidateModelChoiceCatalog(operatorRoot, snapshot); err != nil {
		return nil, Bundles{}, err
	}
	return snapshot, checks.take(), nil
}

// NewManager starts the Server's configuration manager and returns the
// bundles its startup load validated, which the caller seeds without
// discovering the operator root again. The bundle checks run again on every
// reload before a managed publication, ahead of any checks the caller adds.
func NewManager(options config.ManagerOptions) (*config.Manager, Bundles, error) {
	checks := newBundleChecks()
	options.OperatorRootChecks = append(checks.operatorRootChecks(), options.OperatorRootChecks...)
	manager, err := config.NewManager(options)
	if err != nil {
		return nil, Bundles{}, err
	}
	if err := config.ValidateModelChoiceCatalog(options.OperatorRoot, manager.Snapshot()); err != nil {
		return nil, Bundles{}, err
	}
	return manager, checks.take(), nil
}
