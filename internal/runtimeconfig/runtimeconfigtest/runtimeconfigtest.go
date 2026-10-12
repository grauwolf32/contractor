// Package runtimeconfigtest holds the RuntimeConfig fixtures that tests of
// other packages share. Production code never imports it.
package runtimeconfigtest

import "github.com/grauwolf32/contractor/internal/runtimeconfig"

// BuiltInRunSnapshot pins a Run to the built-in RuntimeConfig through the
// default label at binding revision 1 and pins no other label or credential,
// the snapshot a Run without Run labels pins on a freshly migrated database.
func BuiltInRunSnapshot() runtimeconfig.RunSnapshot {
	return runtimeconfig.RunSnapshot{
		Default: runtimeconfig.PinnedLabel{
			Label: runtimeconfig.DefaultLabel, BindingRevision: 1,
			Config: runtimeconfig.Ref{
				Name: runtimeconfig.BuiltInName, Version: runtimeconfig.BuiltInVersion,
				Digest: runtimeconfig.BuiltInDigest,
			},
		},
		Labels: []runtimeconfig.PinnedLabel{}, LLMCredentialIDs: []string{}, RuntimeCredentialIDs: []string{},
	}
}
