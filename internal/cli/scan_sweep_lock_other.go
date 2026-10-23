//go:build !unix

package cli

import "errors"

func lockSweepManifest(string) (func(), error) {
	return nil, errors.New("scan-sweep currently requires Unix advisory file locks")
}
