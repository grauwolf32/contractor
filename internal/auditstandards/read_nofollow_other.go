//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package auditstandards

import (
	"fmt"
	"io"
	"os"
)

func readSourceManifestNoFollow(rootPath string) ([]byte, error) {
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	before, err := root.Lstat(ManifestPath)
	if err != nil || !before.Mode().IsRegular() || before.Size() > MaximumManifestBytes {
		if err == nil && before.Size() > MaximumManifestBytes {
			return nil, errSourceManifestLimit
		}
		return nil, fmt.Errorf("Audit standard source manifest is not regular")
	}
	file, err := root.Open(ManifestPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(before, opened) {
		return nil, fmt.Errorf("Audit standard source manifest changed before reading")
	}
	data, err := io.ReadAll(io.LimitReader(file, MaximumManifestBytes+1))
	if err != nil || len(data) > MaximumManifestBytes {
		if len(data) > MaximumManifestBytes {
			return nil, errSourceManifestLimit
		}
		return nil, err
	}
	after, err := root.Lstat(ManifestPath)
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) ||
		after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		return nil, fmt.Errorf("Audit standard source manifest changed while reading")
	}
	return data, nil
}
