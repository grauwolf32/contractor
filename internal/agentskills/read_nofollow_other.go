//go:build !aix && !darwin && !dragonfly && !freebsd && !linux && !netbsd && !openbsd && !solaris

package agentskills

import (
	"fmt"
	"os"
)

func readSourceFileNoFollow(rootPath, portablePath string, limit int64) ([]byte, error) {
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	info, err := root.Lstat(portablePath)
	if err != nil || !info.Mode().IsRegular() {
		return nil, fmt.Errorf("source member is not regular")
	}
	if info.Size() > limit {
		return nil, errSourceLimit
	}
	data, err := root.ReadFile(portablePath)
	if err != nil {
		return nil, err
	}
	after, err := root.Lstat(portablePath)
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(info, after) {
		return nil, fmt.Errorf("source member changed while reading")
	}
	return data, nil
}
