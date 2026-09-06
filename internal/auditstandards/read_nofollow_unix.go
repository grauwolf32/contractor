//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package auditstandards

import (
	"fmt"
	"io"

	"golang.org/x/sys/unix"
)

func readSourceManifestNoFollow(rootPath string) ([]byte, error) {
	rootFD, err := unix.Open(rootPath, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	defer unix.Close(rootFD)
	manifestFD, err := unix.Openat(rootFD, ManifestPath, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	defer unix.Close(manifestFD)
	var stat unix.Stat_t
	if err := unix.Fstat(manifestFD, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG {
		return nil, fmt.Errorf("Audit standard source manifest is not regular")
	}
	if stat.Size > MaximumManifestBytes {
		return nil, errSourceManifestLimit
	}
	return io.ReadAll(io.NewSectionReader(manifestFDReaderAt(manifestFD), 0, stat.Size))
}

type manifestFDReaderAt int

func (fd manifestFDReaderAt) ReadAt(buffer []byte, offset int64) (int, error) {
	count, err := unix.Pread(int(fd), buffer, offset)
	if count == 0 && err == nil {
		return 0, io.EOF
	}
	return count, err
}
