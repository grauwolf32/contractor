//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package agentskills

import (
	"fmt"
	"io"

	"golang.org/x/sys/unix"
)

func readSourceFileNoFollow(rootPath, portablePath string, limit int64) ([]byte, error) {
	rootFD, err := unix.Open(rootPath, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	defer unix.Close(rootFD)

	currentFD := rootFD
	ownedFD := -1
	parts := sourcePathParts(portablePath)
	for index, part := range parts {
		flags := unix.O_RDONLY | unix.O_CLOEXEC | unix.O_NOFOLLOW
		if index < len(parts)-1 {
			flags |= unix.O_DIRECTORY
		}
		nextFD, openErr := unix.Openat(currentFD, part, flags, 0)
		if ownedFD >= 0 {
			_ = unix.Close(ownedFD)
			ownedFD = -1
		}
		if openErr != nil {
			return nil, openErr
		}
		currentFD = nextFD
		ownedFD = nextFD
	}
	if ownedFD < 0 {
		return nil, fmt.Errorf("empty source path")
	}
	defer unix.Close(ownedFD)
	var stat unix.Stat_t
	if err := unix.Fstat(ownedFD, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG {
		return nil, fmt.Errorf("source member is not regular")
	}
	if stat.Size > limit {
		return nil, errSourceLimit
	}
	reader := io.NewSectionReader(fdReaderAt(ownedFD), 0, int64(stat.Size))
	return io.ReadAll(reader)
}

type fdReaderAt int

func (fd fdReaderAt) ReadAt(buffer []byte, offset int64) (int, error) {
	count, err := unix.Pread(int(fd), buffer, offset)
	if count == 0 && err == nil {
		return 0, io.EOF
	}
	return count, err
}
