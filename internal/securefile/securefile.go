// Package securefile reads small secret files that must be regular,
// non-symlinked, and closed to group and other access.
package securefile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

// OwnerOnly forbids every group and other permission bit.
const OwnerOnly os.FileMode = 0o077

// Each error names the check that rejected the file so callers can keep their
// own error identities and messages.
var (
	// ErrPath reports a path that is empty, relative, or not clean.
	ErrPath = errors.New("secure file path must be clean and absolute")
	// ErrUnsafePath reports a path that is missing, a symlink, not a regular
	// file, or has forbidden permission bits.
	ErrUnsafePath = errors.New("secure file must be regular, non-symlinked, and permission-restricted")
	// ErrOpen reports a failure to open the path without following symlinks.
	ErrOpen = errors.New("open secure file")
	// ErrUnsafeHandle reports an opened file that is not regular, has
	// forbidden permission bits, or is empty or larger than the size bound.
	ErrUnsafeHandle = errors.New("secure file is unsafe or outside its size bound")
	// ErrRead reports a failed, empty, or oversized bounded read.
	ErrRead = errors.New("read bounded secure file")
)

// Read returns the contents of an owner-only file of 1 to maxBytes bytes.
func Read(path string, maxBytes int64) ([]byte, error) {
	return ReadRestricted(path, maxBytes, OwnerOnly)
}

// ReadRestricted returns the contents of a file of 1 to maxBytes bytes whose
// permissions have none of the forbidden bits. The path is checked before
// opening and the open handle is checked again, so a swapped file is refused.
// The caller owns the returned bytes and should clear them after use.
func ReadRestricted(path string, maxBytes int64, forbidden os.FileMode) ([]byte, error) {
	if strings.TrimSpace(path) == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrPath
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() ||
		info.Mode().Perm()&forbidden != 0 {
		return nil, ErrUnsafePath
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, ErrOpen
	}
	handle := os.NewFile(uintptr(fd), "secure-file")
	defer handle.Close()
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil || stat.Mode&unix.S_IFMT != unix.S_IFREG ||
		os.FileMode(stat.Mode).Perm()&forbidden != 0 || stat.Size < 1 || stat.Size > maxBytes {
		return nil, ErrUnsafeHandle
	}
	data, err := io.ReadAll(io.LimitReader(handle, maxBytes+1))
	if err != nil || len(data) == 0 || int64(len(data)) > maxBytes {
		clear(data)
		return nil, ErrRead
	}
	return data, nil
}
