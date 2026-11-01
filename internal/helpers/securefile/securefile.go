// Package securefile reads files without following symlinks and within a size
// bound: small secret files that must be regular, owned by the process's
// effective user or root, and closed to group and other access (Read,
// ReadRestricted), and operator-provided files below a directory root
// (ReadBeneath).
package securefile

import (
	"errors"
	"fmt"
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
	// file, has forbidden permission bits, or is owned by another user.
	ErrUnsafePath = errors.New("secure file must be regular, non-symlinked, and permission-restricted")
	// ErrOpen reports a failure to open the path without following symlinks.
	ErrOpen = errors.New("open secure file")
	// ErrUnsafeHandle reports an opened file that is not regular, has
	// forbidden permission bits, is owned by another user, or is empty or
	// larger than the size bound.
	ErrUnsafeHandle = errors.New("secure file is unsafe or outside its size bound")
	// ErrRead reports a failed, empty, or oversized bounded read.
	ErrRead = errors.New("read bounded secure file")

	// ErrBeneathPath reports a path below a root that is empty, absolute, or
	// has an empty, "." or ".." component.
	ErrBeneathPath = errors.New("path below the root must be relative, slash-separated and clean")
	// ErrNotRegular reports a root or path below it that is a symlink or
	// passes through one, or a path that names no regular file. It carries no
	// operating-system error: the content changed, the read did not fail.
	ErrNotRegular = errors.New("root and path below it must reach a regular file without symlinks")
	// ErrTooLarge reports a file below a root that is larger than its size
	// bound, also when it grows while it is read.
	ErrTooLarge = errors.New("file below the root exceeds its size bound")
)

const beneathDirectoryFlags = unix.O_RDONLY | unix.O_DIRECTORY | unix.O_NOFOLLOW | unix.O_CLOEXEC

// ReadBeneath returns the contents of the regular file at relative, a
// slash-separated path below the directory root, when it has at most maxBytes
// bytes. root and every component of relative are opened without following a
// symlink, each component from its already opened parent, so a link swapped
// in after the caller's own checks is refused instead of followed. The final
// open does not wait for a FIFO writer. Unlike Read it applies no owner or
// permission rule and accepts an empty file: it reads operator-provided
// content, not secrets.
func ReadBeneath(root, relative string, maxBytes int64) ([]byte, error) {
	components := strings.Split(relative, "/")
	for _, component := range components {
		if component == "" || component == "." || component == ".." {
			return nil, ErrBeneathPath
		}
	}
	directory, err := unix.Open(root, beneathDirectoryFlags, 0)
	if err != nil {
		return nil, beneathOpenError("root", err)
	}
	for _, component := range components[:len(components)-1] {
		next, openErr := unix.Openat(directory, component, beneathDirectoryFlags, 0)
		_ = unix.Close(directory)
		if openErr != nil {
			return nil, beneathOpenError("directory below the root", openErr)
		}
		directory = next
	}
	fd, err := unix.Openat(
		directory, components[len(components)-1], unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC|unix.O_NONBLOCK, 0,
	)
	_ = unix.Close(directory)
	if err != nil {
		return nil, beneathOpenError("file below the root", err)
	}
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("stat file below the root: %w", err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG {
		_ = unix.Close(fd)
		return nil, ErrNotRegular
	}
	if stat.Size > maxBytes {
		_ = unix.Close(fd)
		return nil, ErrTooLarge
	}
	// A verified regular file never blocks; restore ordinary blocking reads.
	if err := unix.SetNonblock(fd, false); err != nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("open file below the root: %w", err)
	}
	file := os.NewFile(uintptr(fd), "file-below-root")
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read file below the root: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, ErrTooLarge
	}
	return data, nil
}

// beneathOpenError reports a symlink refused by O_NOFOLLOW (ELOOP, or ENOTDIR
// when a directory was expected) and a file where a directory was expected as
// ErrNotRegular. Every other open failure, such as a missing path or a denied
// permission, keeps its operating-system error.
func beneathOpenError(what string, err error) error {
	if errors.Is(err, unix.ELOOP) || errors.Is(err, unix.ENOTDIR) {
		return ErrNotRegular
	}
	return fmt.Errorf("open %s without following symlinks: %w", what, err)
}

// Read returns the contents of an owner-only file of 1 to maxBytes bytes.
func Read(path string, maxBytes int64) ([]byte, error) {
	return ReadRestricted(path, maxBytes, OwnerOnly)
}

// ReadRestricted returns the contents of a file of 1 to maxBytes bytes whose
// permissions have none of the forbidden bits. Permission bits restrict only
// group and other access, so the file must also be owned by the effective
// user or by root: another owner could rewrite it at will. The path is checked
// before opening and the open handle is checked again, so a swapped file is
// refused. The caller owns the returned bytes and should clear them after use.
func ReadRestricted(path string, maxBytes int64, forbidden os.FileMode) ([]byte, error) {
	if strings.TrimSpace(path) == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrPath
	}
	var stat unix.Stat_t
	if err := unix.Lstat(path, &stat); err != nil || !safeStat(stat, forbidden) {
		return nil, ErrUnsafePath
	}
	return readOpened(path, maxBytes, forbidden)
}

// readOpened opens path without following a symlink or waiting for a FIFO
// writer or device swapped in after the path check, then accepts only what
// the open handle itself reports.
func readOpened(path string, maxBytes int64, forbidden os.FileMode) ([]byte, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_CLOEXEC|unix.O_NONBLOCK, 0)
	if err != nil {
		return nil, ErrOpen
	}
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil || !safeStat(stat, forbidden) ||
		stat.Size < 1 || stat.Size > maxBytes {
		_ = unix.Close(fd)
		return nil, ErrUnsafeHandle
	}
	// A verified regular file never blocks; restore ordinary blocking reads.
	if err := unix.SetNonblock(fd, false); err != nil {
		_ = unix.Close(fd)
		return nil, ErrOpen
	}
	handle := os.NewFile(uintptr(fd), "secure-file")
	defer handle.Close()
	data, err := io.ReadAll(io.LimitReader(handle, maxBytes+1))
	if err != nil || len(data) == 0 || int64(len(data)) > maxBytes {
		clear(data)
		return nil, ErrRead
	}
	return data, nil
}

func safeStat(stat unix.Stat_t, forbidden os.FileMode) bool {
	return stat.Mode&unix.S_IFMT == unix.S_IFREG &&
		os.FileMode(stat.Mode).Perm()&forbidden == 0 &&
		(stat.Uid == uint32(os.Geteuid()) || stat.Uid == 0)
}
