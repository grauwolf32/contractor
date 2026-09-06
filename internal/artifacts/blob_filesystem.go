package artifacts

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
)

var blobKeyPattern = regexp.MustCompile(`^[0-9a-f]{2}/[0-9a-f]{32}$`)
var stagingKeyPattern = regexp.MustCompile(`^\.staging-[0-9a-f]{32}$`)

// FilesystemBlobStore owns a descriptor rooted at the configured directory.
// Close is called only after all Server request/background work has drained.
type FilesystemBlobStore struct{ root *os.Root }

func OpenFilesystemBlobStore(ctx context.Context, path string) (*FilesystemBlobStore, error) {
	if _, err := ValidateBlobConfig(string(BlobFilesystem), path); err != nil {
		return nil, err
	}
	// Reject existing symlink components before creating a dedicated root.
	current := string(filepath.Separator)
	for _, component := range strings.Split(strings.TrimPrefix(path, current), string(filepath.Separator)) {
		current = filepath.Join(current, component)
		info, err := os.Lstat(current)
		if errors.Is(err, os.ErrNotExist) {
			break
		}
		if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, errors.New("invalid artifact blob directory")
		}
	}
	if err := os.MkdirAll(path, 0700); err != nil {
		return nil, errors.New("artifact blob directory is not writable")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, errors.New("cannot open artifact blob directory")
	}
	store := &FilesystemBlobStore{root: root}
	probe, err := store.Store(ctx, []byte("contractor blob probe"))
	if err == nil {
		_, err = store.Read(ctx, probe)
	}
	if err == nil {
		err = store.Delete(ctx, probe)
	}
	if err != nil {
		_ = root.Close()
		return nil, fmt.Errorf("artifact blob directory probe failed: %w", err)
	}
	return store, nil
}

func (s *FilesystemBlobStore) Close() error { return s.root.Close() }

func (s *FilesystemBlobStore) Store(ctx context.Context, data []byte) (result BlobObject, err error) {
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if len(data) > MaxPayloadSize {
		return result, ErrPayloadTooLarge
	}
	id, err := randomOpaqueID("")
	if err != nil {
		return result, err
	}
	shard := id[:2]
	if err := s.root.Mkdir(shard, 0700); err != nil && !errors.Is(err, os.ErrExist) {
		return result, blobIOError(err)
	}
	if err := s.checkDirectory(shard); err != nil {
		return result, err
	}
	staging, key := ".staging-"+id, shard+"/"+id
	f, err := s.root.OpenFile(staging, os.O_CREATE|os.O_EXCL|os.O_WRONLY|syscall.O_NOFOLLOW, 0600)
	if err != nil {
		return result, blobIOError(err)
	}
	defer func() { _ = f.Close(); _ = s.root.Remove(staging) }()
	hash := sha256.New()
	for offset := 0; offset < len(data); {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		end := min(offset+64*1024, len(data))
		n, err := f.Write(data[offset:end])
		if err != nil {
			return result, blobIOError(err)
		}
		if n != end-offset {
			return result, io.ErrShortWrite
		}
		_, _ = hash.Write(data[offset:end])
		offset = end
	}
	if err := f.Close(); err != nil {
		return result, blobIOError(err)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	// A hard link publishes a complete inode without overwriting an existing key.
	// Both names are rooted on the same filesystem; the staging name is removed.
	if err := s.root.Link(staging, key); err != nil {
		return result, blobIOError(err)
	}
	return BlobObject{Backend: BlobFilesystem, Key: key, Digest: hash.Sum(nil), Size: int64(len(data))}, nil
}

func (s *FilesystemBlobStore) Read(ctx context.Context, object BlobObject) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if object.Backend != BlobFilesystem || !blobKeyPattern.MatchString(object.Key) || object.Size < 0 || object.Size > MaxPayloadSize {
		return nil, ErrArtifactIntegrity
	}
	if err := s.checkDirectory(object.Key[:2]); err != nil {
		return nil, err
	}
	info, err := s.root.Lstat(object.Key)
	if err != nil {
		return nil, blobIOError(err)
	}
	if !info.Mode().IsRegular() || info.Size() != object.Size {
		return nil, ErrArtifactIntegrity
	}
	f, err := s.root.OpenFile(object.Key, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, blobIOError(err)
	}
	defer f.Close()
	actual, err := f.Stat()
	if err != nil || !actual.Mode().IsRegular() || actual.Size() != object.Size {
		return nil, ErrArtifactIntegrity
	}
	data := make([]byte, int(object.Size))
	for offset := 0; offset < len(data); {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(offset+64*1024, len(data))
		n, err := io.ReadFull(f, data[offset:end])
		if err != nil {
			return nil, ErrArtifactIntegrity
		}
		offset += n
	}
	var extra [1]byte
	if n, err := f.Read(extra[:]); n != 0 || err != io.EOF {
		return nil, ErrArtifactIntegrity
	}
	if err := verifyBlob(object, data); err != nil {
		return nil, err
	}
	return data, nil
}

func (s *FilesystemBlobStore) Delete(ctx context.Context, object BlobObject) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if object.Backend != BlobFilesystem || !blobKeyPattern.MatchString(object.Key) {
		return ErrArtifactIntegrity
	}
	if err := s.checkDirectory(object.Key[:2]); err != nil {
		if errors.Is(err, ErrBlobMissing) {
			return nil
		}
		return err
	}
	info, err := s.root.Lstat(object.Key)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return blobIOError(err)
	}
	if !info.Mode().IsRegular() {
		return ErrArtifactIntegrity
	}
	if err := s.root.Remove(object.Key); err != nil && !errors.Is(err, os.ErrNotExist) {
		return blobIOError(err)
	}
	return nil
}

func (s *FilesystemBlobStore) checkDirectory(name string) error {
	info, err := s.root.Lstat(name)
	if err != nil {
		return blobIOError(err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return ErrArtifactIntegrity
	}
	return nil
}

func blobIOError(err error) error {
	if errors.Is(err, os.ErrNotExist) {
		return ErrBlobMissing
	}
	return errors.New("artifact blob storage I/O failed")
}
