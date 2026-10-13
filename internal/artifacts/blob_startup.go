package artifacts

import (
	"context"
	"errors"
	"io"
)

// OpenBlobStore chooses new-write storage once per process startup. Existing
// objects are always routed by their own backend, independently of that choice.
// Callers close the returned io.Closer after all request/background work drains.
// The returned backend describes large writes in auto mode; small writes stay
// inline. In auto mode a failed file preparation may use PostgreSQL before any
// registry publication; a failed or ambiguous registry write is never repeated.
func OpenBlobStore(ctx context.Context, mode BlobBackend, path string, inlineMaxBytes int64) (BlobBackend, BlobStore, error) {
	mode, err := ValidateBlobConfig(string(mode), path)
	if err != nil {
		return "", nil, err
	}
	if err := ValidateBlobInlineLimit(inlineMaxBytes); err != nil {
		return "", nil, err
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	selected := BlobPostgres
	var writer BlobStore = PostgresBlobStore{}
	var files *FilesystemBlobStore
	if mode != BlobPostgres && path != "" {
		files, err = OpenFilesystemBlobStore(ctx, path)
		if err == nil {
			selected, writer = BlobFilesystem, files
		} else if ctx.Err() != nil {
			return "", nil, ctx.Err()
		} else if mode == BlobFilesystem || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return "", nil, err
		}
	}
	fileError := ErrBlobIO
	if files == nil && path != "" {
		// PostgreSQL writes can coexist with readable files on a read-only
		// volume. Opening existing storage does not create or probe it.
		files, err = openFilesystemBlobRoot(ctx, path, false)
		if err != nil {
			if ctx.Err() != nil {
				return "", nil, ctx.Err()
			}
			fileError = err
		}
	}
	if err := ctx.Err(); err != nil {
		if files != nil {
			_ = files.Close()
		}
		return "", nil, err
	}
	return selected, &routedBlobStore{
		writer: writer, files: files, fileError: fileError,
		inlineMaxBytes: inlineMaxBytes, tiered: mode == BlobAuto,
	}, nil
}

type routedBlobStore struct {
	writer         BlobStore
	files          *FilesystemBlobStore
	fileError      error
	inlineMaxBytes int64
	tiered         bool
}

func (s *routedBlobStore) preparesFile(size int) bool {
	_, filesystem := s.writer.(*FilesystemBlobStore)
	return filesystem && (!s.tiered || int64(size) > s.inlineMaxBytes)
}

func (s *routedBlobStore) Store(ctx context.Context, data []byte) (BlobObject, error) {
	if s.tiered && int64(len(data)) <= s.inlineMaxBytes {
		return (PostgresBlobStore{}).Store(ctx, data)
	}
	object, err := s.writer.Store(ctx, data)
	if err != nil && s.tiered && ctx.Err() == nil &&
		(errors.Is(err, ErrBlobIO) || errors.Is(err, ErrBlobMissing) || errors.Is(err, io.ErrShortWrite)) {
		return (PostgresBlobStore{}).Store(ctx, data)
	}
	return object, err
}

func (s *routedBlobStore) reader(object BlobObject) (BlobStore, error) {
	switch object.Backend {
	case BlobPostgres:
		return PostgresBlobStore{}, nil
	case BlobFilesystem:
		if s.files != nil {
			return s.files, nil
		}
		return nil, s.fileError
	default:
		return nil, ErrArtifactIntegrity
	}
}

func (s *routedBlobStore) Read(ctx context.Context, object BlobObject) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	store, err := s.reader(object)
	if err != nil {
		return nil, err
	}
	return store.Read(ctx, object)
}

func (s *routedBlobStore) Check(ctx context.Context, object BlobObject) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	store, err := s.reader(object)
	if err != nil {
		return err
	}
	return store.Check(ctx, object)
}

func (s *routedBlobStore) Delete(ctx context.Context, object BlobObject) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	store, err := s.reader(object)
	if err != nil {
		return err
	}
	return store.Delete(ctx, object)
}

func (s *routedBlobStore) Close() error {
	if s.files != nil {
		return s.files.Close()
	}
	return nil
}
