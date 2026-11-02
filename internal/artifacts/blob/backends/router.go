package backends

import (
	"context"
	"errors"
	"io"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
	"github.com/grauwolf32/contractor/internal/artifacts/blob/backends/filesystem"
	inline "github.com/grauwolf32/contractor/internal/artifacts/blob/backends/postgres"
)

// Open chooses new-write storage once per process startup. Existing
// objects are always routed by their own backend, independently of that choice.
// Callers close the returned io.Closer after all request/background work drains.
// The returned backend describes large writes in auto mode; small writes stay
// inline. In auto mode a failed file preparation may use PostgreSQL before any
// registry publication; a failed or ambiguous registry write is never repeated.
func Open(ctx context.Context, mode blob.Backend, path string, inlineMaxBytes int64) (blob.Backend, blob.Storage, error) {
	mode, err := blob.ValidateConfig(string(mode), path)
	if err != nil {
		return "", nil, err
	}
	if err := blob.ValidateInlineLimit(inlineMaxBytes); err != nil {
		return "", nil, err
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	selected := blob.Postgres
	var writer blob.Storage = inline.Store{}
	var files *filesystem.Store
	if mode != blob.Postgres && path != "" {
		files, err = filesystem.Open(ctx, path)
		if err == nil {
			selected, writer = blob.Filesystem, files
		} else if ctx.Err() != nil {
			return "", nil, ctx.Err()
		} else if mode == blob.Filesystem || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return "", nil, err
		}
	}
	fileError := blob.ErrIO
	if files == nil && path != "" {
		// PostgreSQL writes can coexist with readable files on a read-only
		// volume. Opening existing storage does not create or probe it.
		files, err = filesystem.OpenExisting(ctx, path)
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
		inlineMaxBytes: inlineMaxBytes, tiered: mode == blob.Auto,
	}, nil
}

type routedBlobStore struct {
	writer         blob.Storage
	files          *filesystem.Store
	fileError      error
	inlineMaxBytes int64
	tiered         bool
}

func (s *routedBlobStore) Store(ctx context.Context, data []byte) (blob.Object, error) {
	if s.tiered && int64(len(data)) <= s.inlineMaxBytes {
		return (inline.Store{}).Store(ctx, data)
	}
	object, err := s.writer.Store(ctx, data)
	if err != nil && s.tiered && ctx.Err() == nil &&
		(errors.Is(err, blob.ErrIO) || errors.Is(err, blob.ErrMissing) || errors.Is(err, io.ErrShortWrite)) {
		return (inline.Store{}).Store(ctx, data)
	}
	return object, err
}

func (s *routedBlobStore) reader(object blob.Object) (blob.Storage, error) {
	switch object.Backend {
	case blob.Postgres:
		return inline.Store{}, nil
	case blob.Filesystem:
		if s.files != nil {
			return s.files, nil
		}
		return nil, s.fileError
	default:
		return nil, blob.ErrIntegrity
	}
}

func (s *routedBlobStore) Read(ctx context.Context, object blob.Object) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	store, err := s.reader(object)
	if err != nil {
		return nil, err
	}
	return store.Read(ctx, object)
}

func (s *routedBlobStore) Check(ctx context.Context, object blob.Object) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	store, err := s.reader(object)
	if err != nil {
		return err
	}
	return store.Check(ctx, object)
}

func (s *routedBlobStore) Delete(ctx context.Context, object blob.Object) error {
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
