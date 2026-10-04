// Package sourcezip encodes bounded, deterministic source archives shared by
// Git imports and local directory bundles. Callers validate their own source
// paths and file handles before handing members to this writer.
package sourcezip

import (
	"archive/zip"
	"bytes"
	"compress/flate"
	"context"
	"errors"
	"fmt"
	"io"
	"time"
)

const (
	MaxArchiveBytes = 64 << 20
	MaxEntries      = 10000
	MaxFileBytes    = 4 << 20
	MaxPathBytes    = 512
)

var (
	ErrLimit      = errors.New("source ZIP exceeds its resource limit")
	ErrMemberSize = errors.New("source ZIP member size differs from its declaration")
)

type Member struct {
	Name  string
	Size  int64
	Write func(io.Writer) error
}

type Options struct {
	// CompressionLevel zero keeps archive/zip's default Deflate compressor.
	CompressionLevel int
	// ModifiedField preserves Git import's historical ZIP header encoding.
	// Local bundles use FileHeader.SetModTime instead.
	ModifiedField bool
	// MaxBytes defaults to MaxArchiveBytes. A smaller bound is useful in tests.
	MaxBytes int
}

// Encode writes members in the caller's order with fixed mode and timestamp.
// It checks the shared source limits against the declared sizes before
// reading any member, requires each member to write exactly its declared
// Size, bounds compressed bytes and observes ctx during compression and
// between members.
func Encode(ctx context.Context, members []Member, options Options) ([]byte, error) {
	maximum := options.MaxBytes
	if maximum == 0 {
		maximum = MaxArchiveBytes
	}
	if maximum < 0 || maximum > MaxArchiveBytes || len(members) == 0 || len(members) > MaxEntries {
		return nil, ErrLimit
	}
	var expanded int64
	for _, member := range members {
		if member.Name == "" || len(member.Name) > MaxPathBytes ||
			member.Size < 0 || member.Size > MaxFileBytes ||
			member.Size > MaxArchiveBytes-expanded || member.Write == nil {
			return nil, ErrLimit
		}
		expanded += member.Size
	}
	var buffer bytes.Buffer
	writer := zip.NewWriter(&boundedWriter{ctx: ctx, buffer: &buffer, maximum: maximum})
	if options.CompressionLevel != 0 {
		writer.RegisterCompressor(zip.Deflate, func(destination io.Writer) (io.WriteCloser, error) {
			return flate.NewWriter(destination, options.CompressionLevel)
		})
	}
	fixedTime := time.Date(1980, time.January, 1, 0, 0, 0, 0, time.UTC)
	for _, member := range members {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		header := &zip.FileHeader{Name: member.Name, Method: zip.Deflate}
		header.SetMode(0o644)
		if options.ModifiedField {
			header.Modified = fixedTime
		} else {
			header.SetModTime(fixedTime)
		}
		entry, err := writer.CreateHeader(header)
		if err != nil {
			return nil, err
		}
		// A source that changed after its size was checked must not carry a
		// member past the limits validated above.
		counted := &memberWriter{writer: entry, name: member.Name, remaining: member.Size}
		if err := member.Write(counted); err != nil {
			return nil, err
		}
		if counted.remaining != 0 {
			return nil, fmt.Errorf("%w: %s", ErrMemberSize, member.Name)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}

// memberWriter rejects any write that would exceed a member's declared size.
type memberWriter struct {
	writer    io.Writer
	name      string
	remaining int64
}

func (w *memberWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, fmt.Errorf("%w: %s", ErrMemberSize, w.name)
	}
	n, err := w.writer.Write(p)
	w.remaining -= int64(n)
	return n, err
}

type boundedWriter struct {
	ctx     context.Context
	buffer  *bytes.Buffer
	maximum int
}

func (w *boundedWriter) Write(p []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	if len(p) > w.maximum-w.buffer.Len() {
		return 0, ErrLimit
	}
	return w.buffer.Write(p)
}
