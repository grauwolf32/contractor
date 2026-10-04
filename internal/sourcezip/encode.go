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
	"io"
	"time"
)

const (
	MaxArchiveBytes = 64 << 20
	MaxEntries      = 10000
	MaxFileBytes    = 4 << 20
	MaxPathBytes    = 512
)

var ErrLimit = errors.New("source ZIP exceeds its resource limit")

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
// It checks the shared source limits before reading any member, then bounds
// compressed bytes and observes ctx during compression and between members.
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
		if err := member.Write(entry); err != nil {
			return nil, err
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
