package gitimport

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/packfile"
)

const maxObjects = 50000
const maxObjectBytes = 64 << 20
const maxDeltaDepth = 50

type gitObject struct {
	kind  plumbing.ObjectType
	data  []byte
	depth int
}

type packedObject struct {
	header packfile.ObjectHeader
	object *gitObject
	delta  []byte
}

// Limits apply before inflation or delta allocation, including intermediate
// delta programs/results. The pack is scanned directly from its source; only
// the last 20 bytes are retained to detect trailing data after the SHA-1.
// No thin packs are requested or accepted.
func decodePack(ctx context.Context, source io.Reader) (map[plumbing.Hash]*gitObject, error) {
	if source == nil {
		return nil, ErrContent
	}
	reader := &packStreamReader{ctx: ctx, source: source}
	contentError := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if reader.readError != nil {
			return reader.readError
		}
		return ErrContent
	}
	scanner := packfile.NewScanner(reader)
	version, count, err := scanner.Header()
	if err != nil {
		return nil, contentError()
	}
	if (version != 2 && version != 3) || count == 0 {
		return nil, ErrContent
	}
	if count > maxObjects {
		return nil, ErrBudget
	}
	objects := make(map[plumbing.Hash]*gitObject, count)
	byOffset := make(map[int64]*packedObject, count)
	pending := make([]*packedObject, 0)
	var decoded int64
	charge := func(n int64) error {
		if n < 0 || n > maxObjectBytes || n > MaxDecodedBytes-decoded {
			return ErrBudget
		}
		decoded += n
		return ctx.Err()
	}
	for range count {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		header, err := scanner.NextObjectHeader()
		if err != nil {
			return nil, contentError()
		}
		if err := charge(header.Length); err != nil {
			return nil, err
		}
		switch header.Type {
		case plumbing.CommitObject, plumbing.TreeObject, plumbing.BlobObject, plumbing.TagObject, plumbing.OFSDeltaObject, plumbing.REFDeltaObject:
		default:
			return nil, ErrContent
		}
		// A fixed backing array avoids geometric buffer growth beyond the budget.
		data := make([]byte, int(header.Length))
		sink := &objectWriter{ctx: ctx, data: data}
		n, _, err := scanner.NextObject(sink)
		if err != nil || n != header.Length {
			return nil, contentError()
		}
		record := &packedObject{header: *header}
		byOffset[header.Offset] = record
		if header.Type == plumbing.OFSDeltaObject || header.Type == plumbing.REFDeltaObject {
			if header.Type == plumbing.OFSDeltaObject && (header.OffsetReference >= header.Offset || byOffset[header.OffsetReference] == nil) {
				return nil, ErrContent
			}
			record.delta = data
			pending = append(pending, record)
		} else {
			record.object = &gitObject{kind: header.Type, data: data}
			objects[objectHash(record.object)] = record.object
		}
	}
	checksum, err := scanner.Checksum()
	if err != nil {
		return nil, contentError()
	}
	// Reading another header must hit EOF; reject concatenated/trailing data.
	if _, err := scanner.NextObjectHeader(); err != io.EOF {
		return nil, contentError()
	}
	// The scanner may have buffered trailing bytes. Also probe the underlying
	// stream in case it returned EOF without asking the source for more data.
	var extra [1]byte
	n, err := reader.Read(extra[:])
	if n != 0 || err != io.EOF {
		return nil, contentError()
	}
	if reader.readError != nil {
		return nil, contentError()
	}
	if reader.total < 32 || !bytes.Equal(checksum[:], reader.tail[:]) {
		return nil, ErrContent
	}
	for pass := 0; len(pending) > 0; pass++ {
		if pass > maxDeltaDepth {
			return nil, ErrBudget
		}
		next := pending[:0]
		for _, record := range pending {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			var base *gitObject
			if record.header.Type == plumbing.OFSDeltaObject {
				base = byOffset[record.header.OffsetReference].object
			} else {
				base = objects[record.header.Reference]
			}
			if base == nil {
				next = append(next, record)
				continue
			}
			if base.depth >= maxDeltaDepth {
				return nil, ErrBudget
			}
			baseSize, offset := binary.Uvarint(record.delta)
			if offset <= 0 || baseSize != uint64(len(base.data)) {
				return nil, ErrContent
			}
			size, n := binary.Uvarint(record.delta[offset:])
			if n <= 0 {
				return nil, ErrContent
			}
			if size > maxObjectBytes {
				return nil, ErrBudget
			}
			if err := charge(int64(size)); err != nil {
				return nil, err
			}
			data, err := packfile.PatchDelta(base.data, record.delta)
			if err != nil || uint64(len(data)) != size {
				return nil, ErrContent
			}
			record.delta = nil
			record.object = &gitObject{kind: base.kind, data: data, depth: base.depth + 1}
			objects[objectHash(record.object)] = record.object
		}
		if len(next) == len(pending) {
			return nil, ErrContent
		}
		pending = next
	}
	return objects, ctx.Err()
}

// packStreamReader bounds received bytes and remembers the trailer-sized tail.
// A non-EOF source error is retained so scanner failures can be distinguished
// from invalid Git content without exposing the transport error to the client.
type packStreamReader struct {
	ctx       context.Context
	source    io.Reader
	total     int64
	tail      [20]byte
	readError error
}

func (r *packStreamReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	if len(p) == 0 {
		return 0, nil
	}
	if r.total == MaxReceivedBytes {
		var probe [1]byte
		n, err := r.source.Read(probe[:])
		if n != 0 {
			r.readError = ErrBudget
			return 0, ErrBudget
		}
		if err != nil && err != io.EOF && r.readError == nil {
			r.readError = err
		}
		return 0, err
	}
	if int64(len(p)) > MaxReceivedBytes-r.total {
		p = p[:MaxReceivedBytes-r.total]
	}
	n, err := r.source.Read(p)
	if n > 0 {
		r.total += int64(n)
		if n >= len(r.tail) {
			copy(r.tail[:], p[n-len(r.tail):n])
		} else {
			copy(r.tail[:], r.tail[n:])
			copy(r.tail[len(r.tail)-n:], p[:n])
		}
	}
	if err != nil && err != io.EOF && r.readError == nil {
		r.readError = err
	}
	return n, err
}

func objectHash(object *gitObject) plumbing.Hash {
	hasher := plumbing.NewHasher(object.kind, int64(len(object.data)))
	_, _ = hasher.Write(object.data)
	return hasher.Sum()
}

type objectWriter struct {
	ctx    context.Context
	data   []byte
	offset int
}

func (w *objectWriter) Write(p []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	if len(p) > len(w.data)-w.offset {
		return 0, ErrContent
	}
	n := copy(w.data[w.offset:], p)
	w.offset += n
	return n, nil
}
