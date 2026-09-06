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
// delta programs/results. No thin packs are requested or accepted.
func decodePack(ctx context.Context, raw []byte) (map[plumbing.Hash]*gitObject, error) {
	if len(raw) > MaxReceivedBytes {
		return nil, ErrBudget
	}
	if len(raw) < 32 {
		return nil, ErrContent
	}
	scanner := packfile.NewScanner(bytes.NewReader(raw))
	version, count, err := scanner.Header()
	if err != nil || (version != 2 && version != 3) || count == 0 {
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
			return nil, ErrContent
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
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, ErrContent
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
	if err != nil || !bytes.Equal(checksum[:], raw[len(raw)-20:]) {
		return nil, ErrContent
	}
	// Reading another header must hit EOF; reject concatenated/trailing data.
	if _, err := scanner.NextObjectHeader(); err != io.EOF {
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
