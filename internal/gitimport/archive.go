package gitimport

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/hex"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/go-git/go-git/v5/plumbing"
)

const maxEntries = 10000
const maxFileBytes = 4 << 20
const maxPathBytes = 512

type sourceFile struct {
	name string
	data []byte
}

func archiveSnapshot(ctx context.Context, objects map[plumbing.Hash]*gitObject, selected plumbing.Hash) ([]byte, plumbing.Hash, error) {
	commit := selected
	for depth := 0; ; depth++ {
		object := objects[commit]
		if object == nil {
			return nil, plumbing.ZeroHash, ErrContent
		}
		if object.kind == plumbing.CommitObject {
			break
		}
		if object.kind != plumbing.TagObject || depth >= 8 {
			return nil, plumbing.ZeroHash, ErrContent
		}
		hash, err := headerHash(object.data, "object ")
		if err != nil {
			return nil, plumbing.ZeroHash, err
		}
		commit = hash
	}
	tree, err := headerHash(objects[commit].data, "tree ")
	if err != nil {
		return nil, plumbing.ZeroHash, err
	}
	files := make([]sourceFile, 0)
	seen := map[string]bool{}
	entries, total := 0, 0
	var walk func(plumbing.Hash, string) error
	walk = func(hash plumbing.Hash, prefix string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		object := objects[hash]
		if object == nil || object.kind != plumbing.TreeObject {
			return ErrContent
		}
		data := object.data
		for len(data) > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			entries++
			if entries > maxEntries {
				return ErrBudget
			}
			mode, tail, ok := bytes.Cut(data, []byte{' '})
			if !ok {
				return ErrContent
			}
			name, tail, ok := bytes.Cut(tail, []byte{0})
			if !ok || len(tail) < 20 {
				return ErrContent
			}
			var child plumbing.Hash
			copy(child[:], tail[:20])
			data = tail[20:]
			component := string(name)
			if !safeComponent(component) {
				return ErrContent
			}
			path := prefix + component
			if len(path) > maxPathBytes {
				return ErrBudget
			}
			if seen[path] {
				return ErrContent
			}
			seen[path] = true
			switch string(mode) {
			case "40000":
				if err := walk(child, path+"/"); err != nil {
					return err
				}
			case "100644", "100755":
				blob := objects[child]
				if blob == nil || blob.kind != plumbing.BlobObject {
					return ErrContent
				}
				if len(blob.data) > maxFileBytes || len(blob.data) > MaxArchiveBytes-total {
					return ErrBudget
				}
				if bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\n")) || bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\r\n")) {
					return ErrContent
				}
				total += len(blob.data)
				files = append(files, sourceFile{name: path, data: blob.data})
			default:
				return ErrContent
			}
		}
		return nil
	}
	if err := walk(tree, ""); err != nil {
		return nil, plumbing.ZeroHash, err
	}
	if len(files) == 0 {
		return nil, plumbing.ZeroHash, ErrContent
	}
	sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
	var buffer bytes.Buffer
	writer := zip.NewWriter(&archiveWriter{ctx: ctx, buffer: &buffer})
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return nil, plumbing.ZeroHash, err
		}
		header := &zip.FileHeader{Name: file.name, Method: zip.Deflate, Modified: time.Date(1980, 1, 1, 0, 0, 0, 0, time.UTC)}
		header.SetMode(0644)
		entry, err := writer.CreateHeader(header)
		if err != nil {
			return nil, plumbing.ZeroHash, err
		}
		// Chunking ensures cancellation is observed even for compressible content.
		for data := file.data; len(data) > 0; {
			if err := ctx.Err(); err != nil {
				return nil, plumbing.ZeroHash, err
			}
			n := min(len(data), 32<<10)
			if _, err := entry.Write(data[:n]); err != nil {
				return nil, plumbing.ZeroHash, err
			}
			data = data[n:]
		}
	}
	if err := writer.Close(); err != nil {
		return nil, plumbing.ZeroHash, err
	}
	return buffer.Bytes(), commit, nil
}

func headerHash(data []byte, prefix string) (plumbing.Hash, error) {
	line, _, ok := bytes.Cut(data, []byte{'\n'})
	if !ok || !bytes.HasPrefix(line, []byte(prefix)) || len(line) != len(prefix)+40 {
		return plumbing.ZeroHash, ErrContent
	}
	var hash plumbing.Hash
	if _, err := hex.Decode(hash[:], line[len(prefix):]); err != nil {
		return plumbing.ZeroHash, ErrContent
	}
	return hash, nil
}
func safeComponent(name string) bool {
	if name == "" || name == "." || name == ".." || strings.EqualFold(name, ".git") || !utf8.ValidString(name) || strings.ContainsAny(name, "/:\\\x00") {
		return false
	}
	for _, r := range name {
		if r < 32 || r == 127 {
			return false
		}
	}
	return true
}

type archiveWriter struct {
	ctx    context.Context
	buffer *bytes.Buffer
}

func (w *archiveWriter) Write(p []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	if len(p) > MaxArchiveBytes-w.buffer.Len() {
		return 0, ErrBudget
	}
	return w.buffer.Write(p)
}
