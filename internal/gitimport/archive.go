package gitimport

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"io"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/sourcebundle"
	"github.com/grauwolf32/contractor/internal/sourcezip"
)

const (
	MaxPathBytes = sourcezip.MaxPathBytes
)

type sourceFile struct {
	name string
	data []byte
}

func archiveSnapshot(ctx context.Context, objects map[plumbing.Hash]*gitObject, selected plumbing.Hash, limits sourcezip.Limits, mode Mode) ([]byte, plumbing.Hash, *artifacts.GitAnalysis, error) {
	if err := limits.Validate(); err != nil {
		return nil, plumbing.ZeroHash, nil, ErrBudget
	}
	commit := selected
	for depth := 0; ; depth++ {
		object := objects[commit]
		if object == nil {
			return nil, plumbing.ZeroHash, nil, ErrContent
		}
		if object.kind == plumbing.CommitObject {
			break
		}
		if object.kind != plumbing.TagObject || depth >= 8 {
			return nil, plumbing.ZeroHash, nil, ErrContent
		}
		hash, err := headerHash(object.data, "object ")
		if err != nil {
			return nil, plumbing.ZeroHash, nil, err
		}
		commit = hash
	}
	tree, err := headerHash(objects[commit].data, "tree ")
	if err != nil {
		return nil, plumbing.ZeroHash, nil, err
	}
	report := analysisReport{Excluded: make([]exclusion, 0)}
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
			if entries > limits.MaxEntries {
				return ErrBudget
			}
			entryMode, tail, ok := bytes.Cut(data, []byte{' '})
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
			if len(path) > MaxPathBytes {
				return ErrBudget
			}
			if mode == ModeAnalysis && (path == artifacts.GitAnalysisReportPath || strings.HasPrefix(path, artifacts.GitAnalysisReportPath+"/")) {
				return ErrContent
			}
			if seen[path] {
				return ErrContent
			}
			seen[path] = true
			switch string(entryMode) {
			case "40000":
				if err := walk(child, path+"/"); err != nil {
					return err
				}
			case "100644", "100755":
				blob := objects[child]
				if blob == nil || blob.kind != plumbing.BlobObject {
					return ErrContent
				}
				if mode == ModeAnalysis {
					reason := ""
					switch {
					case bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\n")) || bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\r\n")):
						reason = "lfs_pointer"
					case !sourcebundle.RuntimeTextPath(path):
						reason = "not_source_path"
					case !utf8.Valid(blob.data):
						reason = "non_utf8"
					}
					if reason != "" {
						if err := report.exclude(path, reason); err != nil {
							return err
						}
						continue
					}
				}
				if int64(len(blob.data)) > limits.MaxFileBytes || int64(len(blob.data)) > limits.MaxExpandedBytes-int64(total) {
					return ErrBudget
				}
				if bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\n")) || bytes.HasPrefix(blob.data, []byte("version https://git-lfs.github.com/spec/v1\r\n")) {
					return &UnsupportedEntryError{Kind: "Git LFS pointer", Path: path}
				}
				total += len(blob.data)
				files = append(files, sourceFile{name: path, data: blob.data})
			case "120000":
				if mode == ModeAnalysis {
					if err := report.exclude(path, "symbolic_link"); err != nil {
						return err
					}
					continue
				}
				return &UnsupportedEntryError{Kind: "symbolic link", Path: path}
			case "160000":
				if mode == ModeAnalysis {
					if err := report.exclude(path, "submodule"); err != nil {
						return err
					}
					continue
				}
				return &UnsupportedEntryError{Kind: "submodule", Path: path}
			default:
				return ErrContent
			}
		}
		return nil
	}
	if err := walk(tree, ""); err != nil {
		return nil, plumbing.ZeroHash, nil, err
	}
	if len(files) == 0 {
		return nil, plumbing.ZeroHash, nil, ErrContent
	}
	readable := false
	for _, file := range files {
		readable = readable || sourcebundle.RuntimeTextPath(file.name) && utf8.Valid(file.data)
	}
	if !readable {
		return nil, plumbing.ZeroHash, nil, ErrContent
	}
	var analysis *artifacts.GitAnalysis
	if mode == ModeAnalysis {
		data, summary, err := report.encode(commit.String(), len(files))
		if err != nil {
			return nil, plumbing.ZeroHash, nil, err
		}
		analysis = summary
		files = append(files, sourceFile{name: artifacts.GitAnalysisReportPath, data: data})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].name < files[j].name })
	members := make([]sourcezip.Member, len(files))
	for index, file := range files {
		members[index] = sourcezip.Member{
			Name: file.name, Size: int64(len(file.data)),
			Write: func(entry io.Writer) error {
				// Chunking observes cancellation even for compressible content.
				for data := file.data; len(data) > 0; {
					if err := ctx.Err(); err != nil {
						return err
					}
					n := min(len(data), 32<<10)
					if _, err := entry.Write(data[:n]); err != nil {
						return err
					}
					data = data[n:]
				}
				return nil
			},
		}
	}
	archive, err := sourcezip.Encode(ctx, members, sourcezip.Options{ModifiedField: true, Limits: &limits})
	if errors.Is(err, sourcezip.ErrLimit) {
		return nil, plumbing.ZeroHash, nil, ErrBudget
	}
	if err != nil {
		return nil, plumbing.ZeroHash, nil, err
	}
	return archive, commit, analysis, nil
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
