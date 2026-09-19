// Package artifactpreview provides read-only, bounded inspection of untrusted
// ZIP artifacts. It never extracts entries to a filesystem or follows links.
package artifactpreview

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"io"
	"io/fs"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	MaximumArchiveBytes        = 64 << 20
	MaximumEntries             = 4096 // Includes implicit parent directories in the response.
	MaximumExpandedBytes       = 256 << 20
	MaximumTextBytes           = 256 << 10
	MaximumCompressedTextBytes = 1 << 20
	MaximumPathBytes           = 1024
	MaximumPathDepth           = 32
)

var (
	ErrInvalid  = errors.New("archive is invalid or contains unsafe entries")
	ErrLimit    = errors.New("archive preview limit exceeded")
	ErrNotFound = errors.New("archive file not found")
	ErrNotText  = errors.New("archive file is not supported UTF-8 text")
)

func SupportsMediaType(mediaType string) bool {
	return mediaType == "application/zip" || mediaType == "application/x-zip-compressed" ||
		mediaType == "application/vnd.contractor.agent-skill+zip"
}

type Entry struct {
	Path        string `json:"path"`
	Kind        string `json:"kind"`
	Size        uint64 `json:"size"`
	Previewable bool   `json:"previewable"`
}

type Archive struct {
	Entries []Entry
	files   map[string]*zip.File
}

// Open reads only the directory and local headers. Entry bodies are opened on
// demand by Text. The directory is bounded before archive/zip allocates files.
func Open(ctx context.Context, data []byte) (*Archive, error) {
	if len(data) > MaximumArchiveBytes {
		return nil, ErrLimit
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	directoryOffset, err := checkDirectory(data)
	if err != nil {
		return nil, err
	}
	reader, err := zip.NewReader(contextReaderAt{ctx, bytes.NewReader(data)}, int64(len(data)))
	if err != nil {
		return nil, ErrInvalid
	}
	entries := make(map[string]Entry, len(reader.File))
	archive := &Archive{files: make(map[string]*zip.File, len(reader.File))}
	var expanded uint64
	for _, file := range reader.File {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		directory := strings.HasSuffix(file.Name, "/")
		name := strings.TrimSuffix(file.Name, "/")
		if !ValidPath(name) {
			return nil, ErrInvalid
		}
		if _, exists := entries[name]; exists {
			return nil, ErrInvalid
		}
		if file.Flags&(1|0x40|0x2000) != 0 || file.Method != zip.Store && file.Method != zip.Deflate {
			return nil, ErrInvalid
		}
		mode := file.Mode()
		if directory && (mode.Type() != fs.ModeDir || file.UncompressedSize64 != 0) || !directory && !mode.IsRegular() {
			return nil, ErrInvalid
		}
		if file.UncompressedSize64 > MaximumExpandedBytes-expanded {
			return nil, ErrLimit
		}
		expanded += file.UncompressedSize64
		offset, err := file.DataOffset()
		if err != nil || offset < 0 || offset > int64(directoryOffset) || file.CompressedSize64 > uint64(int64(directoryOffset)-offset) {
			return nil, ErrInvalid
		}
		entry := Entry{Path: name, Kind: "file", Size: file.UncompressedSize64,
			Previewable: file.UncompressedSize64 <= MaximumTextBytes && file.CompressedSize64 <= MaximumCompressedTextBytes}
		if directory {
			entry.Kind, entry.Previewable = "directory", false
		} else {
			archive.files[name] = file
		}
		entries[name] = entry
	}
	// Materialize implicit directories and reject file/directory collisions in
	// either input order. A flat map keeps validation linear in bounded depth.
	for _, file := range reader.File {
		parts := strings.Split(strings.TrimSuffix(file.Name, "/"), "/")
		for i := 1; i < len(parts); i++ {
			parent := strings.Join(parts[:i], "/")
			if entry, exists := entries[parent]; exists && entry.Kind != "directory" {
				return nil, ErrInvalid
			}
			entries[parent] = Entry{Path: parent, Kind: "directory"}
			if len(entries) > MaximumEntries {
				return nil, ErrLimit
			}
		}
	}
	archive.Entries = make([]Entry, 0, len(entries))
	for _, entry := range entries {
		archive.Entries = append(archive.Entries, entry)
	}
	sort.Slice(archive.Entries, func(i, j int) bool { return archive.Entries[i].Path < archive.Entries[j].Path })
	return archive, nil
}

// ValidPath accepts canonical relative UTF-8 names, independently of platform
// filepath rules or GODEBUG's optional ZIP path checks. Names are never cleaned.
func ValidPath(name string) bool {
	if name == "" || len(name) > MaximumPathBytes || !utf8.ValidString(name) || strings.ContainsAny(name, "\\:") {
		return false
	}
	for _, char := range name {
		if unicode.IsControl(char) || unicode.Is(unicode.Cf, char) {
			return false
		}
	}
	parts := strings.Split(name, "/")
	if len(parts) > MaximumPathDepth {
		return false
	}
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return false
		}
	}
	return true
}

func (a *Archive) Text(ctx context.Context, name string) (string, error) {
	if !ValidPath(name) {
		return "", ErrInvalid
	}
	file, exists := a.files[name]
	if !exists {
		return "", ErrNotFound
	}
	if file.UncompressedSize64 > MaximumTextBytes || file.CompressedSize64 > MaximumCompressedTextBytes {
		return "", ErrLimit
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	reader, err := file.Open()
	if err != nil {
		return "", ErrInvalid
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, MaximumTextBytes+1))
	if err != nil {
		return "", ErrInvalid
	}
	if len(data) > MaximumTextBytes {
		return "", ErrLimit
	}
	// Reading through EOF checks archive/zip's size and CRC validation. Never
	// accept truncated previews or use the claimed expanded size as a read limit.
	if uint64(len(data)) != file.UncompressedSize64 {
		return "", ErrInvalid
	}
	if !utf8.Valid(data) {
		return "", ErrNotText
	}
	for _, char := range string(data) {
		if unicode.IsControl(char) && char != '\n' && char != '\r' && char != '\t' {
			return "", ErrNotText
		}
	}
	return string(data), nil
}

type contextReaderAt struct {
	ctx    context.Context
	reader io.ReaderAt
}

func (r contextReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.ReadAt(p, off)
}
