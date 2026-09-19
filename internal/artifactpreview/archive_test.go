package artifactpreview

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io/fs"
	"strings"
	"testing"
)

type member struct {
	name string
	data string
	mode fs.FileMode
}

func makeZIP(t testing.TB, members ...member) []byte {
	t.Helper()
	var data bytes.Buffer
	w := zip.NewWriter(&data)
	for _, item := range members {
		header := &zip.FileHeader{Name: item.name, Method: zip.Deflate}
		if item.mode != 0 {
			header.SetMode(item.mode)
		}
		file, err := w.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := file.Write([]byte(item.data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return data.Bytes()
}

func TestArchiveDirectoryAndLazyText(t *testing.T) {
	data := makeZIP(t,
		member{name: "SKILL.md", data: "# Example\n<script>untrusted()</script>"},
		member{name: "references/папка/read me.md", data: "Текст\n"},
		member{name: "empty/"},
		member{name: "assets/binary.dat", data: "\x00\x01\xff"},
		member{name: "assets/nested.zip", data: string(makeZIP(t, member{name: "inner.txt", data: "inside"}))},
	)
	archive, err := Open(t.Context(), data)
	if err != nil {
		t.Fatal(err)
	}
	if len(archive.Entries) != 8 {
		t.Fatalf("entries: %+v", archive.Entries)
	}
	text, err := archive.Text(t.Context(), "references/папка/read me.md")
	if err != nil || text != "Текст\n" {
		t.Fatalf("text = %q, %v", text, err)
	}
	text, err = archive.Text(t.Context(), "SKILL.md")
	if err != nil || !strings.Contains(text, "<script>") {
		t.Fatalf("source changed = %q, %v", text, err)
	}
	for _, path := range []string{"assets/binary.dat", "assets/nested.zip"} {
		if _, err := archive.Text(t.Context(), path); !errors.Is(err, ErrNotText) {
			t.Errorf("%s = %v", path, err)
		}
	}
	for _, path := range []string{"missing", "empty", "references"} {
		if _, err := archive.Text(t.Context(), path); !errors.Is(err, ErrNotFound) {
			t.Errorf("%s = %v", path, err)
		}
	}
}

func TestArchiveRejectsUnsafePathsAndEntries(t *testing.T) {
	for _, name := range []string{"../secret", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "C:/file", "//host/file", "a\x00b", "a\nb", "a\u202eb", "a/\xff", strings.Repeat("x", MaximumPathBytes+1), strings.Repeat("a/", MaximumPathDepth) + "file"} {
		t.Run(name, func(t *testing.T) {
			if _, err := Open(t.Context(), makeZIP(t, member{name: name})); !errors.Is(err, ErrInvalid) {
				t.Fatalf("unsafe path accepted: %v", err)
			}
		})
	}
	for _, members := range [][]member{
		{{name: "duplicate"}, {name: "duplicate"}},
		{{name: "a"}, {name: "a/b"}},
		{{name: "a/b"}, {name: "a"}},
		{{name: "a/"}, {name: "a"}},
		{{name: "link", data: "../../secret", mode: fs.ModeSymlink | 0777}},
		{{name: "link/", mode: fs.ModeSymlink | 0777}},
		{{name: "pipe", mode: fs.ModeNamedPipe | 0600}},
		{{name: "device", mode: fs.ModeDevice | 0600}},
	} {
		if _, err := Open(t.Context(), makeZIP(t, members...)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("unsafe entries accepted: %+v: %v", members, err)
		}
	}
}

func TestArchiveTextBoundsAndIntegrity(t *testing.T) {
	for _, size := range []int{MaximumTextBytes, MaximumTextBytes + 1} {
		archive, err := Open(t.Context(), makeZIP(t, member{name: "file", data: strings.Repeat("a", size)}))
		if err != nil {
			t.Fatal(err)
		}
		text, err := archive.Text(t.Context(), "file")
		if size == MaximumTextBytes && (err != nil || len(text) != size) {
			t.Fatalf("boundary = %d, %v", len(text), err)
		}
		if size > MaximumTextBytes && (!errors.Is(err, ErrLimit) || archive.Entries[0].Previewable) {
			t.Fatalf("oversize = %v", err)
		}
	}
	t.Run("lying expanded size", func(t *testing.T) {
		data := makeZIP(t, member{name: "bomb", data: strings.Repeat("a", 2*MaximumTextBytes)})
		central := bytes.Index(data, []byte("PK\x01\x02"))
		binary.LittleEndian.PutUint32(data[central+24:], 10)
		archive, err := Open(t.Context(), data)
		if err != nil {
			t.Fatal(err)
		}
		if text, err := archive.Text(t.Context(), "bomb"); err == nil || text != "" {
			t.Fatalf("bomb returned: %d, %v", len(text), err)
		}
	})
	t.Run("CRC checked on file read", func(t *testing.T) {
		data := makeZIP(t, member{name: "file", data: "original"})
		central := bytes.Index(data, []byte("PK\x01\x02"))
		data[central+16] ^= 1
		archive, err := Open(t.Context(), data)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := archive.Text(t.Context(), "file"); !errors.Is(err, ErrInvalid) {
			t.Fatalf("CRC = %v", err)
		}
	})
	t.Run("aggregate overflow", func(t *testing.T) {
		data := makeZIP(t, member{name: "huge"})
		central := bytes.Index(data, []byte("PK\x01\x02"))
		binary.LittleEndian.PutUint32(data[central+24:], 0xffffffff)
		if _, err := Open(t.Context(), data); err == nil {
			t.Fatal("oversized archive accepted")
		}
	})
	t.Run("encrypted and unsupported compression", func(t *testing.T) {
		for _, offset := range []int{8, 10} {
			data := makeZIP(t, member{name: "file"})
			central := bytes.Index(data, []byte("PK\x01\x02"))
			binary.LittleEndian.PutUint16(data[central+offset:], 1)
			if _, err := Open(t.Context(), data); !errors.Is(err, ErrInvalid) {
				t.Fatalf("offset %d = %v", offset, err)
			}
		}
	})
}

func TestArchiveDirectoryBoundsBeforeZIPAllocation(t *testing.T) {
	members := make([]member, MaximumEntries+1)
	for i := range members {
		members[i].name = "a"
	}
	data := makeZIP(t, members...)
	// Lie about both record counts: the preflight must count actual records.
	binary.LittleEndian.PutUint16(data[len(data)-14:], 1)
	binary.LittleEndian.PutUint16(data[len(data)-12:], 1)
	if _, err := Open(t.Context(), data); !errors.Is(err, ErrLimit) {
		t.Fatalf("directory bomb = %v", err)
	}
	for _, broken := range [][]byte{nil, []byte("PK"), data[:20], append(makeZIP(t), 0)} {
		if _, err := Open(t.Context(), broken); err == nil {
			t.Fatal("broken ZIP accepted")
		}
	}
	empty, err := Open(t.Context(), makeZIP(t))
	if err != nil || len(empty.Entries) != 0 {
		t.Fatalf("empty ZIP = %+v, %v", empty, err)
	}
}

func TestArchiveCancellation(t *testing.T) {
	data := makeZIP(t, member{name: "file", data: "contents"})
	ctx, cancel := context.WithCancel(t.Context())
	archive, err := Open(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := Open(ctx, data); !errors.Is(err, context.Canceled) {
		t.Fatalf("open = %v", err)
	}
	if _, err := archive.Text(ctx, "file"); !errors.Is(err, context.Canceled) {
		t.Fatalf("text = %v", err)
	}
}

func TestArchivePreflightRejectsAmbiguousEndRecords(t *testing.T) {
	data := makeZIP(t, member{name: "file"})
	// A second signature inside the comment could make archive/zip select a
	// different directory than a preflight which skips invalid comment lengths.
	binary.LittleEndian.PutUint16(data[len(data)-2:], 23)
	comment := make([]byte, 23)
	copy(comment, "PK\x05\x06")
	data = append(data, comment...)
	if _, err := checkDirectory(data); !errors.Is(err, ErrInvalid) {
		t.Fatalf("ambiguous end record = %v", err)
	}
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	locator := make([]byte, 20)
	copy(locator, "PK\x06\x07")
	_, err := w.CreateHeader(&zip.FileHeader{Name: "file", Comment: string(locator)})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := checkDirectory(buf.Bytes()); !errors.Is(err, ErrInvalid) {
		t.Fatalf("hidden ZIP64 locator = %v", err)
	}
}

func FuzzArchivePreview(f *testing.F) {
	f.Add(makeZIP(f, member{name: "a/b.txt", data: "hello"}))
	f.Add(makeZIP(f))
	f.Add([]byte("PK\x05\x06"))
	f.Fuzz(func(t *testing.T, data []byte) {
		archive, err := Open(t.Context(), data)
		if err != nil {
			return
		}
		if len(archive.Entries) > MaximumEntries {
			t.Fatal("unbounded entries")
		}
		for _, entry := range archive.Entries {
			if !ValidPath(entry.Path) {
				t.Fatal("unsafe path")
			}
			if entry.Previewable {
				text, _ := archive.Text(t.Context(), entry.Path)
				if len(text) > MaximumTextBytes {
					t.Fatal("unbounded text")
				}
				break
			}
		}
	})
}
