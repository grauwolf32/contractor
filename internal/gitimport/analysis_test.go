package gitimport

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/sourcezip"
)

func analysisObjects() (map[plumbing.Hash]*gitObject, plumbing.Hash) {
	objects := map[plumbing.Hash]*gitObject{}
	add := func(kind plumbing.ObjectType, data []byte) plumbing.Hash {
		o := &gitObject{kind: kind, data: data}
		h := objectHash(o)
		objects[h] = o
		return h
	}
	text := add(plumbing.BlobObject, []byte("original source\n"))
	binary := add(plumbing.BlobObject, []byte{0xff, 0xfe})
	lfs := add(plumbing.BlobObject, []byte("version https://git-lfs.github.com/spec/v1\noid sha256:x\n"))
	ignored := add(plumbing.TreeObject, treeEntry("100644", "nested.py", text))
	var tree []byte
	for _, e := range []struct {
		mode, name string
		hash       plumbing.Hash
	}{
		{"100644", "source.py", text}, {"100644", "image.png", text}, {"100644", "invalid.py", binary},
		{"100644", "large.bin", lfs}, {"40000", "dist", ignored}, {"120000", "link", text}, {"160000", "module", plumbing.ZeroHash},
	} {
		tree = append(tree, treeEntry(e.mode, e.name, e.hash)...)
	}
	root := add(plumbing.TreeObject, tree)
	commit := add(plumbing.CommitObject, []byte("tree "+root.String()+"\n\nfixture\n"))
	return objects, commit
}

func TestAnalysisArchiveReportsEveryExclusion(t *testing.T) {
	objects, commit := analysisObjects()
	first, resolved, summary, err := archiveSnapshot(t.Context(), objects, commit, sourcezip.DefaultLimits(), ModeAnalysis)
	if err != nil || resolved != commit || summary == nil {
		t.Fatalf("analysis: %v %+v", err, summary)
	}
	if summary.IncludedFiles != 1 || summary.ExcludedEntries != 6 || summary.Policy != artifacts.GitAnalysisPolicy {
		t.Fatalf("summary: %+v", summary)
	}
	second, _, _, err := archiveSnapshot(t.Context(), objects, commit, sourcezip.DefaultLimits(), ModeAnalysis)
	if err != nil || !bytes.Equal(first, second) {
		t.Fatalf("not deterministic: %v", err)
	}
	z, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	if len(z.File) != 2 {
		t.Fatalf("files: %d", len(z.File))
	}
	var report analysisReport
	for _, f := range z.File {
		r, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		switch f.Name {
		case summary.ReportPath:
			if err := json.Unmarshal(data, &report); err != nil {
				t.Fatal(err)
			}
		case "source.py":
			if string(data) != "original source\n" {
				t.Fatal("source changed")
			}
		default:
			t.Fatalf("excluded member published: %s", f.Name)
		}
	}
	want := []exclusion{{"dist/nested.py", "not_source_path"}, {"image.png", "not_source_path"}, {"invalid.py", "non_utf8"}, {"large.bin", "lfs_pointer"}, {"link", "symbolic_link"}, {"module", "submodule"}}
	actual, _ := json.Marshal(report.Excluded)
	expected, _ := json.Marshal(want)
	if !bytes.Equal(actual, expected) || report.ResolvedCommit != commit.String() {
		t.Fatalf("report: %s", actual)
	}
	if _, _, _, err := archiveSnapshot(t.Context(), objects, commit, sourcezip.DefaultLimits(), ModeSnapshot); !errors.Is(err, ErrContent) {
		t.Fatalf("snapshot accepted: %v", err)
	}
}

func TestAnalysisArchiveFailuresAndBudgets(t *testing.T) {
	for _, test := range []struct {
		name string
		data []byte
		want error
	}{
		{artifacts.GitAnalysisReportPath, []byte("collision"), ErrContent}, {"image.png", []byte("binary extension"), ErrContent},
		{"../escape.py", []byte("unsafe"), ErrContent}, {"source.py", bytes.Repeat([]byte("x"), sourcezip.MaxFileBytes+1), ErrBudget},
	} {
		t.Run(test.name, func(t *testing.T) {
			o, c := snapshotObjects("100644", test.name, test.data)
			if _, _, _, err := archiveSnapshot(t.Context(), o, c, sourcezip.DefaultLimits(), ModeAnalysis); !errors.Is(err, test.want) {
				t.Fatalf("got %v", err)
			}
		})
	}
	objects, commit := analysisObjects()
	for _, limits := range []sourcezip.Limits{
		{MaxEntries: 1, MaxFileBytes: 4 << 20, MaxExpandedBytes: 64 << 20},
		{MaxEntries: 10, MaxFileBytes: 64, MaxExpandedBytes: 64 << 20},
		{MaxEntries: 10, MaxFileBytes: 64, MaxExpandedBytes: 64},
	} {
		if _, _, _, err := archiveSnapshot(t.Context(), objects, commit, limits, ModeAnalysis); !errors.Is(err, ErrBudget) {
			t.Fatalf("budget accepted: %v", err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, _, _, err := archiveSnapshot(ctx, objects, commit, sourcezip.DefaultLimits(), ModeAnalysis); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled: %v", err)
	}
	report := analysisReport{}
	for n := 0; ; n++ {
		err := report.exclude(fmt.Sprintf("%05d-%s", n, strings.Repeat("p", 480)), "not_source_path")
		if err != nil {
			if !errors.Is(err, ErrBudget) || n < 5000 || n > 10000 {
				t.Fatalf("report bound after %d: %v", n, err)
			}
			break
		}
	}
}

func TestAdvertisementAdmitsNextcloudSizedReferenceSet(t *testing.T) {
	adv := packp.NewAdvRefs()
	head := plumbing.NewHash(strings.Repeat("1", 40))
	adv.Head = &head
	for i := range 48000 {
		adv.References[fmt.Sprintf("refs/pull/%d/head", i)] = head
	}
	var data bytes.Buffer
	if err := adv.Encode(&data); err != nil {
		t.Fatal(err)
	}
	parsed, err := decodeAdvertisement(t.Context(), &data)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := resolveRef(parsed, ""); err != nil || got != head {
		t.Fatalf("resolve: %s %v", got, err)
	}
}
