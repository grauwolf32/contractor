package gitimport

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/grauwolf32/contractor/internal/sourcezip"
)

func TestConfiguredSourceBudgetsAcceptLargerSnapshot(t *testing.T) {
	data := bytes.Repeat([]byte("source line\n"), (5<<20)/len("source line\n")+1)
	blob := &gitObject{kind: plumbing.BlobObject, data: data}
	blobHash := objectHash(blob)
	var treeData []byte
	for index := range 14 {
		treeData = append(treeData, treeEntry("100644", fmt.Sprintf("part-%02d.txt", index), blobHash)...)
	}
	tree := &gitObject{kind: plumbing.TreeObject, data: treeData}
	treeHash := objectHash(tree)
	commit := &gitObject{kind: plumbing.CommitObject, data: []byte("tree " + treeHash.String() + "\n\nfixture\n")}
	commitHash := objectHash(commit)
	objects := map[plumbing.Hash]*gitObject{blobHash: blob, treeHash: tree, commitHash: commit}
	if _, _, _, err := archiveSnapshot(t.Context(), objects, commitHash, sourcezip.DefaultLimits(), ModeSnapshot); !errors.Is(err, ErrBudget) {
		t.Fatalf("default limit = %v", err)
	}
	limits := sourcezip.Limits{MaxExpandedBytes: 128 << 20, MaxFileBytes: 16 << 20, MaxEntries: 10000}
	archive, resolved, _, err := archiveSnapshot(t.Context(), objects, commitHash, limits, ModeSnapshot)
	if err != nil || resolved != commitHash {
		t.Fatalf("raised budgets = %v", err)
	}
	reader, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil || len(reader.File) != 14 {
		t.Fatalf("snapshot ZIP = %v", err)
	}
	for _, member := range reader.File {
		if member.UncompressedSize64 != uint64(len(data)) {
			t.Fatal("source bytes lost")
		}
	}
	limits.MaxFileBytes = 4 << 20
	if _, _, _, err := archiveSnapshot(t.Context(), objects, commitHash, limits, ModeSnapshot); !errors.Is(err, ErrBudget) {
		t.Fatalf("file budget = %v", err)
	}
	limits.MaxFileBytes = 16 << 20
	limits.MaxExpandedBytes = 64 << 20
	if _, _, _, err := archiveSnapshot(t.Context(), objects, commitHash, limits, ModeSnapshot); !errors.Is(err, ErrBudget) {
		t.Fatalf("expanded budget = %v", err)
	}
	limits.MaxExpandedBytes = 128 << 20
	limits.MaxEntries = 13
	if _, _, _, err := archiveSnapshot(t.Context(), objects, commitHash, limits, ModeSnapshot); !errors.Is(err, ErrBudget) {
		t.Fatalf("entry budget = %v", err)
	}
}

func TestConfiguredPackBudgetsRejectBeforeInflation(t *testing.T) {
	pack := makePack(packEntry{kind: plumbing.BlobObject, data: bytes.Repeat([]byte("source\n"), 1000)})
	limits := DefaultLimits()
	if _, err := decodePack(context.Background(), bytes.NewReader(pack), limits); err != nil {
		t.Fatal(err)
	}
	limits.MaxDecodedBytes = 6999
	if _, err := decodePack(context.Background(), bytes.NewReader(pack), limits); !errors.Is(err, ErrBudget) {
		t.Fatalf("decoded budget = %v", err)
	}
	limits = DefaultLimits()
	limits.MaxReceivedBytes = int64(len(pack)) - 1
	if _, err := decodePack(context.Background(), bytes.NewReader(pack), limits); !errors.Is(err, ErrBudget) {
		t.Fatalf("wire budget = %v", err)
	}
	limits.MaxReceivedBytes++
	if _, err := decodePack(context.Background(), bytes.NewReader(pack), limits); err != nil {
		t.Fatalf("exact wire budget = %v", err)
	}
}
