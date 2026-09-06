package gitimport

import (
	"archive/zip"
	"bytes"
	"compress/zlib"
	"context"
	"crypto/sha1"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp"
)

func TestRemotePolicy(t *testing.T) {
	for _, raw := range []string{"file:///repo", "git://example.com/repo", "http://example.com/repo", "https://u:p@example.com/repo", "https://u@example.com/repo", "ssh://u:p@example.com/repo", "https://example.com/repo?token=x", "https://example.com/repo#x", "https://example.com/repo?", "ssh://git@example.com/%00", "ssh://-bad@example.com/repo", "https://example.com:0/repo"} {
		t.Run(raw, func(t *testing.T) {
			if _, err := ParseRemote(raw); !errors.Is(err, ErrURL) {
				t.Fatalf("got %v", err)
			}
		})
	}
	remote, err := ParseRemote("git@EXAMPLE.com:team/repo.git")
	if err != nil || remote.URL != "ssh://git@example.com:22/~/team/repo.git" || remote.Path != "team/repo.git" {
		t.Fatalf("remote=%+v err=%v", remote, err)
	}
	for _, raw := range []string{"git@EXAMPLE.com:team/repo.git", "git@example.com:/team/repo.git", "git@example.com:team/100%repo.git", "ssh://git@example.com/~/team/repo.git"} {
		parsed, err := ParseRemote(raw)
		if err != nil {
			t.Fatal(err)
		}
		again, err := ParseRemote(parsed.URL)
		if err != nil || parsed != again {
			t.Fatalf("normalization changed remote: %+v -> %+v: %v", parsed, again, err)
		}
	}
	absolute, _ := ParseRemote("git@example.com:/team/repo.git")
	if absolute.Path != "/team/repo.git" {
		t.Fatalf("absolute SCP path changed: %+v", absolute)
	}
	client, _ := NewClient(Config{AllowedRemotes: []string{"127.0.0.1:443"}})
	if _, err := client.dial(context.Background(), "tcp", "127.0.0.1:443"); !errors.Is(err, ErrDestination) {
		t.Fatalf("loopback: %v", err)
	}
	if _, err := client.dial(context.Background(), "tcp", "example.com:443"); !errors.Is(err, ErrDestination) {
		t.Fatalf("allowlist: %v", err)
	}
}

func TestResolveRef(t *testing.T) {
	adv := packp.NewAdvRefs()
	head := plumbing.NewHash(strings.Repeat("1", 40))
	other := plumbing.NewHash(strings.Repeat("2", 40))
	adv.Head = &head
	adv.References["refs/heads/main"] = head
	adv.References["refs/tags/v1"] = other
	for ref, want := range map[string]plumbing.Hash{"": head, "main": head, "refs/heads/main": head, "v1": other, "refs/tags/v1": other} {
		if got, err := resolveRef(adv, ref); err != nil || got != want {
			t.Fatalf("%s: %s %v", ref, got, err)
		}
	}
	adv.References["refs/tags/main"] = head
	for _, ref := range []string{"main", "missing", head.String(), "refs/pull/1"} {
		if _, err := resolveRef(adv, ref); !errors.Is(err, ErrRef) {
			t.Fatalf("%s: %v", ref, err)
		}
	}
}

func TestHTTPSRejectsRedirectAndUntrustedTLS(t *testing.T) {
	redirected := false
	target := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { redirected = true; w.WriteHeader(500) }))
	defer target.Close()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	remote, _ := ParseRemote(server.URL + "/repo")
	client, _ := NewClient(Config{AllowedRemotes: []string{remote.Address}})
	client.allowLoopback = true
	if _, err := client.Fetch(context.Background(), remote, "", nil); err == nil {
		t.Fatal("untrusted TLS succeeded")
	}
	client.tlsConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	if _, err := client.Fetch(context.Background(), remote, "", nil); !errors.Is(err, ErrRemote) {
		t.Fatalf("redirect: %v", err)
	}
	if redirected {
		t.Fatal("redirect was followed")
	}
}

type packEntry struct {
	kind     plumbing.ObjectType
	data     []byte
	base     plumbing.Hash
	declared *int64
}

func makePack(entries ...packEntry) []byte {
	var b bytes.Buffer
	b.WriteString("PACK")
	_ = binary.Write(&b, binary.BigEndian, uint32(2))
	_ = binary.Write(&b, binary.BigEndian, uint32(len(entries)))
	for _, entry := range entries {
		size := int64(len(entry.data))
		if entry.declared != nil {
			size = *entry.declared
		}
		first := byte(entry.kind)<<4 | byte(size&15)
		size >>= 4
		if size > 0 {
			first |= 128
		}
		b.WriteByte(first)
		for size > 0 {
			next := byte(size & 127)
			size >>= 7
			if size > 0 {
				next |= 128
			}
			b.WriteByte(next)
		}
		if entry.kind == plumbing.REFDeltaObject {
			b.Write(entry.base[:])
		}
		w := zlib.NewWriter(&b)
		_, _ = w.Write(entry.data)
		_ = w.Close()
	}
	sum := sha1.Sum(b.Bytes())
	b.Write(sum[:])
	return b.Bytes()
}

func TestPackBudgetsAndDelta(t *testing.T) {
	base := &gitObject{kind: plumbing.BlobObject, data: []byte("hello")}
	delta := []byte{5, 6, 6, 'h', 'e', 'l', 'l', 'o', '!'}
	pack := makePack(packEntry{kind: plumbing.REFDeltaObject, data: delta, base: objectHash(base)}, packEntry{kind: base.kind, data: base.data})
	objects, err := decodePack(context.Background(), pack)
	if err != nil || objects[objectHash(&gitObject{kind: plumbing.BlobObject, data: []byte("hello!")})] == nil {
		t.Fatalf("delta: %v", err)
	}
	oversize := int64(maxObjectBytes + 1)
	largeDelta := binary.AppendUvarint([]byte{5}, maxObjectBytes+1)
	badChecksum := bytes.Clone(pack)
	badChecksum[len(badChecksum)-1] ^= 1
	tooMany := bytes.Clone(pack)
	binary.BigEndian.PutUint32(tooMany[8:12], maxObjects+1)
	for name, tc := range map[string]struct {
		pack []byte
		err  error
	}{
		"object-size": {makePack(packEntry{kind: plumbing.BlobObject, data: []byte("x"), declared: &oversize}), ErrBudget},
		"delta-size":  {makePack(packEntry{kind: base.kind, data: base.data}, packEntry{kind: plumbing.REFDeltaObject, data: largeDelta, base: objectHash(base)}), ErrBudget},
		"checksum":    {badChecksum, ErrContent}, "trailing": {append(bytes.Clone(pack), 0), ErrContent}, "count": {tooMany, ErrBudget}, "truncated": {pack[:len(pack)-1], ErrContent},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := decodePack(context.Background(), tc.pack); !errors.Is(err, tc.err) {
				t.Fatalf("got %v", err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := decodePack(ctx, pack); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
}

func TestDeltaDepthAndAdvertisementLimits(t *testing.T) {
	base := &gitObject{kind: plumbing.BlobObject, data: []byte("base")}
	entries := []packEntry{{kind: base.kind, data: base.data}}
	for i := 0; i <= maxDeltaDepth; i++ {
		next := append(bytes.Clone(base.data), 'x')
		delta := binary.AppendUvarint(nil, uint64(len(base.data)))
		delta = binary.AppendUvarint(delta, uint64(len(next)))
		delta = append(delta, byte(len(next)))
		delta = append(delta, next...)
		entries = append(entries, packEntry{kind: plumbing.REFDeltaObject, data: delta, base: objectHash(base)})
		base = &gitObject{kind: plumbing.BlobObject, data: next}
	}
	if _, err := decodePack(context.Background(), makePack(entries...)); !errors.Is(err, ErrBudget) {
		t.Fatalf("delta depth: %v", err)
	}
	adv := packp.NewAdvRefs()
	for i := 0; i <= maxReferences; i++ {
		adv.References[fmt.Sprintf("refs/heads/%d", i)] = plumbing.NewHash(strings.Repeat("1", 40))
	}
	if _, err := resolveRef(adv, "0"); !errors.Is(err, ErrBudget) {
		t.Fatalf("refs: %v", err)
	}
	var advertisement bytes.Buffer
	firstLine := strings.Repeat("1", 40) + " HEAD\x00shallow\n"
	fmt.Fprintf(&advertisement, "%04x%s", len(firstLine)+4, firstLine)
	for advertisement.Len() < maxAdvertisementBytes+65536 {
		line := strings.Repeat("1", 40) + " refs/heads/" + fmt.Sprint(advertisement.Len()) + strings.Repeat("x", 4000) + "\n"
		fmt.Fprintf(&advertisement, "%04x%s", len(line)+4, line)
	}
	if _, err := decodeAdvertisement(context.Background(), &advertisement); !errors.Is(err, ErrBudget) {
		t.Fatalf("advertisement: %v", err)
	}
}

func TestNetworkCancellation(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(started); <-r.Context().Done() }))
	defer server.Close()
	remote, _ := ParseRemote(server.URL + "/repo")
	client, _ := NewClient(Config{AllowedRemotes: []string{remote.Address}})
	client.allowLoopback = true
	client.tlsConfig = server.Client().Transport.(*http.Transport).TLSClientConfig.Clone()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { <-started; cancel() }()
	if _, err := client.Fetch(ctx, remote, "", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel in-flight: %v", err)
	}
}

func treeEntry(mode, name string, hash plumbing.Hash) []byte {
	return append([]byte(mode+" "+name+"\x00"), hash[:]...)
}
func snapshotObjects(mode, name string, data []byte) (map[plumbing.Hash]*gitObject, plumbing.Hash) {
	blob := &gitObject{kind: plumbing.BlobObject, data: data}
	blobHash := objectHash(blob)
	tree := &gitObject{kind: plumbing.TreeObject, data: treeEntry(mode, name, blobHash)}
	treeHash := objectHash(tree)
	commit := &gitObject{kind: plumbing.CommitObject, data: []byte("tree " + treeHash.String() + "\n\nfixture\n")}
	commitHash := objectHash(commit)
	return map[plumbing.Hash]*gitObject{blobHash: blob, treeHash: tree, commitHash: commit}, commitHash
}
func TestSnapshotContentAndDeterminism(t *testing.T) {
	objects, commit := snapshotObjects("100755", "hello.txt", []byte("hello\n"))
	first, resolved, err := archiveSnapshot(context.Background(), objects, commit)
	if err != nil || resolved != commit {
		t.Fatal(err)
	}
	second, _, err := archiveSnapshot(context.Background(), objects, commit)
	if err != nil || !bytes.Equal(first, second) {
		t.Fatalf("not deterministic: %v", err)
	}
	reader, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	if len(reader.File) != 1 || reader.File[0].Name != "hello.txt" || reader.File[0].Mode().Perm() != 0644 {
		t.Fatalf("ZIP: %+v", reader.File)
	}
	r, _ := reader.File[0].Open()
	data, _ := io.ReadAll(r)
	_ = r.Close()
	if string(data) != "hello\n" {
		t.Fatal("wrong bytes")
	}
	for _, tc := range []struct {
		mode, name string
		data       []byte
		want       error
	}{
		{"120000", "link", []byte("target"), ErrContent}, {"160000", "module", nil, ErrContent}, {"100644", "../escape", nil, ErrContent}, {"100644", "C:drive", nil, ErrContent}, {"100644", ".git", nil, ErrContent}, {"100644", strings.Repeat("x", 513), nil, ErrBudget}, {"100644", "large", make([]byte, maxFileBytes+1), ErrBudget}, {"100644", "lfs", []byte("version https://git-lfs.github.com/spec/v1\noid sha256:x\n"), ErrContent},
	} {
		t.Run(fmt.Sprintf("%s-%s", tc.mode, tc.name[:min(len(tc.name), 20)]), func(t *testing.T) {
			objects, commit := snapshotObjects(tc.mode, tc.name, tc.data)
			if _, _, err := archiveSnapshot(context.Background(), objects, commit); !errors.Is(err, tc.want) {
				t.Fatalf("got %v", err)
			}
		})
	}
}

func TestSnapshotEntryAndExpandedLimits(t *testing.T) {
	for _, tc := range []struct {
		name  string
		count int
		size  int
	}{
		{"entries", maxEntries + 1, 0},
		{"expanded", MaxArchiveBytes/maxFileBytes + 1, maxFileBytes},
	} {
		t.Run(tc.name, func(t *testing.T) {
			blob := &gitObject{kind: plumbing.BlobObject, data: make([]byte, tc.size)}
			blobHash := objectHash(blob)
			var treeData []byte
			for index := range tc.count {
				treeData = append(treeData, treeEntry("100644", fmt.Sprintf("file-%05d", index), blobHash)...)
			}
			tree := &gitObject{kind: plumbing.TreeObject, data: treeData}
			treeHash := objectHash(tree)
			commit := &gitObject{kind: plumbing.CommitObject, data: []byte("tree " + treeHash.String() + "\n\nfixture\n")}
			commitHash := objectHash(commit)
			objects := map[plumbing.Hash]*gitObject{blobHash: blob, treeHash: tree, commitHash: commit}
			if _, _, err := archiveSnapshot(context.Background(), objects, commitHash); !errors.Is(err, ErrBudget) {
				t.Fatalf("snapshot limit: %v", err)
			}
		})
	}
}
