package filesystem

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/grauwolf32/contractor/internal/artifacts/blob"
)

const cleanupBatchSize = 256

var blobShardPattern = regexp.MustCompile(`^[0-9a-f]{2}$`)

type CleanupReport struct {
	Applied    bool  `json:"applied"`
	Referenced int64 `json:"referenced"`
	Missing    int64 `json:"missing"`
	Orphans    int64 `json:"orphans"`
	Staging    int64 `json:"staging"`
	Removed    int64 `json:"removed"`
	Ignored    int64 `json:"ignored"`
}

// CleanupFilesystemBlobs is offline-only for apply. The caller must require an
// explicit acknowledgement that every writer sharing this registry is stopped.
// Dry-run opens an existing root without creating directories or probe files.
func Cleanup(ctx context.Context, path string, apply bool, registry References) (report CleanupReport, err error) {
	report.Applied = apply
	if _, err = blob.ValidateConfig(string(blob.Filesystem), path); err != nil {
		return
	}
	current := string(filepath.Separator)
	for _, component := range strings.Split(strings.TrimPrefix(path, current), string(filepath.Separator)) {
		current = filepath.Join(current, component)
		var info os.FileInfo
		info, err = os.Lstat(current)
		if err != nil {
			return report, blobIOError(err)
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return report, blob.ErrIntegrity
		}
	}
	root, openErr := os.OpenRoot(path)
	if openErr != nil {
		return report, blobIOError(openErr)
	}
	defer root.Close()
	s := &Store{root: root}
	// Check references in keyset batches independently of the directory scan so
	// missing referenced files remain visible and metadata is never repaired.
	after := ""
	for {
		keys, queryErr := registry.Page(ctx, after, cleanupBatchSize)
		if queryErr != nil {
			return report, queryErr
		}
		if len(keys) == 0 {
			break
		}
		for _, key := range keys {
			if !blobKeyPattern.MatchString(key) {
				return report, blob.ErrIntegrity
			}
			if e := s.checkDirectory(key[:2]); e != nil && !errors.Is(e, blob.ErrMissing) {
				return report, e
			}
			info, e := root.Lstat(key)
			report.Referenced++
			if errors.Is(e, os.ErrNotExist) {
				report.Missing++
				continue
			}
			if e != nil {
				return report, blobIOError(e)
			}
			if !info.Mode().IsRegular() {
				return report, blob.ErrIntegrity
			}
		}
		after = keys[len(keys)-1]
	}
	process := func(keys []string) error {
		if len(keys) == 0 {
			return nil
		}
		refs, e := registry.Referenced(ctx, keys)
		if e != nil {
			return e
		}
		for _, key := range keys {
			if refs[key] {
				continue
			}
			report.Orphans++
			if apply {
				if e := s.Delete(ctx, blob.Object{Backend: blob.Filesystem, Key: key}); e != nil {
					return e
				}
				report.Removed++
			}
		}
		return nil
	}
	directory, e := root.Open(".")
	if e != nil {
		return report, blobIOError(e)
	}
	defer directory.Close()
	for {
		if err = ctx.Err(); err != nil {
			return
		}
		entries, e := directory.ReadDir(cleanupBatchSize)
		if e != nil && e != io.EOF {
			return report, blobIOError(e)
		}
		for _, entry := range entries {
			if entry.Type()&os.ModeSymlink != 0 {
				return report, blob.ErrIntegrity
			}
			name := entry.Name()
			if stagingKeyPattern.MatchString(name) {
				info, e := entry.Info()
				if e != nil {
					return report, blobIOError(e)
				}
				if !info.Mode().IsRegular() {
					return report, blob.ErrIntegrity
				}
				report.Staging++
				if apply {
					if e := root.Remove(name); e != nil {
						return report, blobIOError(e)
					}
					report.Removed++
				}
				continue
			}
			if !blobShardPattern.MatchString(name) || !entry.IsDir() {
				report.Ignored++
				continue
			}
			if err = s.checkDirectory(name); err != nil {
				return
			}
			if err = scanBlobShard(ctx, s, name, process, &report); err != nil {
				return
			}
		}
		if e == io.EOF {
			break
		}
	}
	return
}

func scanBlobShard(ctx context.Context, s *Store, name string, process func([]string) error, report *CleanupReport) error {
	f, err := s.root.Open(name)
	if err != nil {
		return blobIOError(err)
	}
	defer f.Close()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		entries, err := f.ReadDir(cleanupBatchSize)
		if err != nil && err != io.EOF {
			return blobIOError(err)
		}
		keys := make([]string, 0, len(entries))
		for _, entry := range entries {
			if entry.Type()&os.ModeSymlink != 0 {
				return blob.ErrIntegrity
			}
			key := name + "/" + entry.Name()
			if !blobKeyPattern.MatchString(key) {
				report.Ignored++
				continue
			}
			info, e := entry.Info()
			if e != nil {
				return blobIOError(e)
			}
			if !info.Mode().IsRegular() {
				return blob.ErrIntegrity
			}
			keys = append(keys, key)
		}
		if e := process(keys); e != nil {
			return e
		}
		if err == io.EOF {
			return nil
		}
	}
}

// References is the registry's bounded key inventory for offline cleanup.
// It supplies metadata only; filesystem traversal never interprets registry SQL.
type References interface {
	Page(context.Context, string, int) ([]string, error)
	Referenced(context.Context, []string) (map[string]bool, error)
}
