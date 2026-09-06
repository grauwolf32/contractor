package artifacts

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
)

const cleanupBatchSize = 256

var blobShardPattern = regexp.MustCompile(`^[0-9a-f]{2}$`)

type BlobCleanupReport struct {
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
func CleanupFilesystemBlobs(ctx context.Context, db postgres.DBTX, path string, apply bool) (report BlobCleanupReport, err error) {
	report.Applied = apply
	if _, err = ValidateBlobConfig(string(BlobFilesystem), path); err != nil {
		return
	}
	var backend *string
	if err = db.QueryRow(ctx, `SELECT backend FROM artifact_blob_settings WHERE singleton`).Scan(&backend); err != nil {
		return
	}
	if backend == nil || *backend != string(BlobFilesystem) {
		return report, ErrBlobBackendMismatch
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
			return report, ErrArtifactIntegrity
		}
	}
	root, openErr := os.OpenRoot(path)
	if openErr != nil {
		return report, blobIOError(openErr)
	}
	defer root.Close()
	s := &FilesystemBlobStore{root: root}
	// Check references in keyset batches independently of the directory scan so
	// missing referenced files remain visible and metadata is never repaired.
	after := ""
	for {
		rows, qerr := db.Query(ctx, `SELECT object_key FROM artifact_blobs WHERE backend='filesystem' AND object_key > $1 ORDER BY object_key LIMIT $2`, after, cleanupBatchSize)
		if qerr != nil {
			return report, qerr
		}
		var keys []string
		for rows.Next() {
			var key string
			if err = rows.Scan(&key); err != nil {
				rows.Close()
				return
			}
			keys = append(keys, key)
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return
		}
		if len(keys) == 0 {
			break
		}
		for _, key := range keys {
			if !blobKeyPattern.MatchString(key) {
				return report, ErrArtifactIntegrity
			}
			if e := s.checkDirectory(key[:2]); e != nil && !errors.Is(e, ErrBlobMissing) {
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
				return report, ErrArtifactIntegrity
			}
		}
		after = keys[len(keys)-1]
	}
	process := func(keys []string) error {
		if len(keys) == 0 {
			return nil
		}
		rows, e := db.Query(ctx, `SELECT object_key FROM artifact_blobs WHERE object_key = ANY($1::text[])`, keys)
		if e != nil {
			return e
		}
		refs := make(map[string]bool, len(keys))
		for rows.Next() {
			var key string
			if e = rows.Scan(&key); e != nil {
				rows.Close()
				return e
			}
			refs[key] = true
		}
		e = rows.Err()
		rows.Close()
		if e != nil {
			return e
		}
		for _, key := range keys {
			if refs[key] {
				continue
			}
			report.Orphans++
			if apply {
				if e := s.Delete(ctx, BlobObject{Backend: BlobFilesystem, Key: key}); e != nil {
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
				return report, ErrArtifactIntegrity
			}
			name := entry.Name()
			if stagingKeyPattern.MatchString(name) {
				info, e := entry.Info()
				if e != nil {
					return report, blobIOError(e)
				}
				if !info.Mode().IsRegular() {
					return report, ErrArtifactIntegrity
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

func scanBlobShard(ctx context.Context, s *FilesystemBlobStore, name string, process func([]string) error, report *BlobCleanupReport) error {
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
				return ErrArtifactIntegrity
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
				return ErrArtifactIntegrity
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
