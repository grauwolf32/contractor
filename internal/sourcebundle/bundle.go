package sourcebundle

import (
	"archive/zip"
	"bytes"
	"compress/flate"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/grauwolf32/contractor/internal/gitimport"
)

// Source ZIP limits match the runtime source_analysis toolset and Git imports.
const (
	MaxArchiveBytes  = gitimport.MaxArchiveBytes
	MaxEntries       = gitimport.MaxEntries
	MaxFileBytes     = gitimport.MaxFileBytes
	MaxExpandedBytes = gitimport.MaxArchiveBytes
	MaxPathBytes     = gitimport.MaxPathBytes
	maxPathParts     = 128
)

var ErrArchiveTooLarge = errors.New("source ZIP exceeds the 64 MiB Artifact limit")

type Options struct {
	IncludeIgnored bool
}

type Bundle struct {
	Data          []byte
	Files         int
	ExpandedBytes int64
	SHA256        string
	// SkippedRepositories counts submodules and nested Git repositories left out of the bundle.
	SkippedRepositories int
}

type sourceFile struct {
	hostPath string
	path     string
	info     fs.FileInfo
}

func Build(source string, options Options) (Bundle, error) {
	root, err := filepath.Abs(filepath.Clean(source))
	if err != nil {
		return Bundle{}, fmt.Errorf("resolve source directory: %w", err)
	}
	info, err := os.Lstat(root)
	if err != nil {
		return Bundle{}, fmt.Errorf("inspect source directory: %w", err)
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return Bundle{}, errors.New("source must be a regular directory, not a symlink")
	}

	paths, skipped, err := candidatePaths(root, options.IncludeIgnored)
	if err != nil {
		return Bundle{}, err
	}
	paths, err = applyContractorIgnore(root, paths)
	if err != nil {
		return Bundle{}, err
	}
	files, expanded, err := inspectFiles(root, paths)
	if err != nil {
		return Bundle{}, err
	}

	buffer := &limitedBuffer{maximum: MaxArchiveBytes}
	writer := zip.NewWriter(buffer)
	writer.RegisterCompressor(zip.Deflate, func(destination io.Writer) (io.WriteCloser, error) {
		return flate.NewWriter(destination, 6)
	})
	fixedTime := time.Date(1980, time.January, 1, 0, 0, 0, 0, time.UTC)
	for _, file := range files {
		header := &zip.FileHeader{Name: file.path, Method: zip.Deflate}
		header.SetMode(0o644)
		header.SetModTime(fixedTime)
		entry, createErr := writer.CreateHeader(header)
		if createErr != nil {
			_ = writer.Close()
			return Bundle{}, archiveError(createErr)
		}
		if copyErr := copyRegularFile(entry, file); copyErr != nil {
			_ = writer.Close()
			return Bundle{}, copyErr
		}
	}
	if err := writer.Close(); err != nil {
		return Bundle{}, archiveError(err)
	}
	payload := append([]byte(nil), buffer.Bytes()...)
	digest := sha256.Sum256(payload)
	return Bundle{
		Data: payload, Files: len(files), ExpandedBytes: expanded,
		SHA256:              "sha256:" + hex.EncodeToString(digest[:]),
		SkippedRepositories: skipped,
	}, nil
}

func candidatePaths(root string, includeIgnored bool) ([]string, int, error) {
	if !includeIgnored {
		if _, err := exec.LookPath("git"); err == nil {
			command := exec.Command("git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".")
			output, commandErr := command.Output()
			if commandErr == nil {
				paths, skipped := skipNestedRepositories(root, splitNUL(output))
				return paths, skipped, nil
			}
			var exit *exec.ExitError
			if !errors.As(commandErr, &exit) || exit.ExitCode() != 128 {
				return nil, 0, fmt.Errorf("enumerate Git working tree: %w", commandErr)
			}
			if insideGitWorkTree(root) {
				return nil, 0, fmt.Errorf("enumerate Git working tree: %w: %s; use --include-ignored to walk the directory explicitly",
					commandErr, strings.TrimSpace(string(exit.Stderr)))
			}
		} else if insideGitWorkTree(root) {
			return nil, 0, errors.New("git is required to honor .gitignore; use --include-ignored to walk the directory explicitly")
		}
	}

	paths := make([]string, 0)
	err := filepath.WalkDir(root, func(hostPath string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if hostPath == root {
			return nil
		}
		relative, err := filepath.Rel(root, hostPath)
		if err != nil {
			return err
		}
		portable := filepath.ToSlash(relative)
		if entry.IsDir() {
			if portable == ".git" || portable == ".contractor" || strings.HasPrefix(portable, ".git/") || strings.HasPrefix(portable, ".contractor/") {
				return filepath.SkipDir
			}
			return nil
		}
		paths = append(paths, portable)
		return nil
	})
	if err != nil {
		return nil, 0, fmt.Errorf("walk source directory: %w", err)
	}
	return paths, 0, nil
}

// skipNestedRepositories drops submodule gitlinks and untracked nested
// repositories, which git ls-files reports as directories rather than files.
func skipNestedRepositories(root string, paths []string) ([]string, int) {
	result := make([]string, 0, len(paths))
	skipped := 0
	for _, path := range paths {
		if strings.HasSuffix(path, "/") {
			skipped++
			continue
		}
		if info, err := os.Lstat(filepath.Join(root, filepath.FromSlash(path))); err == nil && info.IsDir() {
			skipped++
			continue
		}
		result = append(result, path)
	}
	return result, skipped
}

// insideGitWorkTree reports whether root belongs to a Git work tree, so a
// failing git ls-files is not mistaken for a plain directory.
func insideGitWorkTree(root string) bool {
	if _, err := exec.LookPath("git"); err == nil {
		output, err := exec.Command("git", "-C", root, "rev-parse", "--is-inside-work-tree").Output()
		if err == nil && strings.TrimSpace(string(output)) == "true" {
			return true
		}
	}
	for directory := root; ; {
		if _, err := os.Lstat(filepath.Join(directory, ".git")); err == nil {
			return true
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return false
		}
		directory = parent
	}
}

func applyContractorIgnore(root string, paths []string) ([]string, error) {
	ignorePath := filepath.Join(root, ".contractorignore")
	info, err := os.Lstat(ignorePath)
	if errors.Is(err, os.ErrNotExist) {
		return paths, nil
	}
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return nil, errors.New(".contractorignore must be a regular file")
	}
	if _, err := exec.LookPath("git"); err != nil {
		return nil, errors.New("git is required to evaluate .contractorignore")
	}
	temporary, err := os.MkdirTemp("", "contractor-ignore-")
	if err != nil {
		return nil, fmt.Errorf("create ignore matcher state: %w", err)
	}
	defer os.RemoveAll(temporary)
	gitDir := filepath.Join(temporary, "repo.git")
	if output, err := exec.Command("git", "init", "--quiet", "--bare", gitDir).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("initialize ignore matcher: %w: %s", err, strings.TrimSpace(string(output)))
	}
	// An empty work tree has no .gitignore files, so core.excludesFile is the
	// only pattern source. With the source root as work tree, -v would report a
	// higher-priority .gitignore match or negation instead of .contractorignore.
	// check-ignore --no-index matches paths lexically, so directory patterns
	// still apply to the leading components of nested paths.
	workTree := filepath.Join(temporary, "work-tree")
	if err := os.Mkdir(workTree, 0o700); err != nil {
		return nil, fmt.Errorf("create ignore matcher state: %w", err)
	}
	arguments := []string{
		"--git-dir=" + gitDir, "--work-tree=" + workTree,
		"-c", "core.excludesFile=" + ignorePath,
		"check-ignore", "--no-index", "-v", "-z", "--stdin",
	}
	command := exec.Command("git", arguments...)
	command.Dir = workTree
	command.Stdin = bytes.NewReader(joinNUL(paths))
	output, commandErr := command.Output()
	if commandErr != nil {
		var exit *exec.ExitError
		if !errors.As(commandErr, &exit) || exit.ExitCode() != 1 {
			return nil, fmt.Errorf("evaluate .contractorignore: %w", commandErr)
		}
	}
	fields := splitNUL(output)
	if len(fields)%4 != 0 {
		return nil, errors.New("git returned an invalid ignore result")
	}
	ignored := make(map[string]struct{})
	cleanIgnore, err := filepath.Abs(ignorePath)
	if err != nil {
		return nil, fmt.Errorf("resolve .contractorignore: %w", err)
	}
	for index := 0; index < len(fields); index += 4 {
		if strings.HasPrefix(fields[index+2], "!") {
			continue
		}
		origin, err := filepath.Abs(fields[index])
		if err == nil && origin == cleanIgnore {
			ignored[fields[index+3]] = struct{}{}
		}
	}
	result := make([]string, 0, len(paths))
	for _, path := range paths {
		if _, excluded := ignored[path]; !excluded {
			result = append(result, path)
		}
	}
	return result, nil
}

func inspectFiles(root string, paths []string) ([]sourceFile, int64, error) {
	seen := make(map[string]struct{}, len(paths))
	files := make([]sourceFile, 0, len(paths))
	var expanded int64
	for _, candidate := range paths {
		candidate = strings.TrimPrefix(filepath.ToSlash(candidate), "./")
		if candidate == ".git" || strings.HasPrefix(candidate, ".git/") ||
			candidate == ".contractor" || strings.HasPrefix(candidate, ".contractor/") {
			continue
		}
		portable, err := portablePath(candidate)
		if err != nil {
			return nil, 0, err
		}
		if _, duplicate := seen[portable]; duplicate {
			return nil, 0, fmt.Errorf("source paths collide after normalization: %s", portable)
		}
		seen[portable] = struct{}{}
		hostPath := filepath.Join(root, filepath.FromSlash(candidate))
		info, err := os.Lstat(hostPath)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, 0, fmt.Errorf("inspect source member %s: %w", portable, err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, 0, fmt.Errorf("source member %s is a symbolic link, which source push does not upload; exclude it with .contractorignore", portable)
		}
		if !info.Mode().IsRegular() {
			return nil, 0, fmt.Errorf("source member %s is not a regular file; exclude it with .contractorignore", portable)
		}
		if len(files) >= MaxEntries {
			return nil, 0, fmt.Errorf("source exceeds the %d file limit", MaxEntries)
		}
		if info.Size() > MaxFileBytes {
			return nil, 0, fmt.Errorf("source member %s exceeds the %d MiB per-file limit", portable, MaxFileBytes>>20)
		}
		if info.Size() > MaxExpandedBytes-expanded {
			return nil, 0, fmt.Errorf("source exceeds the %d MiB expanded size limit", MaxExpandedBytes>>20)
		}
		expanded += info.Size()
		files = append(files, sourceFile{hostPath: hostPath, path: portable, info: info})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	return files, expanded, nil
}

func portablePath(value string) (string, error) {
	value = norm.NFC.String(value)
	if value == "" || strings.HasPrefix(value, "/") || strings.Contains(value, "\\") ||
		strings.Contains(value, "\x00") || strings.Contains(value, "://") || !utf8.ValidString(value) {
		return "", fmt.Errorf("source path %q is not a portable workspace path", value)
	}
	if len(value) > MaxPathBytes {
		return "", fmt.Errorf("source path %q exceeds the %d-byte path limit", value, MaxPathBytes)
	}
	parts := strings.Split(value, "/")
	if len(parts) > maxPathParts {
		return "", fmt.Errorf("source path %q exceeds the workspace depth limit", value)
	}
	for index, part := range parts {
		if part == "" || part == "." || part == ".." ||
			(index == 0 && len(part) >= 2 && part[1] == ':' && isASCIIAlpha(part[0])) {
			return "", fmt.Errorf("source path %q is not a portable workspace path", value)
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return "", fmt.Errorf("source path %q contains a control character", value)
			}
		}
	}
	return value, nil
}

func copyRegularFile(destination io.Writer, file sourceFile) error {
	source, err := os.Open(file.hostPath)
	if err != nil {
		return fmt.Errorf("open source member %s: %w", file.path, err)
	}
	defer source.Close()
	opened, err := source.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(file.info, opened) {
		return fmt.Errorf("source member %s changed while packaging", file.path)
	}
	written, err := io.Copy(destination, source)
	if err != nil {
		return archiveError(err)
	}
	if written != opened.Size() {
		return fmt.Errorf("source member %s changed while packaging", file.path)
	}
	return nil
}

type limitedBuffer struct {
	bytes.Buffer
	maximum int
}

func (b *limitedBuffer) Write(value []byte) (int, error) {
	remaining := b.maximum - b.Len()
	if remaining <= 0 {
		return 0, ErrArchiveTooLarge
	}
	if len(value) > remaining {
		written, _ := b.Buffer.Write(value[:remaining])
		return written, ErrArchiveTooLarge
	}
	return b.Buffer.Write(value)
}

func archiveError(err error) error {
	if errors.Is(err, ErrArchiveTooLarge) {
		return ErrArchiveTooLarge
	}
	return fmt.Errorf("build source ZIP: %w", err)
}

func splitNUL(value []byte) []string {
	value = bytes.TrimSuffix(value, []byte{0})
	if len(value) == 0 {
		return nil
	}
	parts := bytes.Split(value, []byte{0})
	result := make([]string, len(parts))
	for index, part := range parts {
		result[index] = string(part)
	}
	return result
}

func joinNUL(values []string) []byte {
	var result bytes.Buffer
	for _, value := range values {
		result.WriteString(value)
		result.WriteByte(0)
	}
	return result.Bytes()
}

func isASCIIAlpha(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z'
}
