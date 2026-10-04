package sourcebundle

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/sourcezip"
)

// Source ZIP limits and member paths match the runtime source_analysis toolset
// and Git imports.
const (
	MaxArchiveBytes  = sourcezip.MaxArchiveBytes
	MaxEntries       = sourcezip.MaxEntries
	MaxFileBytes     = sourcezip.MaxFileBytes
	MaxExpandedBytes = sourcezip.MaxArchiveBytes
	MaxPathBytes     = sourcezip.MaxPathBytes
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
	relativePath string
	path         string
	info         fs.FileInfo
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
	sourceRoot, err := os.OpenRoot(root)
	if err != nil {
		return Bundle{}, fmt.Errorf("open source directory: %w", err)
	}
	defer sourceRoot.Close()
	openedRoot, err := sourceRoot.Stat(".")
	if err != nil || !os.SameFile(info, openedRoot) {
		return Bundle{}, errors.New("source directory changed while packaging")
	}

	paths, skipped, err := candidatePaths(root, options.IncludeIgnored)
	if err != nil {
		return Bundle{}, err
	}
	paths, err = applyContractorIgnore(root, paths)
	if err != nil {
		return Bundle{}, err
	}
	files, expanded, err := inspectFiles(sourceRoot, paths)
	if err != nil {
		return Bundle{}, err
	}
	if len(files) == 0 {
		return Bundle{}, errors.New("source has no files to package after ignore filtering")
	}

	members := make([]sourcezip.Member, len(files))
	for index, file := range files {
		members[index] = sourcezip.Member{
			Name: file.path, Size: file.info.Size(),
			Write: func(destination io.Writer) error {
				return copyRegularFile(destination, sourceRoot, file)
			},
		}
	}
	payload, err := sourcezip.Encode(context.Background(), members, sourcezip.Options{CompressionLevel: 6})
	if err != nil {
		return Bundle{}, archiveError(err)
	}
	return Bundle{
		Data: payload, Files: len(files), ExpandedBytes: expanded,
		SHA256:              contentdigest.Bytes(payload),
		SkippedRepositories: skipped,
	}, nil
}

func candidatePaths(root string, includeIgnored bool) ([]string, int, error) {
	if !includeIgnored {
		if _, err := exec.LookPath("git"); err == nil {
			command := exec.Command("git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", ".")
			output, commandErr := command.Output()
			if commandErr == nil {
				// An unmerged index emits the same raw path once per stage. Package
				// the working-tree file once; distinct names still reach the NFC
				// collision check in inspectFiles.
				seen := make(map[string]struct{})
				unique := make([]string, 0)
				for _, path := range splitNUL(output) {
					if _, duplicate := seen[path]; duplicate {
						continue
					}
					seen[path] = struct{}{}
					unique = append(unique, path)
				}
				paths, skipped := skipNestedRepositories(root, unique)
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
	skipped := 0
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
		if strings.EqualFold(entry.Name(), ".git") {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			if portable == ".contractor" {
				return filepath.SkipDir
			}
			nested, err := containsGitMarker(hostPath)
			if err != nil {
				return err
			}
			if nested {
				skipped++
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
	return paths, skipped, nil
}

func containsGitMarker(directory string) (bool, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return false, err
	}
	for _, entry := range entries {
		if strings.EqualFold(entry.Name(), ".git") {
			return true, nil
		}
	}
	return false, nil
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

func inspectFiles(root *os.Root, paths []string) ([]sourceFile, int64, error) {
	seen := make(map[string]struct{}, len(paths))
	files := make([]sourceFile, 0, len(paths))
	var expanded int64
	for _, candidate := range paths {
		candidate = strings.TrimPrefix(filepath.ToSlash(candidate), "./")
		portable, err := portablePath(candidate)
		if err != nil {
			return nil, 0, err
		}
		if containsGitComponent(portable) || portable == ".contractor" || strings.HasPrefix(portable, ".contractor/") {
			continue
		}
		if _, duplicate := seen[portable]; duplicate {
			return nil, 0, fmt.Errorf("source paths collide after normalization: %s", portable)
		}
		seen[portable] = struct{}{}
		parts := strings.Split(candidate, "/")
		missingParent := false
		parent := ""
		for _, part := range parts[:len(parts)-1] {
			parent = filepath.Join(parent, part)
			parentInfo, parentErr := root.Lstat(parent)
			if errors.Is(parentErr, os.ErrNotExist) {
				missingParent = true
				break
			}
			if parentErr != nil {
				return nil, 0, fmt.Errorf("inspect source member %s: %w", filepath.ToSlash(parent), parentErr)
			}
			if parentInfo.Mode()&os.ModeSymlink != 0 {
				return nil, 0, fmt.Errorf("source member %s is a symbolic link, which source push does not upload; exclude it with .contractorignore", filepath.ToSlash(parent))
			}
			if !parentInfo.IsDir() {
				return nil, 0, fmt.Errorf("source member %s is not a directory; exclude it with .contractorignore", filepath.ToSlash(parent))
			}
		}
		if missingParent {
			continue
		}
		relativePath := filepath.FromSlash(candidate)
		info, err := root.Lstat(relativePath)
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
		files = append(files, sourceFile{relativePath: relativePath, path: portable, info: info})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].path < files[j].path })
	return files, expanded, nil
}

func containsGitComponent(path string) bool {
	for _, part := range strings.Split(path, "/") {
		if strings.EqualFold(part, ".git") {
			return true
		}
	}
	return false
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
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return "", fmt.Errorf("source path %q is not a portable workspace path", value)
		}
		if strings.Contains(part, ":") {
			return "", fmt.Errorf("source path %q contains ':' in a path component; exclude it with .contractorignore", value)
		}
		for _, character := range part {
			if character < 0x20 || character == 0x7f {
				return "", fmt.Errorf("source path %q contains a control character", value)
			}
		}
	}
	return value, nil
}

func copyRegularFile(destination io.Writer, root *os.Root, file sourceFile) error {
	source, err := root.Open(file.relativePath)
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

func archiveError(err error) error {
	if errors.Is(err, sourcezip.ErrLimit) {
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
