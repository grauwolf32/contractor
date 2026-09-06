package cli

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/grauwolf32/contractor/internal/publicclient"
)

var contextNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`)

type ServerContext struct {
	Server    string `json:"server"`
	CAFile    string `json:"caFile,omitempty"`
	TokenFile string `json:"tokenFile,omitempty"`
	AllowHTTP bool   `json:"allowHttp,omitempty"`
}

type ContextConfig struct {
	CurrentContext string                   `json:"currentContext,omitempty"`
	Contexts       map[string]ServerContext `json:"contexts"`
}

type ContextStore struct {
	path string
}

func NewContextStore(path string) *ContextStore {
	return &ContextStore{path: filepath.Clean(path)}
}

func DefaultContextPath(getenv func(string) string) (string, error) {
	if value := strings.TrimSpace(getenv("CONTRACTOR_CLI_CONFIG")); value != "" {
		absolute, err := filepath.Abs(value)
		if err != nil {
			return "", fmt.Errorf("resolve CONTRACTOR_CLI_CONFIG: %w", err)
		}
		return absolute, nil
	}
	directory, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolve user configuration directory: %w", err)
	}
	return filepath.Join(directory, "contractor", "config.json"), nil
}

func (s *ContextStore) Path() string { return s.path }

func (s *ContextStore) Load() (ContextConfig, error) {
	info, err := os.Lstat(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return ContextConfig{Contexts: map[string]ServerContext{}}, nil
	}
	if err != nil {
		return ContextConfig{}, fmt.Errorf("inspect CLI context file: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() > 1024*1024 {
		return ContextConfig{}, errors.New("CLI context path must be a regular non-symlink file no larger than 1 MiB")
	}
	file, err := os.Open(s.path)
	if err != nil {
		return ContextConfig{}, fmt.Errorf("open CLI context file: %w", err)
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !os.SameFile(info, opened) {
		return ContextConfig{}, errors.New("CLI context file changed while opening it")
	}
	decoder := json.NewDecoder(io.LimitReader(file, 1024*1024+1))
	decoder.DisallowUnknownFields()
	var config ContextConfig
	if err := decoder.Decode(&config); err != nil {
		return ContextConfig{}, fmt.Errorf("decode CLI context file: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return ContextConfig{}, errors.New("CLI context file contains trailing data")
	}
	if config.Contexts == nil {
		config.Contexts = map[string]ServerContext{}
	}
	if err := validateContextConfig(config); err != nil {
		return ContextConfig{}, err
	}
	return config, nil
}

func (s *ContextStore) Save(config ContextConfig) error {
	if config.Contexts == nil {
		config.Contexts = map[string]ServerContext{}
	}
	if err := validateContextConfig(config); err != nil {
		return err
	}
	if info, err := os.Lstat(s.path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return errors.New("CLI context path must be a regular non-symlink file")
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect CLI context path: %w", err)
	}
	directory := filepath.Dir(s.path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return fmt.Errorf("create CLI context directory: %w", err)
	}
	payload, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("encode CLI contexts: %w", err)
	}
	payload = append(payload, '\n')
	temporary, err := os.CreateTemp(directory, ".config-*.tmp")
	if err != nil {
		return fmt.Errorf("create temporary CLI context file: %w", err)
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("secure temporary CLI context file: %w", err)
	}
	if _, err := temporary.Write(payload); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write CLI contexts: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync CLI contexts: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close CLI contexts: %w", err)
	}
	if err := os.Rename(temporaryPath, s.path); err != nil {
		return fmt.Errorf("replace CLI context file: %w", err)
	}
	return nil
}

func (s *ContextStore) Put(name string, context ServerContext, makeCurrent bool) (ContextConfig, error) {
	config, err := s.Load()
	if err != nil {
		return ContextConfig{}, err
	}
	if !contextNamePattern.MatchString(name) {
		return ContextConfig{}, errors.New("context name must contain 1 through 64 letters, digits, dots, underscores, or hyphens")
	}
	server, err := publicclient.NormalizeServer(context.Server, context.AllowHTTP)
	if err != nil {
		return ContextConfig{}, err
	}
	context.Server = server
	for label, path := range map[string]*string{"CA file": &context.CAFile, "token file": &context.TokenFile} {
		if strings.TrimSpace(*path) == "" {
			*path = ""
			continue
		}
		absolute, err := filepath.Abs(filepath.Clean(*path))
		if err != nil {
			return ContextConfig{}, fmt.Errorf("resolve %s: %w", strings.ToLower(label), err)
		}
		*path = absolute
	}
	config.Contexts[name] = context
	if makeCurrent || config.CurrentContext == "" {
		config.CurrentContext = name
	}
	if err := s.Save(config); err != nil {
		return ContextConfig{}, err
	}
	return config, nil
}

func (s *ContextStore) Use(name string) (ContextConfig, error) {
	config, err := s.Load()
	if err != nil {
		return ContextConfig{}, err
	}
	if _, found := config.Contexts[name]; !found {
		return ContextConfig{}, fmt.Errorf("context %q does not exist", name)
	}
	config.CurrentContext = name
	if err := s.Save(config); err != nil {
		return ContextConfig{}, err
	}
	return config, nil
}

func (s *ContextStore) Remove(name string) (ContextConfig, error) {
	config, err := s.Load()
	if err != nil {
		return ContextConfig{}, err
	}
	if _, found := config.Contexts[name]; !found {
		return ContextConfig{}, fmt.Errorf("context %q does not exist", name)
	}
	delete(config.Contexts, name)
	if config.CurrentContext == name {
		config.CurrentContext = ""
	}
	if err := s.Save(config); err != nil {
		return ContextConfig{}, err
	}
	return config, nil
}

func (c ContextConfig) Resolve(name string) (string, ServerContext, error) {
	if name == "" {
		name = c.CurrentContext
	}
	if name == "" {
		return "", ServerContext{}, errors.New("no context selected; use context add/use or --server")
	}
	context, found := c.Contexts[name]
	if !found {
		return "", ServerContext{}, fmt.Errorf("context %q does not exist", name)
	}
	return name, context, nil
}

func (c ContextConfig) Names() []string {
	result := make([]string, 0, len(c.Contexts))
	for name := range c.Contexts {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func validateContextConfig(config ContextConfig) error {
	for name, context := range config.Contexts {
		if !contextNamePattern.MatchString(name) {
			return fmt.Errorf("context file contains invalid context name %q", name)
		}
		if _, err := publicclient.NormalizeServer(context.Server, context.AllowHTTP); err != nil {
			return fmt.Errorf("context %q: %w", name, err)
		}
		if context.CAFile != "" && !filepath.IsAbs(context.CAFile) {
			return fmt.Errorf("context %q contains a non-absolute CA file", name)
		}
		if context.TokenFile != "" && !filepath.IsAbs(context.TokenFile) {
			return fmt.Errorf("context %q contains a non-absolute token file", name)
		}
	}
	if config.CurrentContext != "" {
		if _, found := config.Contexts[config.CurrentContext]; !found {
			return fmt.Errorf("current context %q does not exist", config.CurrentContext)
		}
	}
	return nil
}
