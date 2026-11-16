package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
	"github.com/grauwolf32/contractor/internal/helpers/randomid"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"golang.org/x/sys/unix"
)

const catalogLockName = "catalog.lock.json"
const catalogLockSchema = "contractor.catalog-lock/v1"

type catalogLock struct {
	Schema  string            `json:"schema"`
	Entries map[string]string `json:"entries"`
}

func readCatalogLock(root string) (catalogLock, bool, error) {
	data, err := readCatalogFile(root, catalogLockName)
	if errors.Is(err, os.ErrNotExist) {
		return catalogLock{Schema: catalogLockSchema, Entries: map[string]string{}}, false, nil
	}
	if err != nil {
		return catalogLock{}, false, err
	}
	var lock catalogLock
	if err := strictjson.Decode(data, &lock); err != nil {
		return lock, true, fmt.Errorf("invalid catalog lock: %w", err)
	}
	if lock.Schema != catalogLockSchema || lock.Entries == nil {
		return lock, true, errors.New("invalid catalog lock schema or entries")
	}
	for key, digest := range lock.Entries {
		kind, identity, ok := strings.Cut(key, "/")
		if !ok || !contentdigest.Valid(digest) {
			return lock, true, errors.New("invalid catalog lock entry")
		}
		switch kind {
		case "instructions":
			if _, err := validateInstructionRef(key); err != nil {
				return lock, true, err
			}
		case "workflows", "agent-templates", "model-policies", "llm-gateways", "audit-profiles":
			if _, err := resolved.ParseSelector(identity); err != nil {
				return lock, true, err
			}
		default:
			return lock, true, errors.New("unknown catalog lock kind")
		}
	}
	return lock, true, nil
}

func (s *Snapshot) catalogDigests() (map[string]string, error) {
	entries := map[string]string{}
	add := func(kind, identity, digest string) {
		if s.configurationSource(ConfigurationKind(kind), identity) == ConfigurationSourceOperator {
			entries[kind+"/"+identity] = digest
		}
	}
	for id, v := range s.workflows {
		digest, err := contentdigest.JSON(v)
		if err != nil {
			return nil, err
		}
		add("workflows", id, digest)
	}
	for id, v := range s.templates {
		add("agent-templates", id, v.Ref.Digest)
	}
	for id, v := range s.policies {
		add("model-policies", id, v.Ref.Digest)
	}
	for id, v := range s.gateways {
		add("llm-gateways", id, v.Ref.Digest)
	}
	for id, v := range s.auditProfiles {
		digest, err := v.Digest()
		if err != nil {
			return nil, err
		}
		add("audit-profiles", id, digest)
	}
	for ref, v := range s.instructions {
		if s.configurationSource(ConfigurationKind("instructions"), ref) == ConfigurationSourceOperator {
			entries[ref] = v.Digest
		}
	}
	return entries, nil
}

// ValidateCatalogLock checks an existing operator lock. Release validation sets
// required; custom operator roots may deliberately have no lock. Managed
// publications retain their separate immutable identity mechanism.
func ValidateCatalogLock(root string, snapshot *Snapshot, required bool) error {
	_, err := checkCatalogLock(root, snapshot, required, false)
	return err
}

func checkCatalogLock(root string, snapshot *Snapshot, required, extend bool) (catalogLock, error) {
	lock, exists, err := readCatalogLock(root)
	if err != nil {
		return lock, err
	}
	if !exists && required {
		return lock, errors.New("catalog.lock.json is required")
	}
	if !exists && !extend {
		return lock, nil
	}
	current, err := snapshot.catalogDigests()
	if err != nil {
		return lock, err
	}
	for key, digest := range current {
		previous, known := lock.Entries[key]
		if known && previous != digest {
			return lock, fmt.Errorf("catalog lock drift at %s: publish a new identity", key)
		}
		if !known && !extend {
			return lock, fmt.Errorf("catalog identity %s is missing from the lock", key)
		}
		lock.Entries[key] = digest
	}
	// Retired entries remain so removing and later reusing an identity cannot
	// silently change its content. Generation only extends this history.
	return lock, nil
}

// UpdateCatalogLock validates the complete roots and atomically extends the
// operator lock. Like catalog edits, this is an offline, single-writer action.
func UpdateCatalogLock(operatorRoot, managedRoot string, checks ...OperatorRootCheck) (*Snapshot, error) {
	snapshot, err := loadUnionReadOnly(operatorRoot, managedRoot, true, checks...)
	if err != nil {
		return nil, err
	}
	lock, err := checkCatalogLock(operatorRoot, snapshot, false, true)
	if err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(lock, "", "  ")
	if err != nil {
		return nil, err
	}
	data = append(data, '\n')
	if len(data) > maximumCatalogFileBytes {
		return nil, errors.New("catalog lock exceeds its 4 MiB bound")
	}
	if err := writeCatalogLock(operatorRoot, data); err != nil {
		return nil, err
	}
	return snapshot, nil
}

func writeCatalogLock(root string, data []byte) error {
	directory, err := unix.Open(root, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer unix.Close(directory)
	name, err := randomid.New("catalog-lock")
	if err != nil {
		return err
	}
	name = "." + name + ".tmp"
	descriptor, err := unix.Openat(directory, name, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o644)
	if err != nil {
		return err
	}
	file := os.NewFile(uintptr(descriptor), name)
	defer file.Close()
	defer unix.Unlinkat(directory, name, 0)
	if _, err := file.Write(data); err != nil {
		return err
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := unix.Renameat(directory, name, directory, catalogLockName); err != nil {
		return err
	}
	return unix.Fsync(directory)
}
