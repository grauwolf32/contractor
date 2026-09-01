package credentials

import (
	"context"
	"crypto/hmac"
	"errors"
	"sync"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

const maximumRuntimeCredentialReferences = 128

type RuntimeCredentialServiceOptions struct {
	Pool    *pgxpool.Pool
	Cipher  *TokenCipher
	Usage   RuntimeCredentialUsageChecker
	Barrier *LifecycleBarrier
	Now     func() time.Time
}

type RuntimeCredentialService struct {
	pool       *pgxpool.Pool
	repository *RuntimeCredentialRepository
	cipher     *TokenCipher
	usage      RuntimeCredentialUsageChecker
	barrier    *LifecycleBarrier
	now        func() time.Time
	mu         sync.Mutex
}

func NewRuntimeCredentialService(options RuntimeCredentialServiceOptions) (*RuntimeCredentialService, error) {
	if options.Pool == nil || options.Usage == nil || options.Barrier == nil {
		return nil, errors.New("Runtime credential lifecycle dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &RuntimeCredentialService{
		pool: options.Pool, repository: NewRuntimeCredentialRepository(options.Pool),
		cipher: options.Cipher, usage: options.Usage, barrier: options.Barrier, now: options.Now,
	}, nil
}

func (s *RuntimeCredentialService) List(
	ctx context.Context, afterCredentialID string, limit int,
) ([]RuntimeCredentialMetadata, error) {
	if s == nil || s.repository == nil {
		return nil, errors.New("Runtime credential service is not configured")
	}
	return s.repository.ListActiveMetadata(ctx, afterCredentialID, limit)
}

func (s *RuntimeCredentialService) Get(ctx context.Context, credentialID string) (RuntimeCredentialMetadata, error) {
	if s == nil || s.repository == nil {
		return RuntimeCredentialMetadata{}, errors.New("Runtime credential service is not configured")
	}
	record, err := s.repository.GetActiveRecord(ctx, credentialID)
	if err != nil {
		return RuntimeCredentialMetadata{}, err
	}
	return record.Metadata, nil
}

func (s *RuntimeCredentialService) Create(
	ctx context.Context, request RuntimeCredentialCreateRequest,
) (RuntimeCredentialCreateResult, error) {
	if s == nil || s.repository == nil || s.barrier == nil {
		return RuntimeCredentialCreateResult{}, errors.New("Runtime credential service is not configured")
	}
	if validateRuntimeCredentialID(request.CredentialID) != nil || !validRuntimeCredentialKind(request.Material.kind) ||
		len(request.Material.canonical) == 0 || len(request.Material.canonical) > MaximumRuntimePlaintextBytes ||
		!validActorID(request.ActorID) {
		return RuntimeCredentialCreateResult{}, runtimeInvalid("Runtime credential create request is invalid")
	}
	keyDigest, err := runtimeCredentialKeyDigest(request.IdempotencyKey)
	if err != nil {
		return RuntimeCredentialCreateResult{}, err
	}
	if s.cipher == nil {
		return RuntimeCredentialCreateResult{}, ErrKeyUnavailable
	}
	requestMAC, err := s.cipher.RuntimeCredentialRequestMAC(request.CredentialID, request.Material)
	if err != nil {
		return RuntimeCredentialCreateResult{}, err
	}
	defer wipeBytes(requestMAC)

	s.mu.Lock()
	defer s.mu.Unlock()
	var result RuntimeCredentialCreateResult
	err = s.barrier.WithCredentialMutation(ctx, func() error {
		if replay, found, replayErr := s.creationReplay(ctx, keyDigest, requestMAC, request.CredentialID, request.Material.kind); replayErr != nil {
			return replayErr
		} else if found {
			result = replay
			return nil
		}

		envelope, sealErr := s.cipher.SealRuntimeCredential(request.CredentialID, request.Material)
		if sealErr != nil {
			return sealErr
		}
		createdAt := runtimeDatabaseTime(s.now())
		record := RuntimeCredentialRecord{
			Metadata: RuntimeCredentialMetadata{
				CredentialID: request.CredentialID, Kind: request.Material.kind,
				CreatedBy: request.ActorID, CreatedAt: createdAt,
			},
			Envelope: envelope,
		}
		creation := RuntimeCredentialCreation{
			IdempotencyKeyDigest: keyDigest, RequestMAC: append([]byte(nil), requestMAC...),
			CredentialID: request.CredentialID, Kind: request.Material.kind,
			ActorID: request.ActorID, CreatedAt: createdAt,
		}
		defer wipeBytes(creation.RequestMAC)
		transactionErr := persistencepostgres.InTx(ctx, s.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			repository := NewRuntimeCredentialRepository(tx)
			if _, insertErr := repository.InsertRecord(ctx, record); insertErr != nil {
				if errors.Is(insertErr, ErrRuntimeCredentialConflict) {
					if replay, found, replayErr := s.creationReplayWithRepository(
						ctx, repository, keyDigest, requestMAC, request.CredentialID, request.Material.kind,
					); replayErr != nil {
						return replayErr
					} else if found {
						result = replay
						return nil
					}
				}
				return insertErr
			}
			inserted, insertErr := repository.InsertCreation(ctx, creation)
			if insertErr != nil {
				return insertErr
			}
			if !inserted {
				replay, found, replayErr := s.creationReplayWithRepository(
					ctx, repository, keyDigest, requestMAC, request.CredentialID, request.Material.kind,
				)
				if replayErr != nil {
					return replayErr
				}
				if !found {
					return ErrRuntimeCredentialConflict
				}
				result = replay
				return nil
			}
			result = RuntimeCredentialCreateResult{Credential: record.Metadata}
			return nil
		})
		return safeRuntimeCredentialStoreError(transactionErr)
	})
	if err != nil {
		return RuntimeCredentialCreateResult{}, err
	}
	return result, nil
}

// ValidateRuntimeCredential checks one active record without taking the
// lifecycle fence itself. Callers that persist a reference must invoke it from
// the same catalog's WithCredentialReferences callback; RuntimeConfig's
// publisher and binding service enforce that ordering.
func (s *RuntimeCredentialService) ValidateRuntimeCredential(
	ctx context.Context, credentialID string, allowedKinds ...string,
) error {
	if s == nil || s.repository == nil {
		return errors.New("Runtime credential service is not configured")
	}
	record, err := s.repository.GetActiveRecord(ctx, credentialID)
	if err != nil {
		return err
	}
	return validateAllowedRuntimeKinds(record.Metadata.Kind, allowedKinds)
}

func (s *RuntimeCredentialService) WithCredentialReferences(ctx context.Context, fn func() error) error {
	if s == nil || s.barrier == nil {
		return errors.New("Runtime credential service is not configured")
	}
	return s.barrier.WithCredentialReferences(ctx, fn)
}

// Use decrypts only after the selected consumer and expected kind are known,
// holds the shared lifecycle fence for the synchronous consumer, and wipes the
// canonical plaintext before returning.
func (s *RuntimeCredentialService) Use(
	ctx context.Context,
	credentialID string,
	allowedKinds []string,
	consumer func(*RuntimeCredentialMaterial) error,
) error {
	if s == nil || s.repository == nil || s.barrier == nil || consumer == nil {
		return runtimeInvalid("Runtime credential consumer is invalid")
	}
	return s.barrier.WithCredentialReferences(ctx, func() error {
		record, err := s.repository.GetActiveRecord(ctx, credentialID)
		if err != nil {
			return err
		}
		if err := validateAllowedRuntimeKinds(record.Metadata.Kind, allowedKinds); err != nil {
			return err
		}
		if s.cipher == nil {
			return ErrKeyUnavailable
		}
		material, err := s.cipher.OpenRuntimeCredential(record.Metadata.CredentialID, record.Metadata.Kind, record.Envelope)
		if err != nil {
			return err
		}
		defer material.Destroy()
		if consumerErr := consumer(&material); consumerErr != nil {
			if contextError := ctx.Err(); contextError != nil {
				return contextError
			}
			return errors.New("Runtime credential consumer failed")
		}
		return nil
	})
}

// UsePlaintext is the allocation-bound adapter over Use. It exposes contract
// kinds so Scheduler need not depend on credential storage types, while the
// canonical plaintext remains inside the same synchronous lifecycle fence.
func (s *RuntimeCredentialService) UsePlaintext(
	ctx context.Context,
	credentialID string,
	allowedKinds []contracts.RuntimeCredentialKind,
	consumer func(contracts.RuntimeCredentialKind, []byte) error,
) error {
	if consumer == nil {
		return runtimeInvalid("Runtime credential consumer is invalid")
	}
	allowed := make([]string, len(allowedKinds))
	for index, kind := range allowedKinds {
		if err := kind.Validate(); err != nil {
			return runtimeInvalid("Runtime credential kind is invalid")
		}
		allowed[index] = string(kind)
	}
	return s.Use(ctx, credentialID, allowed, func(material *RuntimeCredentialMaterial) error {
		return material.WithPlaintext(func(kind RuntimeCredentialKind, plaintext []byte) error {
			return consumer(contracts.RuntimeCredentialKind(kind), plaintext)
		})
	})
}

func (s *RuntimeCredentialService) Delete(
	ctx context.Context, credentialID, actor string,
) (RuntimeCredentialDeleteResult, error) {
	if s == nil || s.repository == nil || s.barrier == nil ||
		validateRuntimeCredentialID(credentialID) != nil || !validActorID(actor) {
		return RuntimeCredentialDeleteResult{}, runtimeInvalid("Runtime credential delete request is invalid")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var result RuntimeCredentialDeleteResult
	err := s.barrier.WithCredentialMutation(ctx, func() error {
		if _, tombstoneErr := s.repository.GetTombstone(ctx, credentialID); tombstoneErr == nil {
			result.Replayed = true
			return nil
		} else if !errors.Is(tombstoneErr, ErrRuntimeCredentialNotFound) {
			return tombstoneErr
		}
		if _, getErr := s.repository.GetAnyRecord(ctx, credentialID); getErr != nil {
			return getErr
		}
		usage, usageErr := s.usage.InspectRuntimeCredentialUsage(ctx, credentialID, maximumRuntimeCredentialReferences)
		if usageErr != nil {
			if contextError := ctx.Err(); contextError != nil {
				return contextError
			}
			return errors.New("inspect Runtime credential references")
		}
		usage = normalizeRuntimeUsage(usage, maximumRuntimeCredentialReferences)
		if !usage.Empty() {
			return &RuntimeCredentialInUseError{Usage: usage}
		}
		inserted, insertErr := s.repository.InsertTombstone(ctx, RuntimeCredentialTombstone{
			CredentialID: credentialID, ActorID: actor, DeletedAt: runtimeDatabaseTime(s.now()),
		})
		if insertErr != nil {
			return insertErr
		}
		result.Replayed = !inserted
		return nil
	})
	if err != nil {
		return RuntimeCredentialDeleteResult{}, safeRuntimeCredentialStoreError(err)
	}
	return result, nil
}

func (s *RuntimeCredentialService) creationReplay(
	ctx context.Context,
	keyDigest string,
	requestMAC []byte,
	credentialID string,
	kind RuntimeCredentialKind,
) (RuntimeCredentialCreateResult, bool, error) {
	return s.creationReplayWithRepository(ctx, s.repository, keyDigest, requestMAC, credentialID, kind)
}

func (s *RuntimeCredentialService) creationReplayWithRepository(
	ctx context.Context,
	repository *RuntimeCredentialRepository,
	keyDigest string,
	requestMAC []byte,
	credentialID string,
	kind RuntimeCredentialKind,
) (RuntimeCredentialCreateResult, bool, error) {
	creation, err := repository.GetCreation(ctx, keyDigest)
	if errors.Is(err, ErrRuntimeCredentialNotFound) {
		return RuntimeCredentialCreateResult{}, false, nil
	}
	if err != nil {
		return RuntimeCredentialCreateResult{}, false, err
	}
	if creation.CredentialID != credentialID || creation.Kind != kind || !hmac.Equal(creation.RequestMAC, requestMAC) {
		return RuntimeCredentialCreateResult{}, false, ErrRuntimeCredentialConflict
	}
	record, err := repository.GetAnyRecord(ctx, creation.CredentialID)
	if err != nil || record.Metadata.Kind != creation.Kind {
		return RuntimeCredentialCreateResult{}, false, ErrRuntimeCredentialConflict
	}
	return RuntimeCredentialCreateResult{Credential: record.Metadata, Replayed: true}, true, nil
}
