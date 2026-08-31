package runtimeconfig

import (
	"context"
	"errors"
	"time"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PublishResult struct {
	Version  Version
	Replayed bool
}

type Publisher struct {
	pool        *pgxpool.Pool
	resolver    GatewayResolver
	credentials RuntimeCredentialCatalog
	now         func() time.Time
}

type PublisherOptions struct {
	Pool               *pgxpool.Pool
	GatewayResolver    GatewayResolver
	RuntimeCredentials RuntimeCredentialCatalog
	Now                func() time.Time
}

func NewPublisher(options PublisherOptions) (*Publisher, error) {
	if options.Pool == nil || options.RuntimeCredentials == nil {
		return nil, errors.New("RuntimeConfig publisher dependencies are incomplete")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Publisher{
		pool: options.Pool, resolver: options.GatewayResolver,
		credentials: options.RuntimeCredentials,
		now:         options.Now,
	}, nil
}

// Publish deliberately performs its first durable replay lookup before using
// the mutable Gateway resolver.
func (p *Publisher) Publish(ctx context.Context, document []byte, idempotencyKey, actor string) (PublishResult, error) {
	if p == nil || p.pool == nil || p.credentials == nil {
		return PublishResult{}, invalid("RuntimeConfig publisher is not configured")
	}
	if !validActor(actor) {
		return PublishResult{}, invalid("publication actor is invalid")
	}
	prepared, err := PreparePublication(document)
	if err != nil {
		return PublishResult{}, err
	}
	keyDigest, err := DigestIdempotencyKey(idempotencyKey)
	if err != nil {
		return PublishResult{}, err
	}
	repository := NewRepository(p.pool)
	if replay, found, err := lookupReplay(ctx, repository, keyDigest, prepared.RequestDigest()); err != nil {
		return PublishResult{}, err
	} else if found {
		return replay, nil
	}

	var result PublishResult
	err = p.credentials.WithCredentialReferences(ctx, func() error {
		// A deletion may have committed between the lock-free fast replay lookup
		// and acquiring the shared reference barrier. Replay still wins before
		// any mutable validation or Gateway resolution.
		if replay, found, replayErr := lookupReplay(ctx, repository, keyDigest, prepared.RequestDigest()); replayErr != nil {
			return replayErr
		} else if found {
			result = replay
			return nil
		}
		version, resolveErr := prepared.Resolve(ctx, p.resolver)
		if resolveErr != nil {
			return resolveErr
		}
		if validationErr := validateSpecRuntimeCredentials(ctx, version.Spec, p.credentials); validationErr != nil {
			return validationErr
		}
		version.ActorID = actor
		version.CreatedAt = p.now()
		publication := Publication{
			IdempotencyKeyDigest: keyDigest, RequestDigest: prepared.RequestDigest(), Ref: version.Ref,
			ActorID: actor, PublishedAt: version.CreatedAt,
		}

		return persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
			txRepository := NewRepository(tx)
			if replay, found, replayErr := lookupReplay(ctx, txRepository, keyDigest, prepared.RequestDigest()); replayErr != nil {
				return replayErr
			} else if found {
				result = replay
				return nil
			}
			_, insertErr := txRepository.InsertVersion(ctx, version)
			if insertErr != nil {
				if errors.Is(insertErr, ErrConflict) {
					if replay, found, replayErr := lookupReplay(ctx, txRepository, keyDigest, prepared.RequestDigest()); replayErr != nil {
						return replayErr
					} else if found {
						result = replay
						return nil
					}
				}
				return insertErr
			}
			inserted, insertErr := txRepository.InsertPublication(ctx, publication)
			if insertErr != nil {
				return insertErr
			}
			if !inserted {
				replay, found, replayErr := lookupReplay(ctx, txRepository, keyDigest, prepared.RequestDigest())
				if replayErr != nil {
					return replayErr
				}
				if !found {
					return ErrConflict
				}
				result = replay
				return nil
			}
			result = PublishResult{Version: version}
			return nil
		})
	})
	if err != nil {
		return PublishResult{}, err
	}
	return result, nil
}

type publicationReader interface {
	GetPublication(context.Context, string) (Publication, error)
	GetVersionByRef(context.Context, Ref) (Version, error)
}

func lookupReplay(ctx context.Context, repository publicationReader, keyDigest, requestDigest string) (PublishResult, bool, error) {
	publication, err := repository.GetPublication(ctx, keyDigest)
	if errors.Is(err, ErrNotFound) {
		return PublishResult{}, false, nil
	}
	if err != nil {
		return PublishResult{}, false, err
	}
	if publication.RequestDigest != requestDigest {
		return PublishResult{}, false, ErrConflict
	}
	version, err := repository.GetVersionByRef(ctx, publication.Ref)
	if err != nil {
		return PublishResult{}, false, err
	}
	return PublishResult{Version: version, Replayed: true}, true, nil
}
