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
	pool     *pgxpool.Pool
	resolver GatewayResolver
	now      func() time.Time
}

func NewPublisher(pool *pgxpool.Pool, resolver GatewayResolver, now func() time.Time) *Publisher {
	if now == nil {
		now = time.Now
	}
	return &Publisher{pool: pool, resolver: resolver, now: now}
}

// Publish deliberately performs its first durable replay lookup before using
// the mutable Gateway resolver.
func (p *Publisher) Publish(ctx context.Context, document []byte, idempotencyKey, actor string) (PublishResult, error) {
	if p == nil || p.pool == nil {
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

	version, err := prepared.Resolve(ctx, p.resolver)
	if err != nil {
		return PublishResult{}, err
	}
	version.ActorID = actor
	version.CreatedAt = p.now()
	publication := Publication{
		IdempotencyKeyDigest: keyDigest, RequestDigest: prepared.RequestDigest(), Ref: version.Ref,
		ActorID: actor, PublishedAt: version.CreatedAt,
	}

	var result PublishResult
	err = persistencepostgres.InTx(ctx, p.pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
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
