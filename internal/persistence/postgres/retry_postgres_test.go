package postgres

import (
	"context"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

func TestPostgresDeadlockRetryReplaysWholeTransaction(t *testing.T) {
	url := os.Getenv("CONTRACTOR_TEST_DATABASE_URL")
	if url == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	pool, _ := isolatedPools(t, ctx, url)
	if _, err := pool.Exec(ctx, `CREATE TABLE retry_counters (id integer PRIMARY KEY, value integer NOT NULL); INSERT INTO retry_counters VALUES (1, 0), (2, 0)`); err != nil {
		t.Fatal(err)
	}
	var acquired sync.WaitGroup
	acquired.Add(2)
	attempts := make([]int, 2)
	errorsFound := make(chan error, 2)
	for index := range 2 {
		go func() {
			errorsFound <- InTxWithRetry(ctx, pool, pgx.TxOptions{}, func(tx pgx.Tx) error {
				attempts[index]++
				if _, err := tx.Exec(ctx, `SET LOCAL deadlock_timeout = '50ms'`); err != nil {
					return err
				}
				if _, err := tx.Exec(ctx, `UPDATE retry_counters SET value=value+1 WHERE id=$1`, index+1); err != nil {
					return err
				}
				if attempts[index] == 1 {
					acquired.Done()
					acquired.Wait()
				}
				_, err := tx.Exec(ctx, `UPDATE retry_counters SET value=value+1 WHERE id=$1`, 2-index)
				return err
			})
		}()
	}
	for range 2 {
		if err := <-errorsFound; err != nil {
			t.Fatal(err)
		}
	}
	var sum int
	if err := pool.QueryRow(ctx, `SELECT sum(value) FROM retry_counters`).Scan(&sum); err != nil || sum != 4 || attempts[0]+attempts[1] != 3 {
		t.Fatalf("deadlock recovery sum=%d attempts=%v error=%v", sum, attempts, err)
	}
}
