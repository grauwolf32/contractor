package performance

import "github.com/jackc/pgx/v5/pgxpool"

type PoolReader func() (Pool, Reason)

// WorkingPoolReader reads in-memory statistics only. It never acquires a
// connection or executes SQL, and remains usable during a database outage.
func WorkingPoolReader(pool *pgxpool.Pool) PoolReader {
	return func() (Pool, Reason) {
		if pool == nil {
			return Pool{}, DatabaseUnavailable
		}
		stat := pool.Stat()
		acquired, idle, total, max := uint32(stat.AcquiredConns()), uint32(stat.IdleConns()), uint32(stat.TotalConns()), uint32(stat.MaxConns())
		count, empty, cancelled := uint64(stat.AcquireCount()), uint64(stat.EmptyAcquireCount()), uint64(stat.CanceledAcquireCount())
		duration, wait := stat.AcquireDuration().Seconds(), stat.EmptyAcquireWaitTime().Seconds()
		return Pool{AcquiredConnections: &acquired, IdleConnections: &idle, TotalConnections: &total, MaxConnections: &max, AcquireCount: &count, AcquireDurationSeconds: &duration, EmptyAcquireCount: &empty, EmptyAcquireWaitSeconds: &wait, CanceledAcquireCount: &cancelled}, ""
	}
}
