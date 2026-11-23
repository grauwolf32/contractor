package auth

import (
	"fmt"
	"math"
	"net/netip"
	"sync"
	"time"
)

const globalFailureBackoff = time.Second

type RateLimitError struct {
	RetryAfter time.Duration
}

func (e *RateLimitError) Error() string { return ErrRateLimited.Error() }
func (e *RateLimitError) Unwrap() error { return ErrRateLimited }

type limiterTicket struct{ id uint64 }

type loginAttempt struct {
	id uint64
	ip string
	at time.Time
}

type failureLimiter struct {
	mu                sync.Mutex
	window            time.Duration
	perIPLimit        int
	globalLimit       int
	nextID            uint64
	attempts          []loginAttempt
	globalNextAllowed time.Time
}

func newFailureLimiter(window time.Duration, perIPLimit, globalLimit int) (*failureLimiter, error) {
	if window <= 0 || window > time.Minute || perIPLimit <= 0 || globalLimit < perIPLimit {
		return nil, fmt.Errorf("invalid login limiter configuration")
	}
	return &failureLimiter{window: window, perIPLimit: perIPLimit, globalLimit: globalLimit}, nil
}

func (l *failureLimiter) begin(ip string, now time.Time) (limiterTicket, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.purge(now)
	ip = peerBucket(ip)
	ipCount := 0
	var ipOldest time.Time
	for _, attempt := range l.attempts {
		if attempt.ip != ip {
			continue
		}
		ipCount++
		if ipOldest.IsZero() || attempt.at.Before(ipOldest) {
			ipOldest = attempt.at
		}
	}
	limitedUntil := time.Time{}
	if ipCount >= l.perIPLimit {
		limitedUntil = ipOldest.Add(l.window)
	}
	if len(l.attempts) >= l.globalLimit && now.Before(l.globalNextAllowed) &&
		(limitedUntil.IsZero() || l.globalNextAllowed.After(limitedUntil)) {
		limitedUntil = l.globalNextAllowed
	}
	if !limitedUntil.IsZero() {
		retry := limitedUntil.Sub(now)
		if retry < time.Second {
			retry = time.Second
		}
		retry = time.Duration(math.Ceil(retry.Seconds())) * time.Second
		if retry > time.Minute {
			retry = time.Minute
		}
		return limiterTicket{}, &RateLimitError{RetryAfter: retry}
	}
	if len(l.attempts) >= l.globalLimit {
		// Once failures cross the global threshold, permit one fresh attempt
		// per backoff interval instead of locking out every peer for a minute.
		l.globalNextAllowed = now.Add(globalFailureBackoff)
	} else {
		l.globalNextAllowed = time.Time{}
	}
	l.nextID++
	ticket := limiterTicket{id: l.nextID}
	l.attempts = append(l.attempts, loginAttempt{id: ticket.id, ip: ip, at: now})
	return ticket, nil
}

func peerBucket(ip string) string {
	addr, err := netip.ParseAddr(ip)
	if err != nil {
		return ip
	}
	addr = addr.Unmap()
	if addr.Is6() {
		return netip.PrefixFrom(addr, 64).Masked().Addr().String()
	}
	return addr.String()
}

func (l *failureLimiter) complete(ticket limiterTicket, failed bool) {
	if failed {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for index, attempt := range l.attempts {
		if attempt.id == ticket.id {
			l.attempts = append(l.attempts[:index], l.attempts[index+1:]...)
			return
		}
	}
}

func (l *failureLimiter) purge(now time.Time) {
	kept := l.attempts[:0]
	for _, attempt := range l.attempts {
		if now.Before(attempt.at.Add(l.window)) {
			kept = append(kept, attempt)
		}
	}
	l.attempts = kept
}
