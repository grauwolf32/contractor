package auth

import (
	"fmt"
	"math"
	"sync"
	"time"
)

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
	mu          sync.Mutex
	window      time.Duration
	perIPLimit  int
	globalLimit int
	nextID      uint64
	attempts    []loginAttempt
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
	if len(l.attempts) >= l.globalLimit {
		limitedUntil = l.attempts[0].at.Add(l.window)
		for _, attempt := range l.attempts[1:] {
			candidate := attempt.at.Add(l.window)
			if candidate.Before(limitedUntil) {
				limitedUntil = candidate
			}
		}
	}
	if ipCount >= l.perIPLimit {
		candidate := ipOldest.Add(l.window)
		if limitedUntil.IsZero() || candidate.After(limitedUntil) {
			limitedUntil = candidate
		}
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
	l.nextID++
	ticket := limiterTicket{id: l.nextID}
	l.attempts = append(l.attempts, loginAttempt{id: ticket.id, ip: ip, at: now})
	return ticket, nil
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
