package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"time"
)

const (
	SecureCookieName   = "__Host-contractor_session"
	LoopbackCookieName = "contractor_loopback_session"
	SessionTokenBytes  = 32
	CSRFTokenBytes     = 32

	defaultIdleLimit      = 8 * time.Hour
	defaultAbsoluteLimit  = 24 * time.Hour
	defaultMaxSessions    = 8
	defaultFailureWindow  = time.Minute
	defaultPerIPFailures  = 5
	defaultGlobalFailures = 30
)

type Options struct {
	Now            func() time.Time
	Random         io.Reader
	IdleLimit      time.Duration
	AbsoluteLimit  time.Duration
	MaxSessions    int
	FailureWindow  time.Duration
	PerIPFailures  int
	GlobalFailures int
}

type Service struct {
	mu                       sync.Mutex
	principal                Principal
	hash                     passwordHash
	now                      func() time.Time
	random                   io.Reader
	idleLimit                time.Duration
	absoluteLimit            time.Duration
	maxSessions              int
	nextSequence             uint64
	sessions                 map[[sha256.Size]byte]storedSession
	limiter                  *failureLimiter
	nextRevocationSubscriber uint64
	revocationSubscribers    map[uint64]chan SessionHandle
}

type storedSession struct {
	csrfToken         string
	csrfDigest        [sha256.Size]byte
	createdAt         time.Time
	idleExpiresAt     time.Time
	absoluteExpiresAt time.Time
	sequence          uint64
}

type SessionHandle struct{ digest [sha256.Size]byte }

func (SessionHandle) String() string   { return "auth.SessionHandle([REDACTED])" }
func (SessionHandle) GoString() string { return "auth.SessionHandle([REDACTED])" }

type Session struct {
	Handle            SessionHandle
	Principal         Principal
	CSRFToken         string
	IdleExpiresAt     time.Time
	AbsoluteExpiresAt time.Time
	csrfDigest        [sha256.Size]byte
}

type Login struct {
	CookieValue string
	Session     Session
}

func (s Session) String() string {
	return fmt.Sprintf(
		"auth.Session{UserID:%q CSRFToken:[REDACTED] IdleExpiresAt:%s AbsoluteExpiresAt:%s}",
		s.Principal.UserID,
		s.IdleExpiresAt.Format(time.RFC3339),
		s.AbsoluteExpiresAt.Format(time.RFC3339),
	)
}

func (s Session) GoString() string { return s.String() }

func (l Login) String() string {
	return "auth.Login{CookieValue:[REDACTED] Session:" + l.Session.String() + "}"
}
func (l Login) GoString() string { return l.String() }

func (s *Service) String() string   { return "auth.Service{secrets:[REDACTED]}" }
func (s *Service) GoString() string { return s.String() }

func NewService(bootstrap Bootstrap, options Options) (*Service, error) {
	if !bootstrap.valid || bootstrap.Principal.UserID == "" {
		return nil, ErrInvalidBootstrap
	}
	principal, err := NewPrincipal(bootstrap.Principal.UserID, bootstrap.Principal.Username)
	if err != nil {
		return nil, ErrInvalidBootstrap
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	if options.Random == nil {
		options.Random = rand.Reader
	}
	if options.IdleLimit == 0 {
		options.IdleLimit = defaultIdleLimit
	}
	if options.AbsoluteLimit == 0 {
		options.AbsoluteLimit = defaultAbsoluteLimit
	}
	if options.MaxSessions == 0 {
		options.MaxSessions = defaultMaxSessions
	}
	if options.FailureWindow == 0 {
		options.FailureWindow = defaultFailureWindow
	}
	if options.PerIPFailures == 0 {
		options.PerIPFailures = defaultPerIPFailures
	}
	if options.GlobalFailures == 0 {
		options.GlobalFailures = defaultGlobalFailures
	}
	if options.IdleLimit <= 0 || options.AbsoluteLimit <= options.IdleLimit ||
		options.MaxSessions <= 0 || options.MaxSessions > defaultMaxSessions {
		return nil, fmt.Errorf("invalid browser session configuration")
	}
	limiter, err := newFailureLimiter(options.FailureWindow, options.PerIPFailures, options.GlobalFailures)
	if err != nil {
		return nil, err
	}
	return &Service{
		principal: principal, hash: bootstrap.hash,
		now: options.Now, random: options.Random,
		idleLimit: options.IdleLimit, absoluteLimit: options.AbsoluteLimit,
		maxSessions: options.MaxSessions, sessions: make(map[[sha256.Size]byte]storedSession),
		limiter: limiter, revocationSubscribers: make(map[uint64]chan SessionHandle),
	}, nil
}

func (s *Service) Principal() Principal { return clonePrincipal(s.principal) }

func (s *Service) Login(username string, password []byte, peerIP string) (Login, error) {
	if !usernamePattern.MatchString(username) || ValidatePassword(password) != nil || net.ParseIP(peerIP) == nil {
		return Login{}, ErrInvalidCredentials
	}
	now := s.now().UTC()
	ticket, err := s.limiter.begin(peerIP, now)
	if err != nil {
		return Login{}, err
	}
	failed := true
	defer func() { s.limiter.complete(ticket, failed) }()
	usernameDigest := sha256.Sum256([]byte(username))
	expectedUsernameDigest := sha256.Sum256([]byte(s.principal.Username))
	passwordMatches := s.hash.verify(password)
	usernameMatches := subtle.ConstantTimeCompare(usernameDigest[:], expectedUsernameDigest[:]) == 1
	if !passwordMatches || !usernameMatches {
		return Login{}, ErrInvalidCredentials
	}
	failed = false
	return s.createSession(now)
}

func (s *Service) Lookup(cookieValue string) (Session, error) {
	digest, err := cookieDigest(cookieValue)
	if err != nil {
		return Session{}, ErrInvalidSession
	}
	now := s.now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.sessions[digest]
	if !ok || sessionExpired(stored, now) {
		if ok {
			delete(s.sessions, digest)
			s.publishRevocationLocked(SessionHandle{digest: digest})
		}
		return Session{}, ErrInvalidSession
	}
	return s.session(digest, stored), nil
}

// Check validates a previously authenticated handle without extending its idle
// deadline. Long-lived observational connections use it so their own liveness
// does not keep a browser session alive indefinitely.
func (s *Service) Check(handle SessionHandle) (Session, error) {
	now := s.now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.sessions[handle.digest]
	if !ok || sessionExpired(stored, now) {
		if ok {
			delete(s.sessions, handle.digest)
			s.publishRevocationLocked(handle)
		}
		return Session{}, ErrInvalidSession
	}
	return s.session(handle.digest, stored), nil
}

func (s *Service) Accept(handle SessionHandle) (Session, error) {
	now := s.now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.sessions[handle.digest]
	if !ok || sessionExpired(stored, now) {
		if ok {
			delete(s.sessions, handle.digest)
			s.publishRevocationLocked(handle)
		}
		return Session{}, ErrInvalidSession
	}
	stored.idleExpiresAt = now.Add(s.idleLimit)
	if stored.idleExpiresAt.After(stored.absoluteExpiresAt) {
		stored.idleExpiresAt = stored.absoluteExpiresAt
	}
	s.sessions[handle.digest] = stored
	return s.session(handle.digest, stored), nil
}

func (s *Service) ValidateCSRF(session Session, candidate string) error {
	if len(candidate) != base64.RawURLEncoding.EncodedLen(CSRFTokenBytes) {
		return ErrInvalidCSRF
	}
	candidateDigest := sha256.Sum256([]byte(candidate))
	if subtle.ConstantTimeCompare(candidateDigest[:], session.csrfDigest[:]) != 1 {
		return ErrInvalidCSRF
	}
	return nil
}

func (s *Service) Destroy(handle SessionHandle) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.sessions[handle.digest]; !ok {
		return
	}
	delete(s.sessions, handle.digest)
	s.publishRevocationLocked(handle)
}

// SubscribeRevocations receives coalescing best-effort revocation edges.
// Consumers must also call Check periodically because delivery is deliberately
// non-blocking and process-local.
func (s *Service) SubscribeRevocations() (<-chan SessionHandle, func()) {
	s.mu.Lock()
	s.nextRevocationSubscriber++
	id := s.nextRevocationSubscriber
	updates := make(chan SessionHandle, defaultMaxSessions*2)
	s.revocationSubscribers[id] = updates
	s.mu.Unlock()
	var once sync.Once
	cancel := func() {
		once.Do(func() {
			s.mu.Lock()
			delete(s.revocationSubscribers, id)
			close(updates)
			s.mu.Unlock()
		})
	}
	return updates, cancel
}

func (s *Service) createSession(now time.Time) (Login, error) {
	cookieBytes := make([]byte, SessionTokenBytes)
	if _, err := io.ReadFull(s.random, cookieBytes); err != nil {
		return Login{}, fmt.Errorf("generate browser session: %w", err)
	}
	defer wipe(cookieBytes)
	csrfBytes := make([]byte, CSRFTokenBytes)
	if _, err := io.ReadFull(s.random, csrfBytes); err != nil {
		return Login{}, fmt.Errorf("generate CSRF token: %w", err)
	}
	defer wipe(csrfBytes)
	if subtle.ConstantTimeCompare(cookieBytes, csrfBytes) == 1 {
		return Login{}, fmt.Errorf("generate distinct browser session secrets")
	}
	cookieValue := base64.RawURLEncoding.EncodeToString(cookieBytes)
	csrfToken := base64.RawURLEncoding.EncodeToString(csrfBytes)
	digest := sha256.Sum256([]byte(cookieValue))
	abs := now.Add(s.absoluteLimit)
	stored := storedSession{
		csrfToken: csrfToken, csrfDigest: sha256.Sum256([]byte(csrfToken)),
		createdAt: now, idleExpiresAt: now.Add(s.idleLimit), absoluteExpiresAt: abs,
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.purgeExpired(now)
	if _, collision := s.sessions[digest]; collision {
		return Login{}, fmt.Errorf("generate unique browser session")
	}
	s.nextSequence++
	stored.sequence = s.nextSequence
	if len(s.sessions) >= s.maxSessions {
		s.revokeOldest()
	}
	s.sessions[digest] = stored
	return Login{CookieValue: cookieValue, Session: s.session(digest, stored)}, nil
}

func (s *Service) session(digest [sha256.Size]byte, stored storedSession) Session {
	return Session{
		Handle: SessionHandle{digest: digest}, Principal: clonePrincipal(s.principal),
		CSRFToken: stored.csrfToken, IdleExpiresAt: stored.idleExpiresAt,
		AbsoluteExpiresAt: stored.absoluteExpiresAt, csrfDigest: stored.csrfDigest,
	}
}

func (s *Service) purgeExpired(now time.Time) {
	for digest, stored := range s.sessions {
		if sessionExpired(stored, now) {
			delete(s.sessions, digest)
			s.publishRevocationLocked(SessionHandle{digest: digest})
		}
	}
}

func (s *Service) revokeOldest() {
	var oldestDigest [sha256.Size]byte
	var oldest storedSession
	found := false
	for digest, candidate := range s.sessions {
		if !found || candidate.createdAt.Before(oldest.createdAt) ||
			candidate.createdAt.Equal(oldest.createdAt) && candidate.sequence < oldest.sequence {
			oldestDigest, oldest, found = digest, candidate, true
		}
	}
	if found {
		delete(s.sessions, oldestDigest)
		s.publishRevocationLocked(SessionHandle{digest: oldestDigest})
	}
}

func (s *Service) publishRevocationLocked(handle SessionHandle) {
	for _, subscriber := range s.revocationSubscribers {
		select {
		case subscriber <- handle:
		default:
		}
	}
}

func sessionExpired(stored storedSession, now time.Time) bool {
	return !now.Before(stored.idleExpiresAt) || !now.Before(stored.absoluteExpiresAt)
}

func cookieDigest(value string) ([sha256.Size]byte, error) {
	if len(value) != base64.RawURLEncoding.EncodedLen(SessionTokenBytes) || strings.TrimSpace(value) != value {
		return [sha256.Size]byte{}, ErrInvalidSession
	}
	decoded, err := base64.RawURLEncoding.Strict().DecodeString(value)
	if err != nil || len(decoded) != SessionTokenBytes || base64.RawURLEncoding.EncodeToString(decoded) != value {
		wipe(decoded)
		return [sha256.Size]byte{}, ErrInvalidSession
	}
	wipe(decoded)
	return sha256.Sum256([]byte(value)), nil
}

func PeerIP(remoteAddress string) (string, error) {
	host, _, err := net.SplitHostPort(remoteAddress)
	if err != nil {
		return "", fmt.Errorf("invalid socket peer address")
	}
	address := net.ParseIP(host)
	if address == nil {
		return "", fmt.Errorf("invalid socket peer IP")
	}
	return address.String(), nil
}

func IsRateLimited(err error) (*RateLimitError, bool) {
	var limited *RateLimitError
	ok := errors.As(err, &limited)
	return limited, ok
}
