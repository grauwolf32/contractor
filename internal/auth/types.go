// Package auth implements the single-principal local authentication boundary.
// Browser session state is process-local and deliberately has no persistence
// dependency.
package auth

import (
	"context"
	"errors"
	"regexp"
)

var (
	ErrInvalidBootstrap   = errors.New("invalid local authentication bootstrap")
	ErrInvalidCredentials = errors.New("invalid local credentials")
	ErrRateLimited        = errors.New("local authentication rate limited")
	ErrInvalidSession     = errors.New("invalid browser session")
	ErrInvalidOrigin      = errors.New("invalid browser origin")
	ErrInvalidCSRF        = errors.New("invalid CSRF token")

	principalIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)
	usernamePattern    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`)
)

const (
	CapabilityUser       = "user"
	CapabilityOperations = "operations"
)

type Principal struct {
	UserID       string   `json:"userId"`
	Username     string   `json:"username"`
	Capabilities []string `json:"capabilities"`
}

func NewPrincipal(userID, username string) (Principal, error) {
	if !principalIDPattern.MatchString(userID) || !usernamePattern.MatchString(username) {
		return Principal{}, ErrInvalidBootstrap
	}
	return Principal{
		UserID: userID, Username: username,
		Capabilities: []string{CapabilityUser, CapabilityOperations},
	}, nil
}

func clonePrincipal(source Principal) Principal {
	result := source
	result.Capabilities = append([]string(nil), source.Capabilities...)
	return result
}

type principalContextKey struct{}

func WithPrincipal(ctx context.Context, principal Principal) context.Context {
	return context.WithValue(ctx, principalContextKey{}, clonePrincipal(principal))
}

func PrincipalFromContext(ctx context.Context) (Principal, bool) {
	principal, ok := ctx.Value(principalContextKey{}).(Principal)
	if !ok {
		return Principal{}, false
	}
	return clonePrincipal(principal), true
}
