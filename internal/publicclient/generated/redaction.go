package publicapi

import (
	"fmt"
	"io"
)

// The login and logout responses carry the session cookie in their Set-Cookie
// header, so their header types redact themselves as a whole for every fmt
// verb (SEC-R88). The methods live beside the generated code, which
// make generate-public-client rewrites.

func (LoginResponse200Headers) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "publicapi.LoginResponse200Headers([REDACTED])")
}

func (LogoutResponse204Headers) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, "publicapi.LogoutResponse204Headers([REDACTED])")
}
