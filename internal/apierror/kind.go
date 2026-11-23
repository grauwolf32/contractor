// Package apierror is the registry of the public API's error codes. A Kind is
// one registered code with the HTTP status that carries it and whether the
// same request may succeed when it is sent again. Domain errors report their
// Kind, so the public API answers from the error itself instead of a switch
// over every package's sentinels. Only this package defines a Kind, so every
// code the public API can answer is a registry entry; the registry table of
// docs/spec/platform/errors.md is generated from it.
package apierror

import (
	"errors"
	"fmt"
)

// Kind is one registered public error code. The zero Kind is not a code.
type Kind struct{ entry *entry }

// entry is the registry row of a Kind.
type entry struct {
	// code is the stable value of the public error envelope's code.
	code string
	// status is the HTTP status that carries the code, or 0 for a code that
	// only a WebSocket error frame carries.
	status int
	// retryable tells a client whether the same request may succeed later.
	retryable bool
	// frame marks a code that a WebSocket error frame can also carry.
	frame bool
	// message is the safe default public message.
	message string
	// meaning and owner document the code in the registry table: what it
	// reports, and Markdown links, relative to docs/spec/platform, to the
	// requirements that define it.
	meaning, owner string
}

// Code returns the stable public code.
func (k Kind) Code() string { return k.entry.code }

// Status returns the HTTP status of the code, or 0 for a code that only a
// WebSocket error frame carries.
func (k Kind) Status() int { return k.entry.status }

// Retryable reports whether the same request may succeed when sent again.
func (k Kind) Retryable() bool { return k.entry.retryable }

// Message returns the safe default public message of the code.
func (k Kind) Message() string { return k.entry.message }

// Lookup returns the Kind of a registered code.
func Lookup(code string) (Kind, bool) {
	kind, ok := byCode[code]
	return kind, ok
}

// Classified is an error that reports the Kind the public API answers it
// with.
type Classified interface {
	error
	ErrorKind() Kind
}

// PublicMessager is implemented by a Classified error whose public message
// is more specific than its Kind's message. The message must be safe to show
// to the caller: it may name the request's own values but never internal
// causes.
type PublicMessager interface {
	PublicMessage() string
}

// Of returns the first Classified error in err's chain, which decides the
// public answer to err.
func Of(err error) (Classified, bool) {
	var classified Classified
	if !errors.As(err, &classified) {
		return nil, false
	}
	return classified, true
}

// MessageOf returns the public message of a Classified error.
func MessageOf(classified Classified) string {
	if specific, ok := classified.(PublicMessager); ok {
		if message := specific.PublicMessage(); message != "" {
			return message
		}
	}
	return classified.ErrorKind().Message()
}

// sentinel is a comparable error value of one Kind.
type sentinel struct {
	kind    Kind
	text    string
	message string
	parent  error
}

func (e *sentinel) Error() string         { return e.text }
func (e *sentinel) ErrorKind() Kind       { return e.kind }
func (e *sentinel) PublicMessage() string { return e.message }
func (e *sentinel) Unwrap() error         { return e.parent }

// Option refines a sentinel error.
type Option func(*sentinel)

// WithMessage answers the sentinel with message instead of its Kind's
// default message, for a code that several sentinels share. The message is
// shown to the caller, so it never names an internal cause, and it fits the
// error envelope's 1 to 512 bytes.
func WithMessage(message string) Option {
	if message == "" || len(message) > 512 {
		panic("apierror: a public message holds 1 to 512 bytes")
	}
	return func(e *sentinel) { e.message = message }
}

// New returns a sentinel error of kind with the internal text text.
func New(kind Kind, text string, options ...Option) error {
	return newSentinel(&sentinel{kind: kind, text: text}, options)
}

// Narrow returns a sentinel error of kind for a narrower case of parent: it
// still matches parent with errors.Is, and its text extends parent's with
// detail, but it answers with its own kind and message.
func Narrow(parent error, kind Kind, detail string, options ...Option) error {
	return newSentinel(&sentinel{kind: kind, text: fmt.Sprintf("%s: %s", parent, detail), parent: parent}, options)
}

// newSentinel applies options to e. A sentinel of the zero Kind would answer
// without a code, so it is a programming error.
func newSentinel(e *sentinel, options []Option) error {
	if e.kind.entry == nil {
		panic("apierror: a sentinel needs a registered Kind")
	}
	for _, option := range options {
		option(e)
	}
	return e
}
