// Package core holds one declaration per case of the testonlyexports rules;
// analysis_test.go lists the ones it must report.
package core

import "errors"

// Used has a production caller in cmd/app.
func Used() {}

// TestOnly is called only by a test of this package.
func TestOnly() {}

// Unused is called by nothing.
func Unused() {}

// TaggedOnly is called only by a test behind the fixture build tag.
func TaggedOnly() {}

// XTestOnly is called only by the external test package core_test.
func XTestOnly() {}

// UsedByHelper is called only by the test helper package coretest.
func UsedByHelper() {}

// UsedByProcessTests is called only by a non-test file under tests/.
func UsedByProcessTests() {}

// UsedByGenerated is called only by generated production code.
func UsedByGenerated() {}

// Recursive calls itself; tests are its only other callers.
func Recursive(n int) int {
	if n == 0 {
		return 0
	}
	return Recursive(n - 1)
}

// ErrOnlyTests is compared only by tests.
var ErrOnlyTests = errors.New("only tests compare it")

// Store is used through NewStore and Get.
type Store struct{ name string }

// NewStore has a production caller.
func NewStore() *Store { return &Store{name: "store"} }

// Get has a production caller.
func (s *Store) Get() string { return s.name }

// TestHook is called only by tests.
func (s *Store) TestHook() string { return s.name }

// Hidden is named only by its own method and by tests.
type Hidden struct{}

// Self returns a Hidden of its own.
func (Hidden) Self() Hidden { return Hidden{} }

// Reader is used in production through Read; only tests call Close.
type Reader interface {
	Read() string
	Close()
}

type fileReader struct{}

func (fileReader) Read() string { return "file" }
func (fileReader) Close()       {}

// NewReader has a production caller.
func NewReader() Reader { return fileReader{} }

// Named is printed with fmt, which finds its String method.
type Named struct{}

// String implements fmt.Stringer and has no direct caller.
func (Named) String() string { return "named" }

// Pinger is used in production through both of its methods.
type Pinger interface {
	Ping()
	Name() string
}

// Part implements only Ping; Whole embeds it to implement Pinger.
type Part struct{}

// Ping has no direct caller; Whole promotes it to satisfy Pinger.
func (Part) Ping() {}

// Whole implements Pinger with the Ping it embeds.
type Whole struct{ Part }

// Name implements Pinger.
func (Whole) Name() string { return "whole" }

// Code is an enumeration of which production code uses one member.
type Code int

const (
	// CodeA has a production reader.
	CodeA Code = iota
	// CodeB is compared only by tests.
	CodeB
)

// Asserted is named only by the compile-time check below and by tests.
type Asserted interface{ Do() }

var _ Asserted = Impl{}

// Impl is used in production; nothing calls Do.
type Impl struct{}

// Do implements Asserted, which only tests use.
func (Impl) Do() {}

// Box is generic; production calls Value through an instance.
type Box[T any] struct{ value T }

// NewBox has a production caller.
func NewBox[T any](value T) Box[T] { return Box[T]{value: value} }

// Value has a production caller through Box[int].
func (b Box[T]) Value() T { return b.value }

// Peek is called only by tests.
func (b Box[T]) Peek() T { return b.value }
