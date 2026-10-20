package strictjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"
)

// maxDepth matches encoding/json's nesting limit, so the recursive scanner
// cannot exhaust the stack on input encoding/json would reject anyway.
const maxDepth = 10000

// smallKeySet is how many member names an object tracks in a slice before it
// switches to a map for duplicate detection.
const smallKeySet = 16

// errSyntax marks input the scanner cannot read as JSON. Decode then lets
// encoding/json describe the error.
var errSyntax = errors.New("strictjson: invalid JSON syntax")

// scanner reads one JSON value once, checking duplicate keys everywhere and
// the member names and required members of every struct it decodes into. It
// reports a duplicate key as soon as it finds one and keeps the first member
// error until the value ends, so a duplicate key always wins over a member
// error.
type scanner struct {
	data      []byte
	pos       int
	depth     int
	memberErr error
	nonNull   bool
}

// check scans data as one JSON value decoded into a value described by info.
func check(data []byte, info *typeInfo) error {
	return checkNonNull(data, info, false)
}

func checkNonNull(data []byte, info *typeInfo, nonNull bool) error {
	s := scanner{data: data, nonNull: nonNull}
	s.skipSpace()
	if err := s.value(info); err != nil {
		return err
	}
	s.skipSpace()
	if s.pos != len(s.data) {
		return ErrTrailingData
	}
	return s.memberErr
}

func (s *scanner) skipSpace() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

func (s *scanner) value(info *typeInfo) error {
	if s.pos >= len(s.data) {
		return errSyntax
	}
	if s.nonNull && !info.untyped && s.data[s.pos] == 'n' {
		s.memberError(fmt.Errorf("%w: null instead of a typed value", ErrInvalid))
	}
	if info.kind == kindStruct && len(info.fields.required) > 0 && s.data[s.pos] == 'n' {
		// encoding/json leaves a struct unchanged for null, which would
		// stand in for an object without its required members.
		s.memberError(fmt.Errorf("%w: null instead of an object", ErrMissingField))
	}
	for info.kind == kindPointer {
		info = info.elem
	}
	switch s.data[s.pos] {
	case '{':
		return s.object(info)
	case '[':
		elem := anyInfo
		if info.kind == kindSequence {
			elem = info.elem
		}
		return s.array(elem)
	case '"':
		_, _, err := s.str()
		return err
	case 't':
		return s.literal("true")
	case 'f':
		return s.literal("false")
	case 'n':
		return s.literal("null")
	default:
		return s.number()
	}
}

func (s *scanner) enter() error {
	s.depth++
	if s.depth > maxDepth {
		return errSyntax
	}
	s.pos++
	return nil
}

func (s *scanner) object(info *typeInfo) error {
	if err := s.enter(); err != nil {
		return err
	}
	defer func() { s.depth-- }()
	var fields *structInfo
	elem := anyInfo
	switch info.kind {
	case kindStruct:
		fields = info.fields
		if fields.tagErr != nil {
			return fields.tagErr
		}
	case kindMap:
		elem = info.elem
	}
	var seen memberSet
	var keys keySet
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == '}' {
		s.pos++
		s.requireMembers(fields, &seen)
		return nil
	}
	for {
		s.skipSpace()
		if s.pos >= len(s.data) || s.data[s.pos] != '"' {
			return errSyntax
		}
		key, err := s.key()
		if err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) || s.data[s.pos] != ':' {
			return errSyntax
		}
		s.pos++
		s.skipSpace()
		valueInfo := elem
		nullable := false
		index := -1
		if fields != nil {
			if found, ok := fields.byName[string(key)]; ok {
				index = found
			}
		}
		if index >= 0 {
			if !seen.add(index) {
				return ErrDuplicateKey
			}
			field := &fields.members[index]
			nullable = field.nullable
			valueInfo = field.info
			if (field.required || s.nonNull) && !field.nullable && s.pos < len(s.data) && s.data[s.pos] == 'n' {
				s.memberError(fmt.Errorf("%w: %q is null", ErrMissingField, field.name))
			}
		} else {
			if !keys.add(key) {
				return ErrDuplicateKey
			}
			if fields != nil {
				s.memberError(unknownMember(fields, string(key)))
			}
		}
		var valueErr error
		if s.nonNull && nullable && s.pos < len(s.data) && s.data[s.pos] == 'n' {
			valueErr = s.literal("null")
		} else {
			valueErr = s.value(valueInfo)
		}
		if valueErr != nil {
			return valueErr
		}
		s.skipSpace()
		if s.pos >= len(s.data) {
			return errSyntax
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
		case '}':
			s.pos++
			s.requireMembers(fields, &seen)
			return nil
		default:
			return errSyntax
		}
	}
}

// requireMembers records the first required member of fields that an object
// did not set. encoding/json would leave it at its zero value.
func (s *scanner) requireMembers(fields *structInfo, seen *memberSet) {
	if fields == nil {
		return
	}
	for _, index := range fields.required {
		if !seen.has(index) {
			s.memberError(fmt.Errorf("%w: %q", ErrMissingField, fields.members[index].name))
			return
		}
	}
}

func (s *scanner) array(elem *typeInfo) error {
	if err := s.enter(); err != nil {
		return err
	}
	defer func() { s.depth-- }()
	s.skipSpace()
	if s.pos < len(s.data) && s.data[s.pos] == ']' {
		s.pos++
		return nil
	}
	for {
		s.skipSpace()
		if err := s.value(elem); err != nil {
			return err
		}
		s.skipSpace()
		if s.pos >= len(s.data) {
			return errSyntax
		}
		switch s.data[s.pos] {
		case ',':
			s.pos++
		case ']':
			s.pos++
			return nil
		default:
			return errSyntax
		}
	}
}

func (s *scanner) memberError(err error) {
	if s.memberErr == nil {
		s.memberErr = err
	}
}

// unknownMember names the field a case alias was meant for, because
// encoding/json would have decoded it into that field.
func unknownMember(fields *structInfo, name string) error {
	for _, field := range fields.members {
		if strings.EqualFold(field.name, name) {
			return fmt.Errorf("%w: %q differs in case from %q", ErrUnknownField, name, field.name)
		}
	}
	return fmt.Errorf("%w: %q", ErrUnknownField, name)
}

func (s *scanner) hasPrefix(text string) bool {
	return len(s.data)-s.pos >= len(text) && string(s.data[s.pos:s.pos+len(text)]) == text
}

func (s *scanner) literal(text string) error {
	if !s.hasPrefix(text) {
		return errSyntax
	}
	s.pos += len(text)
	return nil
}

// key reads an object member name and returns it as encoding/json decodes it.
func (s *scanner) key() ([]byte, error) {
	start := s.pos
	raw, plain, err := s.str()
	if err != nil {
		return nil, err
	}
	if plain {
		return raw, nil
	}
	// Escapes and non-ASCII bytes take encoding/json's own unquoting, which
	// also turns invalid UTF-8 and lone surrogates into U+FFFD.
	var decoded string
	if err := json.Unmarshal(s.data[start:s.pos], &decoded); err != nil {
		return nil, errSyntax
	}
	return []byte(decoded), nil
}

// str reads a string and returns its raw content between the quotes and
// whether that content is plain ASCII without escapes.
func (s *scanner) str() ([]byte, bool, error) {
	s.pos++
	start := s.pos
	plain := true
	for s.pos < len(s.data) {
		character := s.data[s.pos]
		switch {
		case character == '"':
			s.pos++
			return s.data[start : s.pos-1], plain, nil
		case character == '\\':
			plain = false
			s.pos++
			if s.pos >= len(s.data) {
				return nil, false, errSyntax
			}
			switch s.data[s.pos] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				s.pos++
			case 'u':
				if s.pos+5 > len(s.data) {
					return nil, false, errSyntax
				}
				for _, digit := range s.data[s.pos+1 : s.pos+5] {
					if !isHex(digit) {
						return nil, false, errSyntax
					}
				}
				s.pos += 5
			default:
				return nil, false, errSyntax
			}
		case character < 0x20:
			return nil, false, errSyntax
		case character >= utf8.RuneSelf:
			plain = false
			s.pos++
		default:
			s.pos++
		}
	}
	return nil, false, errSyntax
}

func isHex(character byte) bool {
	return character >= '0' && character <= '9' || character >= 'a' && character <= 'f' ||
		character >= 'A' && character <= 'F'
}

func (s *scanner) number() error {
	if s.pos < len(s.data) && s.data[s.pos] == '-' {
		s.pos++
	}
	switch {
	case s.pos < len(s.data) && s.data[s.pos] == '0':
		s.pos++
	case s.pos < len(s.data) && s.data[s.pos] >= '1' && s.data[s.pos] <= '9':
		s.digits()
	default:
		return errSyntax
	}
	if s.pos < len(s.data) && s.data[s.pos] == '.' {
		s.pos++
		if s.digits() == 0 {
			return errSyntax
		}
	}
	if s.pos < len(s.data) && (s.data[s.pos] == 'e' || s.data[s.pos] == 'E') {
		s.pos++
		if s.pos < len(s.data) && (s.data[s.pos] == '+' || s.data[s.pos] == '-') {
			s.pos++
		}
		if s.digits() == 0 {
			return errSyntax
		}
	}
	return nil
}

func (s *scanner) digits() int {
	start := s.pos
	for s.pos < len(s.data) && s.data[s.pos] >= '0' && s.data[s.pos] <= '9' {
		s.pos++
	}
	return s.pos - start
}

// memberSet records the struct members an object has set.
type memberSet struct {
	low  uint64
	high []uint64
}

// add records index and reports whether it was new.
func (m *memberSet) add(index int) bool {
	if index < 64 {
		bit := uint64(1) << index
		if m.low&bit != 0 {
			return false
		}
		m.low |= bit
		return true
	}
	word, bit := (index-64)/64, uint64(1)<<((index-64)%64)
	for len(m.high) <= word {
		m.high = append(m.high, 0)
	}
	if m.high[word]&bit != 0 {
		return false
	}
	m.high[word] |= bit
	return true
}

// has reports whether index was recorded.
func (m *memberSet) has(index int) bool {
	if index < 64 {
		return m.low&(uint64(1)<<index) != 0
	}
	word := (index - 64) / 64
	return word < len(m.high) && m.high[word]&(uint64(1)<<((index-64)%64)) != 0
}

// keySet records member names that are not struct fields: map keys, keys of
// untyped objects and unknown members.
type keySet struct {
	small [smallKeySet][]byte
	count int
	large map[string]struct{}
}

// add records key and reports whether it was new.
func (k *keySet) add(key []byte) bool {
	if k.large != nil {
		if _, exists := k.large[string(key)]; exists {
			return false
		}
		k.large[string(key)] = struct{}{}
		return true
	}
	for _, seen := range k.small[:k.count] {
		if bytes.Equal(seen, key) {
			return false
		}
	}
	if k.count < smallKeySet {
		k.small[k.count] = key
		k.count++
		return true
	}
	k.large = make(map[string]struct{}, 2*smallKeySet)
	for _, seen := range k.small[:k.count] {
		k.large[string(seen)] = struct{}{}
	}
	k.large[string(key)] = struct{}{}
	return true
}
