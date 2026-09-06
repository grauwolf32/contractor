package app

import (
	"fmt"
	"net"
	"strconv"
)

// Defer environment validation until flags have overridden it. Errors name the
// setting, never the supplied value, which may accidentally contain a secret.
type deferredBooleanFlag struct {
	value    string
	fallback bool
	explicit bool
}

func (f *deferredBooleanFlag) String() string   { return strconv.FormatBool(f.fallback) }
func (f *deferredBooleanFlag) IsBoolFlag() bool { return true }
func (f *deferredBooleanFlag) Set(value string) error {
	f.value, f.explicit = value, true
	return nil
}
func (f deferredBooleanFlag) parse(name string) (bool, error) {
	if f.value == "" && !f.explicit {
		return f.fallback, nil
	}
	switch f.value {
	case "true":
		return true, nil
	case "false":
		return false, nil
	default:
		return false, fmt.Errorf("%s must be true or false", name)
	}
}

func validPprofListen(address string) bool {
	host, port, err := net.SplitHostPort(address)
	if err != nil || !isLoopbackListenAddress(address) || net.ParseIP(host) == nil || port == "" {
		return false
	}
	for _, digit := range port {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	number, err := strconv.Atoi(port)
	return err == nil && number >= 1 && number <= 65535
}
