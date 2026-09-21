package auth

import (
	"errors"
	"fmt"
	"net"
	"net/netip"
	"sort"
	"strings"
)

const (
	maximumTrustedProxies = 64
	// One hop per trusted proxy plus the client; longer chains are not a
	// deployment this policy describes and are never walked further.
	maximumForwardedHops = maximumTrustedProxies + 1
)

var ErrInvalidTrustedProxy = errors.New("invalid trusted proxy")

// PeerPolicy resolves the client IP that login rate limiting attributes
// failures to. Without trusted proxies it is the socket peer. When the socket
// peer is a trusted proxy, the client is the nearest X-Forwarded-For hop that
// is not itself a trusted proxy, so a client can never choose its own limiter
// bucket by sending forwarded headers of its own.
type PeerPolicy struct {
	trusted []netip.Prefix
	values  []string
}

// NewPeerPolicy accepts exact IP addresses and CIDR prefixes. An empty list is
// the strict default that trusts no forwarded headers.
func NewPeerPolicy(proxies []string) (PeerPolicy, error) {
	if len(proxies) > maximumTrustedProxies {
		return PeerPolicy{}, fmt.Errorf("%w: at most %d trusted proxies are supported", ErrInvalidTrustedProxy, maximumTrustedProxies)
	}
	result := PeerPolicy{}
	seen := make(map[netip.Prefix]struct{}, len(proxies))
	for _, source := range proxies {
		prefix, err := parseTrustedProxy(source)
		if err != nil {
			return PeerPolicy{}, err
		}
		if _, duplicate := seen[prefix]; duplicate {
			return PeerPolicy{}, fmt.Errorf("%w: duplicate trusted proxy %q", ErrInvalidTrustedProxy, source)
		}
		seen[prefix] = struct{}{}
		result.trusted = append(result.trusted, prefix)
		result.values = append(result.values, prefix.String())
	}
	sort.Strings(result.values)
	return result, nil
}

func parseTrustedProxy(source string) (netip.Prefix, error) {
	if source == "" || len(source) > 64 || strings.TrimSpace(source) != source {
		return netip.Prefix{}, fmt.Errorf("%w: %q", ErrInvalidTrustedProxy, source)
	}
	var prefix netip.Prefix
	if strings.Contains(source, "/") {
		parsed, err := netip.ParsePrefix(source)
		if err != nil {
			return netip.Prefix{}, fmt.Errorf("%w: %q is not an IP address or CIDR prefix", ErrInvalidTrustedProxy, source)
		}
		prefix = parsed
	} else {
		address, err := netip.ParseAddr(source)
		if err != nil {
			return netip.Prefix{}, fmt.Errorf("%w: %q is not an IP address or CIDR prefix", ErrInvalidTrustedProxy, source)
		}
		if address.Zone() != "" {
			return netip.Prefix{}, fmt.Errorf("%w: %q must not carry an interface zone", ErrInvalidTrustedProxy, source)
		}
		prefix = netip.PrefixFrom(address, address.BitLen())
	}
	if prefix.Addr().Is4In6() {
		return netip.Prefix{}, fmt.Errorf("%w: %q must be a plain IPv4 or IPv6 prefix, not IPv4-mapped", ErrInvalidTrustedProxy, source)
	}
	if prefix.Bits() == 0 {
		return netip.Prefix{}, fmt.Errorf("%w: %q would trust forwarded headers from every peer", ErrInvalidTrustedProxy, source)
	}
	return prefix.Masked(), nil
}

// Values returns the canonical trusted prefixes for logging and diagnostics.
func (p PeerPolicy) Values() []string { return append([]string(nil), p.values...) }

func (p PeerPolicy) trusts(address netip.Addr) bool {
	for _, prefix := range p.trusted {
		if prefix.Contains(address) {
			return true
		}
	}
	return false
}

// ClientIP attributes a request to one IP for rate limiting. forwardedFor
// carries every X-Forwarded-For header value in wire order. A socket peer that
// is not a trusted proxy is always the client, whatever it forwards. Behind a
// trusted proxy the chain is walked from the proxy-appended end toward the
// client, and the first hop that is not a trusted proxy is the client. A chain
// the proxy did not populate, or one it appended malformed text to, degrades
// to the proxy address rather than failing login.
func (p PeerPolicy) ClientIP(remoteAddress string, forwardedFor []string) (string, error) {
	peer, err := PeerIP(remoteAddress)
	if err != nil {
		return "", err
	}
	if len(p.trusted) == 0 {
		return peer, nil
	}
	peerAddress, err := netip.ParseAddr(peer)
	if err != nil || !p.trusts(peerAddress) {
		return peer, nil
	}
	hops := forwardedHops(forwardedFor)
	if hops == nil {
		return peer, nil
	}
	for index := len(hops) - 1; index >= 0; index-- {
		if !p.trusts(hops[index]) {
			return hops[index].String(), nil
		}
	}
	// Every hop is a trusted proxy: the outermost proxy is the closest thing
	// to a client identity the deployment recorded.
	return hops[0].String(), nil
}

// forwardedHops parses the whole X-Forwarded-For chain. Any malformed hop makes
// the chain unusable, because a proxy that appends unparseable values cannot be
// relied on to have appended the real client either.
func forwardedHops(values []string) []netip.Addr {
	var hops []netip.Addr
	for _, value := range values {
		for _, raw := range strings.Split(value, ",") {
			entry := strings.TrimSpace(raw)
			if entry == "" {
				return nil
			}
			address, err := parseForwardedHop(entry)
			if err != nil {
				return nil
			}
			hops = append(hops, address)
			if len(hops) > maximumForwardedHops {
				return nil
			}
		}
	}
	if len(hops) == 0 {
		return nil
	}
	return hops
}

// parseForwardedHop accepts the plain address form and the bracketed or
// host:port forms some proxies emit; zones and IPv4-mapped IPv6 are normalized
// so one client cannot occupy several limiter buckets.
func parseForwardedHop(entry string) (netip.Addr, error) {
	candidate := entry
	if host, _, err := net.SplitHostPort(entry); err == nil {
		candidate = host
	} else if strings.HasPrefix(entry, "[") && strings.HasSuffix(entry, "]") {
		candidate = entry[1 : len(entry)-1]
	}
	address, err := netip.ParseAddr(candidate)
	if err != nil {
		return netip.Addr{}, err
	}
	return address.WithZone("").Unmap(), nil
}
