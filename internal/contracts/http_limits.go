package contracts

// Common normalized HTTP evidence byte ceilings shared with Runtime transport.
// Provider envelopes and private Control Plane transport have separate budgets.
const (
	MaxHTTPRequestBodyBytes  = 1 << 20
	MaxHTTPResponseBodyBytes = 16 << 20
	MaxHTTPHeaderBlockBytes  = 64 << 10
)
