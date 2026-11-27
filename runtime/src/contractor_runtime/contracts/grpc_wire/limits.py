"""Native wire bounds, pinned against Go by the shared grpc-limits fixture."""

MAX_CONTROL_BYTES = 1 << 20
MAX_A2A_BYTES = 4 << 20
MAX_METADATA_BYTES = 16 << 10
MAX_CONCURRENT_RPCS = 32
