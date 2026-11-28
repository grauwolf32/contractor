"""Native wire bounds, pinned against Go by the shared grpc-limits fixture."""

MAX_CONTROL_BYTES = 1 << 20
MAX_A2A_BYTES = 4 << 20
MAX_STATE_MESSAGE_BYTES = (4 << 20) + 1024
MAX_METADATA_BYTES = 16 << 10
MAX_CONCURRENT_RPCS = 32

# grpcio's first setting is probabilistic above the soft threshold. Every
# native endpoint also applies the same explicit hard ceiling.
METADATA_OPTIONS = (
    ("grpc.max_metadata_size", MAX_METADATA_BYTES),
    ("grpc.absolute_max_metadata_size", MAX_METADATA_BYTES),
)

BULK_CHUNK_BYTES = 256 << 10
BULK_HEADER_BYTES = 1 << 20
BULK_FRAME_BYTES = BULK_HEADER_BYTES + 1024
BULK_LOOKAHEAD_BYTES = 64 << 10

# Shared private-listener/client lifetime policy (CTL-R165).
PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS = 60
IDLE_SECONDS = 30.0
LIFETIME_SECONDS = 300.0
