"""HTTP transport bounds from docs/spec/11-http-and-caido-tools.md.

These are the existing http-tools@1 limits. Finding evidence reuses them so
retention never silently changes the transport's payload or attempt contract.
"""

MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
MAX_PREVIEW_CHARACTERS = 8192
MAX_READ_UNITS = 8192
MAX_HEADERS = 64
MAX_QUERY_KEYS = 64
MAX_HEADER_VALUE_BYTES = 8192
MAX_HEADER_BYTES = 64 * 1024
MAX_QUERY_BYTES = 64 * 1024
MAX_URL_BYTES = 8192
MAX_HISTORY = 128
MAX_COOKIES = 128
MAX_REDIRECTS = 10
MAX_ATTEMPTS = 3
MAX_EXCHANGE_ATTEMPTS = MAX_REDIRECTS + MAX_ATTEMPTS
