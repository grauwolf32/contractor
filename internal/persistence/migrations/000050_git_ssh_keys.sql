CREATE TABLE git_ssh_keys (
    owner_id text PRIMARY KEY CHECK (length(owner_id) BETWEEN 1 AND 256),
    generation text NOT NULL CHECK (generation ~ '^[0-9a-f]{32}$'),
    encryption_schema_version text NOT NULL CHECK (encryption_schema_version = 'git-ssh-key@1'),
    key_id text NOT NULL CHECK (key_id ~ '^sha256:[0-9a-f]{64}$'),
    nonce bytea NOT NULL CHECK (octet_length(nonce) = 12),
    ciphertext bytea NOT NULL CHECK (octet_length(ciphertext) BETWEEN 17 AND 32784),
    fingerprint text NOT NULL,
    key_type text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
