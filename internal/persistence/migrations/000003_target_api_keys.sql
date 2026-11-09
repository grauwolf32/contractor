-- API-key identities reuse the existing encrypted credential owner and holds.
ALTER TABLE runtime_credentials DROP CONSTRAINT runtime_credentials_credential_kind_check;
ALTER TABLE runtime_credentials ADD CONSTRAINT runtime_credentials_credential_kind_check
CHECK (credential_kind = ANY (ARRAY[
 'otlp-headers@1', 'http-proxy-basic@1', 'http-proxy-bearer@1', 'caido-bearer@1',
 'interactsh-token@1', 'http-origin-basic@1', 'http-origin-bearer@1',
 'target-basic@1', 'target-bearer@1', 'target-cookies@1', 'target-api-keys@1'
]));
