-- Add the typed encrypted Interactsh credential; existing ciphertext and receipts remain intact.
ALTER TABLE runtime_credentials DROP CONSTRAINT runtime_credentials_credential_kind_check;
ALTER TABLE runtime_credentials ADD CONSTRAINT runtime_credentials_credential_kind_check
CHECK (credential_kind IN ('otlp-headers@1', 'http-proxy-basic@1', 'http-proxy-bearer@1',
                          'caido-bearer@1', 'interactsh-token@1',
                          'http-origin-basic@1', 'http-origin-bearer@1'));
