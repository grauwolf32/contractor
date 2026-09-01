ALTER TABLE runtime_credentials
    DROP CONSTRAINT runtime_credentials_credential_kind_check;

ALTER TABLE runtime_credentials
    ADD CONSTRAINT runtime_credentials_credential_kind_check
    CHECK (credential_kind IN (
        'otlp-headers@1',
        'http-proxy-basic@1',
        'http-proxy-bearer@1',
        'caido-bearer@1'
    ));

ALTER TABLE runtime_credential_creations
    DROP CONSTRAINT runtime_credential_creations_credential_kind_check;

ALTER TABLE runtime_credential_creations
    ADD CONSTRAINT runtime_credential_creations_credential_kind_check
    CHECK (credential_kind IN (
        'otlp-headers@1',
        'http-proxy-basic@1',
        'http-proxy-bearer@1',
        'caido-bearer@1'
    ));
