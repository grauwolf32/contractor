CREATE TABLE runtime_config_versions (
    name text NOT NULL CHECK (name ~ '^[a-z][a-z0-9_-]*$' AND length(name) <= 63),
    version text NOT NULL CHECK (version ~ '^[A-Za-z0-9][A-Za-z0-9._+-]*$' AND length(version) <= 128),
    digest text NOT NULL CHECK (digest ~ '^sha256:[0-9a-f]{64}$'),
    canonical_document text NOT NULL
        CHECK (octet_length(canonical_document) <= 131072 AND jsonb_typeof(canonical_document::jsonb) = 'object'),
    built_in boolean NOT NULL DEFAULT false,
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND length(actor_id) <= 256),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (name, version),
    UNIQUE (name, version, digest),
    CHECK (
        (name = 'contractor-empty' AND version = '1' AND built_in
         AND digest = 'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f'
         AND canonical_document = '{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}')
        OR
        (NOT built_in AND NOT (name = 'contractor-empty' AND version = '1'))
    )
);

CREATE INDEX runtime_config_versions_created_at_idx
    ON runtime_config_versions (created_at, name, version);

CREATE TABLE runtime_config_publications (
    idempotency_key_digest text PRIMARY KEY
        CHECK (idempotency_key_digest ~ '^sha256:[0-9a-f]{64}$'),
    request_digest text NOT NULL CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    config_name text NOT NULL,
    config_version text NOT NULL,
    config_digest text NOT NULL,
    actor_id text NOT NULL CHECK (btrim(actor_id) <> '' AND length(actor_id) <= 256),
    published_at timestamptz NOT NULL,
    UNIQUE (config_name, config_version),
    FOREIGN KEY (config_name, config_version, config_digest)
        REFERENCES runtime_config_versions (name, version, digest)
);

CREATE INDEX runtime_config_publications_published_at_idx
    ON runtime_config_publications (published_at, config_name, config_version);

CREATE TABLE runtime_label_bindings (
    label text PRIMARY KEY CHECK (label ~ '^[a-z][a-z0-9_-]*$' AND length(label) <= 63),
    config_name text NOT NULL,
    config_version text NOT NULL,
    config_digest text NOT NULL,
    revision numeric(20, 0) NOT NULL
        CHECK (revision >= 1 AND revision <= 18446744073709551615),
    created_by text NOT NULL CHECK (btrim(created_by) <> '' AND length(created_by) <= 256),
    created_at timestamptz NOT NULL,
    updated_by text NOT NULL CHECK (btrim(updated_by) <> '' AND length(updated_by) <= 256),
    updated_at timestamptz NOT NULL,
    FOREIGN KEY (config_name, config_version, config_digest)
        REFERENCES runtime_config_versions (name, version, digest)
);

CREATE INDEX runtime_label_bindings_config_idx
    ON runtime_label_bindings (config_name, config_version, config_digest, label);

CREATE OR REPLACE FUNCTION contractor_protect_runtime_config_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'immutable RuntimeConfig row cannot be changed'
        USING ERRCODE = '23514';
END;
$$;

CREATE TRIGGER runtime_config_versions_protect_immutable
BEFORE UPDATE OR DELETE ON runtime_config_versions
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();

CREATE TRIGGER runtime_config_publications_protect_immutable
BEFORE UPDATE OR DELETE ON runtime_config_publications
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_config_immutable();

CREATE OR REPLACE FUNCTION contractor_protect_runtime_label_binding()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.label = 'default' THEN
            RAISE EXCEPTION 'default RuntimeConfig binding cannot be deleted'
                USING ERRCODE = '23514';
        END IF;
        RETURN OLD;
    END IF;

    IF NEW.label IS DISTINCT FROM OLD.label
       OR NEW.created_by IS DISTINCT FROM OLD.created_by
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.revision IS DISTINCT FROM OLD.revision + 1
       OR ROW(NEW.config_name, NEW.config_version, NEW.config_digest)
          IS NOT DISTINCT FROM
          ROW(OLD.config_name, OLD.config_version, OLD.config_digest)
    THEN
        RAISE EXCEPTION 'RuntimeConfig binding update must be one semantic revision'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER runtime_label_bindings_protect_mutation
BEFORE UPDATE OR DELETE ON runtime_label_bindings
FOR EACH ROW EXECUTE FUNCTION contractor_protect_runtime_label_binding();

INSERT INTO runtime_config_versions (
    name, version, digest, canonical_document, built_in, actor_id, created_at
) VALUES (
    'contractor-empty',
    '1',
    'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f',
    '{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}',
    true,
    'contractor-bootstrap',
    TIMESTAMPTZ '1970-01-01 00:00:00+00'
)
ON CONFLICT (name, version) DO NOTHING;

INSERT INTO runtime_label_bindings (
    label, config_name, config_version, config_digest, revision,
    created_by, created_at, updated_by, updated_at
) VALUES (
    'default',
    'contractor-empty',
    '1',
    'sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f',
    1,
    'contractor-bootstrap',
    TIMESTAMPTZ '1970-01-01 00:00:00+00',
    'contractor-bootstrap',
    TIMESTAMPTZ '1970-01-01 00:00:00+00'
)
ON CONFLICT (label) DO NOTHING;
