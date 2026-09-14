# Managed Gateway credentials

[Documentation index](../README.md) · [Local stack](../guides/local-stack.md)

This optional setup adds LiteLLM virtual-key management to a running local
stack. First provision LiteLLM and its master/salt files using the
[local proxy setup](../deployment.md#llm-gateway).
Set `CONTRACTOR_API_TOKEN` to the Server's bearer token. All commands run from
the repository root.

Managed Gateway credentials use a separate database-encryption key. Generate
it once as an owner-only file and pass only its absolute path to Server.
With the basic local Server already running, read its exact Gateway reference
and generate the admin binding before restarting Server with the new flags:

```shell
umask 077
mkdir -p .local/secrets
openssl rand -base64 32 > .local/secrets/credential-master-key
GATEWAY_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/configurations/llm-gateways/local-litellm/versions/1 | \
  jq -c '.ref | {gatewayId:.name,version,digest}')"
jq -n --argjson gateway "$GATEWAY_REF" \
  --arg key "$(pwd)/.local/secrets/litellm-master-key" \
  '{bindings:[{llmGateway:$gateway,adminKeyFile:$key}]}' \
  > .local/secrets/llm-gateway-admin-bindings.yaml
```

The generated JSON document is also valid YAML. Stop the basic Server, then
restart it in the same terminal with its database and bearer-token environment:

```shell
contractor server run \
  --config ./configs/server.local.yaml \
  --credential-master-key-file="$(pwd)/.local/secrets/credential-master-key" \
  --llm-gateway-admin-bindings-file="$(pwd)/.local/secrets/llm-gateway-admin-bindings.yaml"
```

There is deliberately no environment variable or command-line literal for the
key bytes. The file must contain RFC 4648 base64 for exactly 32 bytes, with at
most one trailing newline, and must not be a symlink or readable by group or
world. The flag is optional while no encrypted credential rows exist; once one
exists, a missing file or a key whose fingerprint differs from the stored rows
makes Server startup fail before accepting traffic.

The admin-binding document is strict YAML and contains no key bytes. Each
entry names the complete digest-bearing Gateway ref and an absolute owner-only
`sk-` key file. Server resolves the ref at startup and remembers its exact
management origin; a missing/wrong digest, insecure file, redirect, malformed
provider response, or unbound Gateway fails closed. Changing an immutable
LLMGatewayConfig therefore requires a new binding entry and Server restart.
The LiteLLM manager uses finite timeouts and bounded bodies and accepts only
the response shape verified by `make test-litellm-contract` against the pinned
image.

Once Server is running, create an active virtual key without ever sending its
token through the public API. This example derives both exact refs from the
safe configuration API:

```shell
export CONTRACTOR_API_TOKEN='replace-with-a-local-api-token'
GATEWAY_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/configurations/llm-gateways/local-litellm/versions/1 | \
  jq -c '.ref | {gatewayId:.name,version,digest}')"
WORKER_POLICY_REF="$(curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  http://127.0.0.1:8080/v1/configurations/model-policies/worker/versions/2 | \
  jq -c '.ref | {policyId:.name,version,digest}')"
jq -n --argjson gateway "$GATEWAY_REF" --argjson policy "$WORKER_POLICY_REF" \
  '{credentialId:"managed-worker",llmGateway:$gateway,
    label:"Local Worker",gatewayPolicy:{modelPolicies:[$policy],
    maxBudget:10,budgetDuration:"1d",rpmLimit:30,maxParallelRequests:2}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H 'Idempotency-Key: create-managed-worker-1' \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/operations/credentials | jq .
```

The response contains only ID, exact Gateway ref, label, effective policy and
creation time. LiteLLM's generated virtual key is encrypted immediately in
PostgreSQL. Select `managed-worker` in Workflow or Run `executionConfig`; keep
the development token environment variables unset when testing this path.
