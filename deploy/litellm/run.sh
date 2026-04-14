podman run --rm -d  \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e LITELLM_MASTER_KEY="sk-litellm-changeme" \
  -e LITELLM_SALT_KEY="sk-random-hash-changeme" \
  --network="host" \
  "ghcr.io/berriai/litellm:main-stable" \
  --config /app/config.yaml
