# Local evaluation configurations

These opt-in manifests are outside the default catalog. Publish a selected
ModelPolicy through the Configuration API and select its exact identity in
`executionConfig.workers.modelPolicy` for an individual Run.

`worker-qwen38-eval@1` assumes the local `worker-model` route serves
Qwen3.8-27B. Compared with the current `worker@2`, it uses temperature 1.0
instead of 0.1 and limits each response to 16,384 instead of 32,768 output
tokens. Its other policy fields match. The 16,384-token limit preserves the
V53-002 experiment's historical setting; this is not a temperature-only
comparison with the current Worker policy or a general replacement for it.
