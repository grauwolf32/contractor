# Local evaluation configurations

These opt-in manifests are outside the default catalog. Publish a selected
ModelPolicy through the Configuration API and select its exact identity in
`executionConfig.workers.modelPolicy` for an individual Run.

`worker-qwen38-eval@1` assumes the local `worker-model` route serves
Qwen3.8-27B. It changes only temperature from `worker@2`; see V53-002 for the
experiment and its limitations. It is not a general replacement for worker@2.
