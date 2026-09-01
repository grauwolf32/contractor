The exact source ZIP and optional cumulative overlay state are already hydrated
into one allocation-private workspace. Read `source.txt`, replace its exact
`before` line with `after`, and inspect the checkpoint diff. Never request or
return a host path. Return a strict successful Stage result with no artifacts;
the Runtime owns and injects the reserved `workspace_state` and `workspace_diff`
slots after their durable writes complete.
