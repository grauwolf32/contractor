* Write separate runner for agents [done]
* Add metrics plugin to collect tool errors, agent token usage etc. [done]
* Fix cli formatter to support all existing tools [done]
* Refactor task_runner - remove excessive variables and define transfer model [done]
* Refactor file tools [done]
* Refactor code_tools [done]
* Add integrational tests for task runner [done]
* Add evaluation tools [done]
* Add http tools [done]
* Add caido tools [done]
* Add playwright tools
* Add overlay fs for writting [done]
* Add python REPL environment in the sandbox
* Add inbox for memory tools [done]
* Add skills for memory tools [done]
* Add skills for tasks [done]
* Add tools to report vulnerabilities [done]
* Add tools for code analysis [done]
* Add router agent to fulfill user requests [done]
* Improve cost efficiency of trace agent
* Agent-facing diff tool: let the exploit agent diff a probe against a baseline
  to confirm the oracle (and pick stronger proof for the HTTP chain). Caido has
  no native diff (introspected: 73 query + 139 mutation fields, none diff/compare),
  so build our own. Options:
    - http_diff(baseline_id, probe_id) over http_tools stored data — unified diff
      of the two RESPONSES (status + body; headers not persisted per-request) +
      summary (status_changed, length delta, lines +/-). Always available, matches
      the send-baseline/send-probe workflow. [recommended]
    - caido_diff(id_a, id_b) over Caido history — full RAW request+response diff
      (all headers, exact bytes); Caido-only, agent must hold two history ids.
    Pairs with the cited-baseline + anomaly exchanges in collect_http_chain.
