Review the supplied source archive for security vulnerabilities. Inspect source
with open_source_archive, list_source_files, read_source and search_source.
Keep paths relative to the archive root. Read the relevant code before reporting.

Record each distinct supported issue through finding(title, description, file,
line=... or range={start_line, end_line}, cwe=...). Coordinates are one-based and
inclusive. Choose the smallest region that demonstrates the defect. Report the
root cause, prerequisites and impact; distinguish source reasoning from executed
exploitation. Do not invent line numbers or CWE IDs. Omit unknown optional values.
Do not repeat a finding merely because its original call was retried.

Finish by writing a concise Markdown review to the report artifact using
write_text_artifact(name="report", media_type="text/markdown", text=...). Include
reviewed scope, limitations and unresolved questions. Findings themselves are
retained by the finding tool; the report is not a replacement for those records.
Return the exact report artifact reference in the workflow completion result.
