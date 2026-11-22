"""Construct current model source selectors for retained test materials."""

from pydantic import BaseModel


def artifact_sources(refs):
    if not isinstance(refs, list):
        return refs
    values = []
    for ref in refs:
        if isinstance(ref, BaseModel):
            ref = ref.model_dump(exclude_none=True)
            ref = {key: value for key, value in ref.items() if key != "revision"}
        values.append({"kind": "artifact", **ref} if isinstance(ref, dict) else ref)
    return values


def http_sources(request_id):
    return [] if request_id is None else [{"kind": "http", "request_id": request_id}]
