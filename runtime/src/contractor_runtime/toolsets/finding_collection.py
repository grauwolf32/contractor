"""Bounded, self-contained findings ZIP codec shared with auditdomain's v1 contract."""

from __future__ import annotations

import calendar
import hashlib
import io
import json
import re
import stat
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any

import jcs

from contractor_runtime.contracts import ArtifactRef

COLLECTION_MEDIA_TYPE = "application/vnd.contractor.findings-collection+zip"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")
MEDIA_TYPE = re.compile(r"[a-z0-9!#$%&'+.^_`|~-]+/[a-z0-9!#$%&'+.^_`|~-]+")


class FindingsError(ValueError):
    """A bounded error code safe to report without logging collection contents."""

    retryable = False

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require(condition: Any, code: str = "findings_collection_invalid") -> None:
    if not condition:
        raise FindingsError(code)


def identifier(value: Any) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return jcs.canonicalize(value)


def object_fields(value: Any, required: str, optional: str = "") -> None:
    keys, extra = set(required.split()), set(optional.split())
    require(isinstance(value, dict) and keys <= value.keys() <= keys | extra)


def array(value: Any, maximum: int, minimum: int = 0) -> list[Any]:
    require(isinstance(value, list))
    require(minimum <= len(value) <= maximum, "findings_limit_exceeded")
    return value


def strict_json(data: bytes, *, canonical_required: bool = True) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
        pending = [(value, 1)]
        nodes = 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            require(depth <= 64 and nodes <= 250000, "findings_limit_exceeded")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
                pending.extend((key, depth + 1) for key in item)
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
            elif isinstance(item, str):
                require("\0" not in item)
                require(len(item.encode("utf-8")) <= 65536, "findings_limit_exceeded")
        if canonical_required:
            require(canonical(value) == data)
        return value
    except FindingsError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise FindingsError("findings_collection_invalid") from error


def document_id(document: dict[str, Any]) -> str:
    scope, ref = document["scope"], document["ref"]
    return (
        "doc-"
        + digest(
            canonical(
                [
                    scope["kind"],
                    scope["id"],
                    ref["namespace"],
                    ref["name"],
                    ref["revision"],
                    document["digest"],
                    document["media_type"],
                    document["size_bytes"],
                ]
            )
        )[7:]
    )


@dataclass(frozen=True)
class FindingCollection:
    metadata: dict[str, Any]
    contents: dict[str, bytes]
    proposals: dict[str, dict[str, Any]]
    digest: str

    @property
    def namespace(self) -> str:
        return "findings-" + self.digest[7:]


def decode_collection(payload: bytes) -> FindingCollection:
    require(0 < len(payload) <= MAX_ARCHIVE_BYTES, "findings_limit_exceeded")
    try:
        raw = _archive(payload)
        manifest = strict_json(raw["manifest.json"])
        object_fields(manifest, "schema package_id kind members")
        require(manifest["schema"] == "contractor.audit.package.v1")
        require(manifest["kind"] == "finding-collection")
        members = array(manifest["members"], 1024, 1)
        require(len(raw) == len(members) + 1)
        by_id: dict[str, Any] = {}
        previous = ""
        for member in members:
            object_fields(member, "id path media_type size digest")
            require(identifier(member["id"]) and member["id"] not in by_id)
            path = member["path"]
            require(isinstance(path, str) and path > previous and path != "manifest.json")
            body = raw[path]
            require(type(member["size"]) is int and member["size"] == len(body))
            require(member["digest"] == digest(body), "findings_digest_mismatch")
            previous = path
            by_id[member["id"]] = member
        collection_member = by_id["collection"]
        require(collection_member["path"] == "collection.json")
        require(collection_member["media_type"] == "application/json")
        metadata_bytes = raw["collection.json"]
        require(len(metadata_bytes) <= 1024 * 1024, "findings_limit_exceeded")
        metadata = strict_json(metadata_bytes)
        require(manifest["package_id"] == "collection-" + digest(metadata_bytes)[7:])
        _validate_metadata(metadata)
        require(len(members) == len(metadata["documents"]) + 1)
        contents: dict[str, bytes] = {}
        for document in metadata["documents"]:
            member = by_id[document["id"]]
            require(
                member
                == {
                    "id": document["id"],
                    "path": "documents/" + document["id"],
                    "media_type": document["media_type"],
                    "size": document["size_bytes"],
                    "digest": document["digest"],
                },
                "findings_digest_mismatch",
            )
            contents[document["id"]] = raw[member["path"]]
        proposals: dict[str, dict[str, Any]] = {}
        for entry in metadata["entries"]:
            proposal_id = entry["proposal_document_id"]
            if proposal_id not in proposals:
                proposals[proposal_id] = _proposal(contents[proposal_id])
            require(
                sorted(proposals[proposal_id]["evidence_ids"])
                == [link["evidence_id"] for link in entry["evidence"]],
                "findings_reference_invalid",
            )
        return FindingCollection(metadata, contents, proposals, digest(payload))
    except FindingsError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
        raise FindingsError("findings_collection_invalid") from error


def _archive(payload: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            require(0 < len(infos) <= 1025, "findings_limit_exceeded")
            raw: dict[str, bytes] = {}
            total = 0
            for info in infos:
                # Collection paths are fixed by the contract, so no extraction or
                # general-purpose filesystem path normalization is necessary.
                name = info.filename
                require(name == info.orig_filename and name not in raw)
                require(
                    name in {"manifest.json", "collection.json"}
                    or re.fullmatch(r"documents/doc-[0-9a-f]{64}", name) is not None
                )
                mode = info.external_attr >> 16
                require(not info.is_dir() and not info.external_attr & 0x10)
                require(stat.S_IFMT(mode) in {0, stat.S_IFREG} and not mode & 0o111)
                require(
                    not info.flag_bits & 1
                    and info.compress_type
                    in {
                        zipfile.ZIP_STORED,
                        zipfile.ZIP_DEFLATED,
                    }
                )
                limit = 512 * 1024 if name == "manifest.json" else MAX_ARCHIVE_BYTES
                require(info.file_size <= limit, "findings_limit_exceeded")
                total += info.file_size
                require(total <= 32 * 1024 * 1024, "findings_limit_exceeded")
                with archive.open(info) as stream:
                    body = stream.read(limit + 1)
                require(len(body) == info.file_size and len(body) <= limit)
                raw[name] = body
            return raw
    except FindingsError:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, EOFError, zlib.error) as error:
        raise FindingsError("findings_collection_invalid") from error


def _timestamp(value: Any) -> None:
    require(isinstance(value, str))
    match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.([0-9]{0,8}[1-9]))?Z",
        value,
        flags=re.ASCII,
    )
    require(match)
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    require(1 <= month <= 12 and 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59)
    require(1 <= day <= calendar.monthrange(year, month)[1])


def _validate_metadata(value: dict[str, Any]) -> None:
    object_fields(value, "schema snapshot_at sources documents entries")
    require(value["schema"] == "contractor.findings.collection.v1")
    _timestamp(value["snapshot_at"])
    source_keys = []
    for source in array(value["sources"], 64, 1):
        object_fields(source, "kind id")
        require(source["kind"] in {"run", "audit"} and identifier(source["id"]))
        source_keys.append((source["kind"], source["id"]))
    require(source_keys == sorted(set(source_keys)))
    sources = set(source_keys)
    documents: dict[str, Any] = {}
    total = 0
    previous = ""
    for document in array(value["documents"], 1023):
        object_fields(document, "id scope ref digest media_type size_bytes")
        scope = document["scope"]
        object_fields(scope, "kind id")
        require(scope["kind"] in {"run", "project", "user"} and identifier(scope["id"]))
        object_fields(document["ref"], "namespace name revision")
        ArtifactRef.model_validate(document["ref"]).require_exact()
        require(
            isinstance(document["digest"], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", document["digest"])
        )
        media_type = document["media_type"]
        require(
            isinstance(media_type, str)
            and len(media_type) <= 127
            and MEDIA_TYPE.fullmatch(media_type)
        )
        size = document["size_bytes"]
        require(type(size) is int and 0 <= size <= MAX_ARCHIVE_BYTES)
        total += size
        require(total <= MAX_ARCHIVE_BYTES, "findings_limit_exceeded")
        require(
            document["id"] == document_id(document) and document["id"] > previous,
            "findings_reference_invalid",
        )
        documents[document["id"]] = document
        previous = document["id"]
    used: set[str] = set()
    proposals: set[str] = set()
    previous = ""
    for entry in array(value["entries"], 256):
        object_fields(
            entry,
            "receipt_id proposal_id run_id invocation_id retention "
            "proposal_document_id evidence reviews",
            "audit_origin audit_holds",
        )
        require(
            all(
                identifier(entry[key])
                for key in (
                    "receipt_id",
                    "proposal_id",
                    "run_id",
                    "invocation_id",
                )
            )
        )
        require(entry["receipt_id"] > previous and entry["proposal_id"] not in proposals)
        previous = entry["receipt_id"]
        proposals.add(entry["proposal_id"])
        require(entry["retention"] in {"source-held", "audit-held", "discarded"})
        selected = ("run", entry["run_id"]) in sources
        if "audit_origin" in entry:
            origin = entry["audit_origin"]
            object_fields(origin, "audit_id execution_id role")
            require(all(identifier(part) for part in origin.values()))
            selected |= ("audit", origin["audit_id"]) in sources
        if "audit_holds" in entry:
            holds = array(entry["audit_holds"], 64, 1)
            require(all(identifier(hold) for hold in holds) and holds == sorted(set(holds)))
            selected |= any(("audit", hold) in sources for hold in holds)
        proposal = documents[entry["proposal_document_id"]]
        require(
            proposal["media_type"] == "application/json"
            and 0 < proposal["size_bytes"] <= 8 * 1024 * 1024
        )
        used.add(proposal["id"])
        evidence_ids = []
        for link in array(entry["evidence"], 256):
            object_fields(link, "evidence_id document_id")
            require(identifier(link["evidence_id"]) and link["document_id"] in documents)
            evidence_ids.append(link["evidence_id"])
            used.add(link["document_id"])
        require(evidence_ids == sorted(set(evidence_ids)))
        review_keys = []
        for review in array(entry["reviews"], 32):
            object_fields(
                review,
                "audit_id finding_id revision state",
                "decision_id assessment_id duplicate_target_id",
            )
            require(
                all(
                    identifier(part)
                    for key, part in review.items()
                    if key not in {"revision", "state"}
                )
            )
            require(type(review["revision"]) is int and 0 < review["revision"] < 2**53)
            require(
                review["state"]
                in {
                    "proposed",
                    "confirmed",
                    "rejected",
                    "duplicate",
                    "needs-evidence",
                }
            )
            review_keys.append((review["audit_id"], review["finding_id"]))
            selected |= ("audit", review["audit_id"]) in sources
        require(review_keys == sorted(set(review_keys)))
        require(selected, "findings_reference_invalid")
    require(used == documents.keys(), "findings_reference_invalid")


def _proposal(data: bytes) -> dict[str, Any]:
    value = strict_json(data, canonical_required=False)
    object_fields(
        value,
        "schema client_key title description subject preconditions evidence_ids limitations",
        "hypothesis standard_refs proposed_checks severity_suggestion",
    )
    require(value["schema"] == "contractor.audit.finding-proposal.v1")
    require(identifier(value["client_key"]))
    for field in ("title", "description"):
        require(isinstance(value[field], str) and value[field].strip())
    object_fields(value["subject"], "kind key")
    require(all(identifier(part) for part in value["subject"].values()))
    # Go's optional string fields accept null as their zero value.
    require(value.get("hypothesis") is None or isinstance(value["hypothesis"], str))
    require(
        value.get("severity_suggestion")
        in {
            None,
            "",
            "informational",
            "low",
            "medium",
            "high",
            "critical",
        }
    )
    for field in ("preconditions", "limitations"):
        require(all(isinstance(part, str) and part.strip() for part in array(value[field], 512)))
    evidence_ids = array(value["evidence_ids"], 256)
    require(all(identifier(part) for part in evidence_ids))
    require(len(evidence_ids) == len(set(evidence_ids)))
    references = value.get("standard_refs")
    for reference in array([] if references is None else references, 512):
        object_fields(reference, "scheme version requirement_id")
        require(identifier(reference["scheme"]) and identifier(reference["requirement_id"]))
        require(isinstance(reference["version"], str) and reference["version"].strip())
    checks = value.get("proposed_checks")
    for check in array([] if checks is None else checks, 512):
        object_fields(check, "objective method")
        require(isinstance(check["objective"], str) and check["objective"].strip())
        require(identifier(check["method"]))
    return value
