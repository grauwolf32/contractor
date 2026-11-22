"""Safe metadata for explicitly selected argument contracts; never payload values."""

MAX_ARGUMENT_PATH_BYTES = 128
MAX_NORMALIZATION_NOTICES = 16


def json_type(value):
    return {
        dict: "object",
        list: "array",
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }.get(type(value), "non_json")


def argument_details(field, *, expected_type=None, actual_type=None):
    # Paths and expected shapes are authored by the calling tool, not extracted
    # from supplied object keys or native exception text.
    details = {"field": field[:MAX_ARGUMENT_PATH_BYTES]}
    if expected_type is not None:
        details["expected_type"] = expected_type
    if actual_type is not None:
        details["actual_type"] = actual_type
    return details


class NormalizationNotices:
    def __init__(self):
        self.fields = []
        self.omitted = 0

    def record(self, field, normalized, to_type):
        if not normalized:
            return
        if len(self.fields) == MAX_NORMALIZATION_NOTICES:
            self.omitted += 1
        else:
            self.fields.append(
                {
                    "field": field[:MAX_ARGUMENT_PATH_BYTES],
                    "from_type": "string",
                    "to_type": to_type,
                }
            )

    def attach(self, result):
        if self.fields:
            result["normalization"] = {"fields": self.fields, "omitted": self.omitted}
        return result
