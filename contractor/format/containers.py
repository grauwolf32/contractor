def xml_tag(tag_name: str, value: str, delimeter: str = "") -> str:
    return f"<{tag_name}>{delimeter}{value}{delimeter}</{tag_name}>"


def list_tag(tag_name: str, value: str, delimeter: str = "*") -> str:
    return f"{delimeter} {tag_name.upper()}: \n{value}\n\n"


def note_tag(tag_name: str, value: str) -> str:
    return f"{tag_name}: {value}"
