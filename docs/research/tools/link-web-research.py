#!/usr/bin/env python3
"""Link research-memo directions and hypotheses to their web source pages."""
from pathlib import Path
import re

path = Path(__file__).resolve().parents[1] / "reports" / "research.html"
web_dir = path.parent / "web"
html = path.read_text(encoding="utf-8")

# Remove links left by older versions of this script when their target page does
# not exist. This also makes the script safe when a report direction is added
# before its supporting web-research page.
linked_dirhead = re.compile(
    r'<div class="dir-head"><a class="dir-num" href="web/([A-Z]{1,2})\.html"[^>]*>\1</a>'
    r'(<h2[^>]*>)<a href="web/\1\.html"[^>]*>(.*?)</a>(</h2>)</div>'
)
n_unlinked_dir = 0


def unlink_missing_dir(m: re.Match[str]) -> str:
    global n_unlinked_dir
    did, h2open, title, h2close = m.group(1), m.group(2), m.group(3), m.group(4)
    if (web_dir / f"{did}.html").is_file():
        return m.group(0)
    n_unlinked_dir += 1
    return (
        f'<div class="dir-head"><span class="dir-num">{did}</span>'
        f"{h2open}{title}{h2close}</div>"
    )


html = linked_dirhead.sub(unlink_missing_dir, html)

linked_hid = re.compile(
    r'<a class="hid" href="web/([A-Z]{1,2})\.html#[^"]+"[^>]*>'
    r'([A-Z]{1,2}\d+)</a>'
)
n_unlinked_hid = 0


def unlink_missing_hid(m: re.Match[str]) -> str:
    global n_unlinked_hid
    did, hypothesis_id = m.group(1), m.group(2)
    if (web_dir / f"{did}.html").is_file():
        return m.group(0)
    n_unlinked_hid += 1
    return f'<span class="hid">{hypothesis_id}</span>'


html = linked_hid.sub(unlink_missing_hid, html)

# Link a direction only when its web/<ID>.html source page exists.
dirhead = re.compile(
    r'<div class="dir-head"><span class="dir-num">([A-Z]{1,2})</span>'
    r'(<h2[^>]*>)(.*?)(</h2>)</div>'
)
n_dir = 0


def repl_dir(m):
    global n_dir
    did, h2open, title, h2close = m.group(1), m.group(2), m.group(3), m.group(4)
    if not (web_dir / f"{did}.html").is_file():
        return m.group(0)
    n_dir += 1
    return (f'<div class="dir-head">'
            f'<a class="dir-num" href="web/{did}.html" target="_blank" title="open web research for {did}">{did}</a>'
            f'{h2open}<a href="web/{did}.html" target="_blank" '
            f'style="color:inherit;border-bottom:1px dotted #58a6ff66">{title}</a>{h2close}</div>')
html = dirhead.sub(repl_dir, html)

# Link hypothesis badges under the same existence rule.
hid = re.compile(r'<span class="hid">([A-Z]{1,2})(\d+)</span>')
n_hid = 0


def repl_hid(m):
    global n_hid
    dirpart, hyp = m.group(1), m.group(1) + m.group(2)
    if not (web_dir / f"{dirpart}.html").is_file():
        return m.group(0)
    n_hid += 1
    return (f'<a class="hid" href="web/{dirpart}.html#{hyp}" target="_blank" '
            f'onclick="event.stopPropagation()" title="open web research for {hyp}">{hyp}</a>')
html = hid.sub(repl_hid, html)

# Add a one-line note in the hero so the affordance is discoverable.
old_note = ('<div class="meta" style="color:#7ee787">Each direction title and hypothesis id is '
            'clickable → opens its web-research file (sources &amp; prior art) in <code>web/</code>.</div>')
note = ('<div class="meta" style="color:#7ee787">Direction titles and hypothesis ids are '
        'clickable when a supporting web-research page is available in <code>web/</code>.</div>')
html = html.replace(old_note, note)
if note not in html:
    html = html.replace('</header>', note + '\n</header>', 1)

path.write_text(html, encoding="utf-8")
print(f"direction links: {n_dir}")
print(f"hypothesis links: {n_hid}")
print(f"removed missing direction links: {n_unlinked_dir}")
print(f"removed missing hypothesis links: {n_unlinked_hid}")
print("remaining dir-num spans:", html.count('<span class="dir-num">'))
print("remaining hid spans:", html.count('<span class="hid">'))
