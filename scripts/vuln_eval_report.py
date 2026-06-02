"""Generate a self-contained HTML report from vuln-eval results.

Called by ``run_vuln_eval.py`` after scoring.  Can also be used standalone::

    poetry run python scripts/vuln_eval_report.py eval_runs/vuln_eval/eval_results.json
"""
from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Chart helpers (matplotlib -> base64 PNG)
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str, style: str = "") -> str:
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" style="{style}"/>'


def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    color: str = "#4a90d9",
    figsize: tuple = (6, 3),
    horizontal: bool = False,
    value_labels: bool = True,
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        bars = ax.barh(labels, values, color=color)
        ax.set_xlabel(xlabel)
        if value_labels:
            for bar, v in zip(bars, values, strict=False):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{v:g}", va="center", fontsize=8)
    else:
        bars = ax.bar(labels, values, color=color)
        ax.set_ylabel(ylabel)
        if value_labels:
            for bar, v in zip(bars, values, strict=False):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:g}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _grouped_bar(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    colors: list[str] | None = None,
    figsize: tuple = (6, 3.5),
    ylabel: str = "",
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    n = len(series)
    w = 0.8 / max(n, 1)
    palette = colors or ["#4a90d9", "#e8723a", "#50c878", "#9b59b6", "#f1c40f"]
    for idx, (name, vals) in enumerate(series.items()):
        offset = (idx - n / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=name, color=palette[idx % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _stacked_bar(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    colors: list[str] | None = None,
    figsize: tuple = (6, 3),
    ylabel: str = "",
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    palette = colors or ["#4a90d9", "#e8723a", "#50c878", "#9b59b6"]
    for idx, (name, vals) in enumerate(series.items()):
        ax.bar(x, vals, bottom=bottom, label=name, color=palette[idx % len(palette)])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _pie_chart(
    labels: list[str],
    values: list[float],
    title: str,
    colors: list[str] | None = None,
    figsize: tuple = (4, 3.5),
) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    nonzero = [(lbl, v) for lbl, v in zip(labels, values, strict=False) if v > 0]
    if not nonzero:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ls, vs = zip(*nonzero, strict=False)
        palette = colors or ["#4CAF50", "#f44336", "#ff9800", "#b0bec5", "#4a90d9", "#9b59b6"]
        ax.pie(vs, labels=ls, autopct="%1.0f%%", colors=palette[:len(vs)], textprops={"fontsize": 8})
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _confusion_matrix_chart(tp: int, fp: int, fn: int, tn: int) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 3.5))
    matrix = np.array([[tp, fp], [fn, tn]])
    labels_map = np.array([["TP", "FP"], ["FN", "TN"]])
    colors = np.array([["#4CAF50", "#f44336"], ["#ff9800", "#b0bec5"]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fill=True,
                                        facecolor=colors[i][j], alpha=0.65))
            ax.text(j + 0.5, 1.5 - i, f"{labels_map[i][j]}\n{matrix[i][j]}",
                    ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Reported", "Not reported"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Not vuln", "Vulnerable"], fontsize=9)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _heatmap_chart(
    row_labels: list[str],
    col_labels: list[str],
    data: list[list[float]],
    title: str,
    figsize: tuple | None = None,
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    if figsize is None:
        figsize = (max(4, len(col_labels) * 1.8), max(3, len(row_labels) * 0.45 + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    arr = np.array(data, dtype=float)
    mask = arr < 0
    display = np.where(mask, 0.0, arr)
    im = ax.imshow(display, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = arr[i][j]
            txt = f"{val:.0%}" if val >= 0 else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val < 0.4 else "black")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _radar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str,
    figsize: tuple = (5, 4.5),
) -> str:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    palette = ["#4a90d9", "#e8723a", "#50c878", "#9b59b6"]
    for idx, (name, vals) in enumerate(series.items()):
        values = vals + vals[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=name,
                color=palette[idx % len(palette)])
        ax.fill(angles, values, alpha=0.1, color=palette[idx % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _timeline_chart(
    sequence: list[dict],
    title: str,
    figsize: tuple = (10, 3),
) -> str:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    categories = {
        "read": ({"read_file", "read_lines", "ls", "glob", "grep", "find_files"}, "#4a90d9"),
        "annotate": ({"annotate_trace", "annotate_sink", "annotate_validate"}, "#50c878"),
        "vuln": ({"report_vulnerability", "list_vulnerabilities", "get_vulnerability"}, "#f44336"),
        "skills": ({"skills_read", "search_memory", "note"}, "#9b59b6"),
        "code": ({"search_def", "search_refs", "find_symbol", "find_callers", "find_callees",
                   "attack_surface", "graph_summary"}, "#f1c40f"),
    }
    cat_order = ["read", "code", "skills", "annotate", "vuln"]

    def _categorize(tool: str) -> str:
        for cat, (names, _) in categories.items():
            if tool in names:
                return cat
        return "read"

    fig, ax = plt.subplots(figsize=figsize)
    for i, step in enumerate(sequence):
        cat = _categorize(step["tool"])
        _, color = categories.get(cat, (set(), "#ccc"))
        y = cat_order.index(cat) if cat in cat_order else 0
        ax.barh(y, 1, left=i, height=0.7, color=color, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([c.title() for c in cat_order], fontsize=9)
    ax.set_xlabel("Tool call index", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(color=categories[c][1], label=c.title()) for c in cat_order]
    ax.legend(handles=patches, fontsize=7, loc="upper right")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _traffic_light(recall: float) -> tuple[str, str]:
    if recall >= 0.50:
        return "#4CAF50", "GOOD"
    if recall >= 0.25:
        return "#ff9800", "FAIR"
    return "#f44336", "LOW"


def _agg(fixtures: list[dict]) -> dict[str, Any]:
    tp = sum(f["tp"] for f in fixtures)
    fp = sum(f["fp"] for f in fixtures)
    fn = sum(f["fn"] for f in fixtures)
    tn = sum(f["tn"] for f in fixtures)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1, "f2": f2,
                "tokens": sum(f["total_tokens"] for f in fixtures),
                "tools": sum(f["total_tool_calls"] for f in fixtures),
                "dur": sum(f["duration_s"] for f in fixtures)}


def _section_executive(data: dict) -> str:
    fixtures = data["fixtures"]
    if not fixtures:
        return "<p>No fixture results.</p>"

    a = _agg(fixtures)
    color, label = _traffic_light(a["r"])
    prompt_v = fixtures[0].get("prompt_version", "?")

    score_chart = _grouped_bar(
        [f["slug"].replace("realvuln-", "") for f in fixtures],
        {"Precision": [f["precision"] for f in fixtures],
         "Recall": [f["recall"] for f in fixtures],
         "F1": [f["f1"] for f in fixtures]},
        "Scores by Fixture", ylabel="Score",
        colors=["#4a90d9", "#4CAF50", "#f1c40f"],
    )

    classification_pie = _pie_chart(
        ["TP", "FP", "FN", "TN"],
        [a["tp"], a["fp"], a["fn"], a["tn"]],
        "Finding Classification",
        colors=["#4CAF50", "#f44336", "#ff9800", "#b0bec5"],
    )

    return f"""
    <div class="summary-grid">
      <div class="metric-card highlight" style="border-color:{color}">
        <div class="metric-value" style="color:{color}">{label}</div>
        <div class="metric-label">Recall Status</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{a['r']:.0%}</div>
        <div class="metric-label">Recall</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{a['p']:.0%}</div>
        <div class="metric-label">Precision</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{a['f1']:.3f}</div>
        <div class="metric-label">F1</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{a['f2']:.3f}</div>
        <div class="metric-label">F2</div>
      </div>
      <div class="metric-card tp">
        <div class="metric-value">{a['tp']}</div>
        <div class="metric-label">True Positives</div>
      </div>
      <div class="metric-card fp">
        <div class="metric-value">{a['fp']}</div>
        <div class="metric-label">False Positives</div>
      </div>
      <div class="metric-card fn">
        <div class="metric-value">{a['fn']}</div>
        <div class="metric-label">False Negatives</div>
      </div>
    </div>
    <table class="info-table">
      <tr><td>Date</td><td>{data.get("timestamp","?")[:19]}</td></tr>
      <tr><td>Model</td><td>{data.get("model","?")}</td></tr>
      <tr><td>Prompt Version</td><td>{prompt_v}</td></tr>
      <tr><td>Fixtures Tested</td><td>{len(fixtures)}</td></tr>
      <tr><td>Total Duration</td><td>{a['dur']:.0f}s</td></tr>
      <tr><td>Total Tokens</td><td>{a['tokens']:,}</td></tr>
      <tr><td>Total Tool Calls</td><td>{a['tools']}</td></tr>
      <tr><td>Tokens / TP</td><td>{a['tokens'] // max(a['tp'], 1):,}</td></tr>
    </table>
    <div class="charts-row">
      {_img(score_chart)}
      {_img(classification_pie)}
    </div>
    """


def _section_fixture_cards(data: dict) -> str:
    html_parts = []
    for f in data["fixtures"]:
        slug = f["slug"]
        short = slug.replace("realvuln-", "")
        color, _ = _traffic_light(f["recall"])

        # CWE detection chart
        cwe_chart = ""
        per_cwe = f.get("per_cwe", {})
        if per_cwe:
            cwes = sorted(per_cwe.keys())
            expected = [per_cwe[c]["expected"] for c in cwes]
            found = [per_cwe[c]["found"] for c in cwes]
            cwe_chart = _stacked_bar(
                cwes, {"Found (TP)": found,
                       "Missed (FN)": [e - fo for e, fo in zip(expected, found, strict=False)]},
                f"CWE Detection — {short}",
                colors=["#4CAF50", "#f44336"],
            )

        # Tool timeline
        timeline_chart = ""
        seq = f.get("tool_sequence", [])
        if seq:
            timeline_chart = _timeline_chart(seq, f"Agent Tool Sequence — {short}")

        # Tool usage bar
        tool_chart = ""
        tc = f.get("tool_counts", {})
        if tc:
            top = sorted(tc.items(), key=lambda x: -x[1])[:15]
            tool_chart = _bar_chart(
                [t[0] for t in top], [t[1] for t in top],
                f"Tool Usage — {short}", horizontal=True,
                figsize=(5, max(2, len(top) * 0.32)),
            )

        # Findings detail table
        gt_by_id = {c["id"]: c for c in f.get("gt_cases", [])}
        findings_rows = ""
        for m in f["matches"]:
            cls = m["classification"]
            cls_color = {"TP": "#4CAF50", "FP": "#f44336", "FN": "#ff9800", "TN": "#b0bec5"}.get(cls, "#ccc")
            gt_id = m.get("ground_truth_id", "—")
            gt_case = gt_by_id.get(gt_id, {})
            gt_cwe = gt_case.get("primary_cwe", "")
            gt_func = gt_case.get("function", "")
            gt_sev = gt_case.get("severity", "")
            gt_desc = gt_case.get("description", "")[:100]
            f_file = m.get("finding_file", "—")
            f_cwe = m.get("finding_cwe", "—")
            findings_rows += (
                f'<tr>'
                f'<td><span class="badge" style="background:{cls_color}">{cls}</span></td>'
                f'<td>{gt_id}</td><td>{gt_cwe}</td><td>{gt_func}</td><td>{gt_sev}</td>'
                f'<td>{f_file}</td><td>{f_cwe}</td>'
                f'<td class="desc">{gt_desc}</td>'
                f'</tr>\n'
            )

        # File coverage
        file_cov = f.get("file_coverage", {})
        cov_rows = ""
        for gf, was_read in sorted(file_cov.items()):
            icon = "&#9745;" if was_read else "&#9744;"
            cls_c = "#4CAF50" if was_read else "#f44336"
            cov_rows += f'<tr><td style="color:{cls_c}">{icon}</td><td>{gf}</td></tr>'

        # Severity distribution
        sev_counter: Counter[str] = Counter()
        for finding in f.get("reported_findings", []):
            sev_counter[finding.get("severity", "unknown")] += 1
        sev_order = ["critical", "high", "medium", "low", "info", "unknown"]
        sev_items = ", ".join(f"{s}: {sev_counter[s]}" for s in sev_order if sev_counter[s])

        html_parts.append(f"""
        <details class="fixture-card" open>
          <summary>
            <span class="fixture-name">{slug}</span>
            <span class="fixture-score" style="border-color:{color}">
              P={f["precision"]:.2f} R={f["recall"]:.2f} F1={f["f1"]:.2f}
              &nbsp;|&nbsp; TP={f["tp"]} FP={f["fp"]} FN={f["fn"]} TN={f["tn"]}
            </span>
            <span class="fixture-meta">{f["gt_vuln_count"]} vulns, {f["gt_fp_trap_count"]} traps, {f["duration_s"]:.0f}s, {f["total_tokens"]:,} tokens</span>
          </summary>
          <div class="fixture-body">
            <div class="charts-row">{_img(cwe_chart)}{_img(tool_chart)}</div>
            {_img(timeline_chart) if timeline_chart else ""}

            <h4>Finding Classification Detail</h4>
            <table class="findings-table">
              <thead><tr><th>Class</th><th>GT ID</th><th>GT CWE</th><th>GT Function</th><th>GT Sev</th><th>Found File</th><th>Found CWE</th><th>Description</th></tr></thead>
              <tbody>{findings_rows}</tbody>
            </table>

            <div class="two-col">
              <div>
                <h4>File Coverage ({sum(file_cov.values())}/{len(file_cov)} GT files read)</h4>
                <table class="compact-table"><tbody>{cov_rows}</tbody></table>
              </div>
              <div>
                <h4>Reported Severities</h4>
                <p>{sev_items or "none"}</p>
                <h4>Skills Loaded</h4>
                <p class="files-list">{", ".join(f.get("skills_loaded", [])) or "none"}</p>
                <h4>Vuln Tool Calls</h4>
                <p>{json.dumps(f.get("vuln_tools_used", {}))}</p>
                <h4>Annotation Tool Calls</h4>
                <p>{json.dumps(f.get("annotation_tools_used", {})) or "none"}</p>
              </div>
            </div>
          </div>
        </details>
        """)
    return "\n".join(html_parts)


def _section_classification(data: dict) -> str:
    fixtures = data["fixtures"]
    a = _agg(fixtures)

    cm_chart = _confusion_matrix_chart(a["tp"], a["fp"], a["fn"], a["tn"])

    # CWE heatmap
    all_cwes: set[str] = set()
    for f in fixtures:
        all_cwes.update(f.get("per_cwe", {}).keys())
    cwes = sorted(all_cwes)
    slugs = [f["slug"].replace("realvuln-", "") for f in fixtures]

    hm_chart = ""
    if cwes and slugs:
        hm_data = []
        for cwe in cwes:
            row = []
            for f in fixtures:
                pc = f.get("per_cwe", {}).get(cwe)
                if pc and pc["expected"] > 0:
                    row.append(pc["found"] / pc["expected"])
                else:
                    row.append(-0.1)
            hm_data.append(row)
        hm_chart = _heatmap_chart(cwes, slugs, hm_data, "CWE Detection Rate by Fixture")

    # Severity accuracy
    sev_chart = ""
    sev_match = {"match": 0, "mismatch": 0, "no_gt": 0}
    for f in fixtures:
        gt_by_id = {c["id"]: c for c in f.get("gt_cases", [])}
        for m in f["matches"]:
            if m["classification"] != "TP":
                continue
            gt_case = gt_by_id.get(m.get("ground_truth_id", ""))
            if not gt_case:
                continue
            gt_sev = gt_case.get("severity", "")
            found_finding = next(
                (fd for fd in f["reported_findings"]
                 if fd["file"] == m.get("finding_file")),
                None,
            )
            if found_finding:
                if found_finding.get("severity", "").lower() == gt_sev.lower():
                    sev_match["match"] += 1
                else:
                    sev_match["mismatch"] += 1
            else:
                sev_match["no_gt"] += 1
    if sum(sev_match.values()) > 0:
        sev_chart = _pie_chart(
            ["Correct", "Wrong", "N/A"],
            [sev_match["match"], sev_match["mismatch"], sev_match["no_gt"]],
            "Severity Accuracy (TP findings)",
            colors=["#4CAF50", "#f44336", "#b0bec5"],
        )

    # FP / FN detail tables
    fp_rows = ""
    fn_rows = ""
    for f in fixtures:
        short = f["slug"].replace("realvuln-", "")
        gt_by_id = {c["id"]: c for c in f.get("gt_cases", [])}

        # Find reported finding details for FPs
        for m in f["matches"]:
            if m["classification"] == "FP":
                finding_detail = next(
                    (fd for fd in f["reported_findings"]
                     if fd["file"] == m.get("finding_file") and fd.get("cwe") == m.get("finding_cwe")),
                    None
                )
                title = finding_detail.get("title", "?") if finding_detail else "?"
                fp_rows += (
                    f'<tr><td>{short}</td><td>{m.get("finding_file","?")}</td>'
                    f'<td>{m.get("finding_cwe","?")}</td><td>{title}</td>'
                    f'<td>{m.get("ground_truth_id","unmatched")}</td></tr>\n'
                )
            elif m["classification"] == "FN":
                gt_case = gt_by_id.get(m.get("ground_truth_id", ""), {})
                was_read = f.get("file_coverage", {}).get(gt_case.get("file", ""), False)
                read_icon = "&#9745;" if was_read else '<span style="color:#f44336">&#9744;</span>'
                fn_rows += (
                    f'<tr><td>{short}</td><td>{m.get("ground_truth_id","?")}</td>'
                    f'<td>{gt_case.get("primary_cwe","?")}</td>'
                    f'<td>{gt_case.get("function","?")}</td>'
                    f'<td>{gt_case.get("vulnerability_class","?")}</td>'
                    f'<td>{read_icon}</td>'
                    f'<td class="desc">{gt_case.get("description","")[:120]}</td></tr>\n'
                )

    return f"""
    <div class="charts-row">
      {_img(cm_chart)}
      {_img(hm_chart)}
      {_img(sev_chart)}
    </div>

    <h3>False Positives ({a['fp']})</h3>
    <table class="findings-table">
      <thead><tr><th>Fixture</th><th>File</th><th>CWE</th><th>Title</th><th>Matched GT</th></tr></thead>
      <tbody>{fp_rows or "<tr><td colspan=5>None</td></tr>"}</tbody>
    </table>

    <h3>False Negatives ({a['fn']})</h3>
    <table class="findings-table">
      <thead><tr><th>Fixture</th><th>GT ID</th><th>CWE</th><th>Function</th><th>Vuln Class</th><th>File Read?</th><th>Description</th></tr></thead>
      <tbody>{fn_rows or "<tr><td colspan=7>None</td></tr>"}</tbody>
    </table>
    """


def _section_tool_usage(data: dict) -> str:
    fixtures = data["fixtures"]

    # Aggregate tool counts
    agg_tools: Counter[str] = Counter()
    for f in fixtures:
        for name, count in f.get("tool_counts", {}).items():
            agg_tools[name] += count

    top = sorted(agg_tools.items(), key=lambda x: -x[1])[:20]
    agg_chart = _bar_chart(
        [t[0] for t in top], [t[1] for t in top],
        "Aggregate Tool Usage (all fixtures)", horizontal=True,
        figsize=(7, max(3, len(top) * 0.33)),
    ) if top else ""

    # Tool category breakdown
    cat_counts: dict[str, int] = defaultdict(int)
    read_tools = {"read_file", "read_lines", "ls", "glob", "grep", "find_files"}
    annot_tools = {"annotate_trace", "annotate_sink", "annotate_validate"}
    vuln_tools = {"report_vulnerability", "list_vulnerabilities", "get_vulnerability"}
    code_tools = {"search_def", "search_refs", "find_symbol", "find_callers",
                  "find_callees", "attack_surface", "graph_summary"}
    skill_tools = {"skills_read", "search_memory", "note"}

    for name, count in agg_tools.items():
        if name in read_tools:
            cat_counts["File I/O"] += count
        elif name in annot_tools:
            cat_counts["Annotation"] += count
        elif name in vuln_tools:
            cat_counts["Vuln Reporting"] += count
        elif name in code_tools:
            cat_counts["Code Analysis"] += count
        elif name in skill_tools:
            cat_counts["Skills/Memory"] += count
        else:
            cat_counts["Other"] += count

    cat_chart = _pie_chart(
        list(cat_counts.keys()), list(cat_counts.values()),
        "Tool Category Distribution",
        colors=["#4a90d9", "#50c878", "#f44336", "#f1c40f", "#9b59b6", "#b0bec5"],
    ) if cat_counts else ""

    vuln_total = sum(
        sum(f.get("vuln_tools_used", {}).values()) for f in fixtures
    )
    annot_total = sum(
        sum(f.get("annotation_tools_used", {}).values()) for f in fixtures
    )

    # Per-fixture tool comparison
    common_tools = [t[0] for t in top[:8]]
    if len(fixtures) > 1 and common_tools:
        per_fix = {}
        for f in fixtures:
            short = f["slug"].replace("realvuln-", "")
            per_fix[short] = [f.get("tool_counts", {}).get(t, 0) for t in common_tools]
        tool_compare = _grouped_bar(
            common_tools, per_fix,
            "Tool Usage by Fixture (top 8)", ylabel="Calls",
        )
    else:
        tool_compare = ""

    # Skills loading
    all_skills: Counter[str] = Counter()
    for f in fixtures:
        for s in f.get("skills_loaded", []):
            all_skills[s] += 1
    skills_rows = "".join(
        f"<tr><td>{s}</td><td>{c}</td></tr>" for s, c in all_skills.most_common()
    )

    # File coverage
    total_gt = sum(len(f.get("gt_files", [])) for f in fixtures)
    total_read = sum(sum(f.get("file_coverage", {}).values()) for f in fixtures)
    cov_pct = total_read / total_gt * 100 if total_gt else 0

    return f"""
    <div class="charts-row">
      {_img(agg_chart)}
      {_img(cat_chart)}
    </div>
    {_img(tool_compare) if tool_compare else ""}

    <div class="summary-grid" style="margin-top:1em">
      <div class="metric-card">
        <div class="metric-value">{sum(agg_tools.values())}</div>
        <div class="metric-label">Total Tool Calls</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{vuln_total}</div>
        <div class="metric-label">Vuln Reports Filed</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{annot_total}</div>
        <div class="metric-label">Annotations Made</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">{cov_pct:.0f}%</div>
        <div class="metric-label">GT File Coverage</div>
      </div>
    </div>

    <h3>Skills Reference Loading</h3>
    <table class="findings-table" style="max-width:500px">
      <thead><tr><th>Skill Reference</th><th>Times Loaded</th></tr></thead>
      <tbody>{skills_rows or "<tr><td colspan=2>No skills loaded</td></tr>"}</tbody>
    </table>
    """


def _section_tokens(data: dict) -> str:
    fixtures = data["fixtures"]
    slugs = [f["slug"].replace("realvuln-", "") for f in fixtures]

    token_chart = _stacked_bar(
        slugs,
        {"Input": [f["input_tokens"] for f in fixtures],
         "Output": [f["output_tokens"] for f in fixtures]},
        "Token Usage by Fixture", ylabel="Tokens",
    ) if slugs else ""

    # Tokens per finding
    tpf_chart = ""
    if len(fixtures) > 1:
        tpf_labels = slugs
        tpf_values = [f["total_tokens"] / max(f["tp"], 1) for f in fixtures]
        tpf_chart = _bar_chart(
            tpf_labels, tpf_values,
            "Tokens per True Positive", ylabel="Tokens/TP",
            color="#e8723a",
        )

    # Efficiency radar
    radar = ""
    if len(fixtures) > 1:
        radar_labels = ["Precision", "Recall", "F1", "File Coverage", "Tool Efficiency"]
        radar_series = {}
        for f in fixtures:
            short = f["slug"].replace("realvuln-", "")
            cov = sum(f.get("file_coverage", {}).values()) / max(len(f.get("file_coverage", {})), 1)
            tool_eff = 1 - f.get("tool_errors", 0) / max(f["total_tool_calls"], 1)
            radar_series[short] = [f["precision"], f["recall"], f["f1"], cov, tool_eff]
        radar = _radar_chart(radar_labels, radar_series, "Fixture Comparison Radar")

    rows = ""
    for f in fixtures:
        tpt = f["total_tokens"] / max(f["tp"], 1)
        rows += (
            f'<tr><td>{f["slug"].replace("realvuln-","")}</td>'
            f'<td>{f["input_tokens"]:,}</td>'
            f'<td>{f["output_tokens"]:,}</td>'
            f'<td>{f["total_tokens"]:,}</td>'
            f'<td>{f["llm_calls"]}</td>'
            f'<td>{tpt:,.0f}</td>'
            f'<td>{f["duration_s"]:.0f}s</td></tr>\n'
        )

    return f"""
    <div class="charts-row">
      {_img(token_chart)}
      {_img(tpf_chart)}
    </div>
    {_img(radar) if radar else ""}
    <table class="findings-table">
      <thead><tr><th>Fixture</th><th>Input Tok</th><th>Output Tok</th><th>Total Tok</th><th>LLM Calls</th><th>Tok/TP</th><th>Duration</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _section_agent_behavior(data: dict) -> str:
    fixtures = data["fixtures"]

    # Read→Annotate→Report workflow analysis
    workflow_rows = ""
    for f in fixtures:
        short = f["slug"].replace("realvuln-", "")
        seq = f.get("tool_sequence", [])
        total = len(seq)
        if total == 0:
            continue

        phases = {"discovery": 0, "analysis": 0, "reporting": 0}
        read_set = {"read_file", "read_lines", "ls", "glob", "grep", "find_files"}
        report_set = {"report_vulnerability", "list_vulnerabilities"}
        annot_set = {"annotate_trace", "annotate_sink", "annotate_validate"}
        code_set = {"search_def", "search_refs", "find_symbol", "find_callers", "find_callees"}

        first_report_idx = None
        first_annot_idx = None
        for i, step in enumerate(seq):
            t = step["tool"]
            if t in read_set:
                phases["discovery"] += 1
            elif t in report_set:
                phases["reporting"] += 1
                if first_report_idx is None:
                    first_report_idx = i
            elif t in annot_set:
                phases["analysis"] += 1
                if first_annot_idx is None:
                    first_annot_idx = i
            elif t in code_set:
                phases["analysis"] += 1

        reads_before_report = first_report_idx if first_report_idx is not None else total
        annotated_before_report = (
            first_annot_idx is not None and first_report_idx is not None
            and first_annot_idx < first_report_idx
        )

        workflow_rows += (
            f'<tr><td>{short}</td>'
            f'<td>{phases["discovery"]}</td>'
            f'<td>{phases["analysis"]}</td>'
            f'<td>{phases["reporting"]}</td>'
            f'<td>{reads_before_report}/{total}</td>'
            f'<td>{"Yes" if annotated_before_report else "<span style=color:#f44336>No</span>"}</td>'
            f'</tr>\n'
        )

    # Finding detail table - what the agent actually reported
    finding_rows = ""
    for f in fixtures:
        short = f["slug"].replace("realvuln-", "")
        for fd in f.get("reported_findings", []):
            sev = fd.get("severity", "?")
            sev_color = {"critical": "#d32f2f", "high": "#f44336", "medium": "#ff9800",
                         "low": "#4a90d9", "info": "#b0bec5"}.get(sev, "#ccc")
            finding_rows += (
                f'<tr><td>{short}</td>'
                f'<td>{fd.get("file","?")}</td>'
                f'<td>{fd.get("cwe","—")}</td>'
                f'<td><span class="badge" style="background:{sev_color}">{sev}</span></td>'
                f'<td>{fd.get("title","?")}</td></tr>\n'
            )

    return f"""
    <h3>Workflow Analysis</h3>
    <p>Does the agent follow the expected Read → Annotate → Report workflow?</p>
    <table class="findings-table">
      <thead><tr><th>Fixture</th><th>Discovery Calls</th><th>Analysis Calls</th><th>Report Calls</th><th>Reads Before 1st Report</th><th>Annotated Before Reporting?</th></tr></thead>
      <tbody>{workflow_rows}</tbody>
    </table>

    <h3>All Reported Findings ({sum(len(f.get("reported_findings",[])) for f in fixtures)})</h3>
    <table class="findings-table">
      <thead><tr><th>Fixture</th><th>File</th><th>CWE</th><th>Severity</th><th>Title</th></tr></thead>
      <tbody>{finding_rows}</tbody>
    </table>
    """


def _section_recommendations(data: dict) -> str:
    fixtures = data["fixtures"]
    recs: list[tuple[str, str, str]] = []  # (severity, title, detail)
    a = _agg(fixtures)

    if a["r"] < 0.30:
        recs.append(("critical", "Low Recall",
            f"Aggregate recall is {a['r']:.0%}. The agent misses most vulnerabilities. "
            "Add a whole-codebase scan mode to the prompt: enumerate entry points, "
            "triage by risk, trace each handler."))
    elif a["r"] < 0.70:
        recs.append(("medium", "Moderate Recall",
            f"Aggregate recall is {a['r']:.0%}. Room for improvement on specific CWE categories."))

    missed_cwes: Counter[str] = Counter()
    for f in fixtures:
        for cwe, info in f.get("per_cwe", {}).items():
            if info["found"] == 0 and info["expected"] > 0:
                missed_cwes[cwe] += info["expected"]
    if missed_cwes:
        top = missed_cwes.most_common(5)
        cwe_list = ", ".join(f"<code>{c}</code> ({n} missed)" for c, n in top)
        recs.append(("high", "Consistently Missed CWEs",
            f"{cwe_list}. Add detection heuristics for these CWEs to the sink catalogue "
            "and cwe-mapping skill reference."))

    all_skills: set[str] = set()
    for f in fixtures:
        all_skills.update(f.get("skills_loaded", []))
    if not all_skills:
        recs.append(("high", "No Skills/References Loaded",
            "The agent never called <code>skills_read</code>. "
            "Make reference loading mandatory in the prompt."))

    annot_total = sum(sum(f.get("annotation_tools_used", {}).values()) for f in fixtures)
    if annot_total == 0:
        recs.append(("medium", "No Annotation Tools Used",
            "The agent reported vulnerabilities without tracing data flows first. "
            "Consider requiring <code>annotate_trace</code>/<code>annotate_sink</code> "
            "before <code>report_vulnerability</code>."))

    if a["fp"] > a["tp"] and a["tp"] > 0:
        recs.append(("high", "High False Positive Rate",
            f"{a['fp']} FP vs {a['tp']} TP. Strengthen the FP filter checklist. "
            "Require verification against the 5-question quick FP filter before reporting."))

    total_gt_files = sum(len(f.get("gt_files", [])) for f in fixtures)
    total_cov = sum(sum(f.get("file_coverage", {}).values()) for f in fixtures)
    if total_gt_files > 0 and total_cov / total_gt_files < 0.7:
        recs.append(("medium", "Low File Coverage",
            f"Only {total_cov}/{total_gt_files} GT-relevant files were read. "
            "The agent needs a more systematic file discovery phase."))

    for f in fixtures:
        seq = f.get("tool_sequence", [])
        report_tools = {"report_vulnerability", "list_vulnerabilities"}
        first_report = next((i for i, s in enumerate(seq) if s["tool"] in report_tools), None)
        if first_report is not None and first_report < 3:
            recs.append(("low", f"Early Reporting in {f['slug'].replace('realvuln-','')}",
                f"First vulnerability report at tool call #{first_report+1}. "
                "Agent may be reporting before sufficient analysis."))
            break

    if not recs:
        recs.append(("info", "No Critical Issues",
            "Performance looks good. Consider expanding fixture coverage."))

    sev_colors = {"critical": "#d32f2f", "high": "#f44336", "medium": "#ff9800",
                  "low": "#4a90d9", "info": "#b0bec5"}
    items = ""
    for sev, title, detail in recs:
        color = sev_colors.get(sev, "#ccc")
        items += (
            f'<div class="rec-card">'
            f'<span class="rec-sev" style="background:{color}">{sev.upper()}</span>'
            f'<strong>{title}</strong><br/>{detail}</div>\n'
        )
    return items


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg: #f8f9fa; --card: #fff; --border: #e0e0e0; --text: #333; --muted: #777; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.55; padding: 2em; max-width: 1200px; margin: auto; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; }
h1 + p { color: var(--muted); font-size: 0.9em; margin-bottom: 1em; }
h2 { font-size: 1.25em; margin: 2em 0 0.6em; border-bottom: 2px solid var(--border); padding-bottom: 0.3em; color: #222; }
h3 { font-size: 1.05em; margin: 1.2em 0 0.4em; }
h4 { font-size: 0.9em; margin: 0.8em 0 0.3em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(115px, 1fr)); gap: 0.7em; margin: 1em 0; }
.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 0.7em; text-align: center; transition: box-shadow 0.15s; }
.metric-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.metric-card.highlight { border-width: 2px; }
.metric-card.tp { border-left: 3px solid #4CAF50; }
.metric-card.fp { border-left: 3px solid #f44336; }
.metric-card.fn { border-left: 3px solid #ff9800; }
.metric-value { font-size: 1.6em; font-weight: 700; }
.metric-label { font-size: 0.72em; color: var(--muted); margin-top: 0.15em; text-transform: uppercase; letter-spacing: 0.03em; }
.info-table { border-collapse: collapse; margin: 0.8em 0; font-size: 0.9em; }
.info-table td { padding: 0.25em 1.2em 0.25em 0; border-bottom: 1px solid #eee; }
.info-table td:first-child { font-weight: 600; color: var(--muted); min-width: 140px; }
.fixture-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; margin: 0.8em 0; }
.fixture-card summary { padding: 0.8em 1em; cursor: pointer; display: flex; align-items: center; gap: 0.8em; flex-wrap: wrap; }
.fixture-card summary:hover { background: #f5f5f5; border-radius: 8px; }
.fixture-name { font-weight: 700; font-size: 1em; }
.fixture-score { font-family: 'SF Mono', Menlo, monospace; font-size: 0.8em; padding: 0.2em 0.5em; border: 2px solid; border-radius: 4px; }
.fixture-meta { font-size: 0.78em; color: var(--muted); }
.fixture-body { padding: 0 1em 1.2em; }
.charts-row { display: flex; flex-wrap: wrap; gap: 1em; margin: 0.5em 0; align-items: flex-start; }
.charts-row img { max-width: 48%; height: auto; border-radius: 4px; }
@media (max-width: 800px) { .charts-row img { max-width: 100%; } }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin-top: 0.5em; }
@media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
.findings-table { border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 0.82em; }
.findings-table th, .findings-table td { padding: 0.35em 0.5em; border: 1px solid var(--border); text-align: left; }
.findings-table th { background: #f0f0f0; font-weight: 600; position: sticky; top: 0; }
.findings-table tr:nth-child(even) { background: #fafafa; }
.findings-table td.desc { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9em; color: var(--muted); }
.compact-table { font-size: 0.85em; border-collapse: collapse; }
.compact-table td { padding: 0.2em 0.5em; border-bottom: 1px solid #eee; }
.files-list { font-family: 'SF Mono', Menlo, monospace; font-size: 0.8em; color: var(--muted); word-break: break-all; }
.badge { display: inline-block; padding: 0.15em 0.5em; border-radius: 3px; color: white; font-size: 0.85em; font-weight: 600; }
.rec-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 0.8em 1em; margin: 0.6em 0; line-height: 1.6; }
.rec-sev { display: inline-block; padding: 0.1em 0.5em; border-radius: 3px; color: white; font-size: 0.75em; font-weight: 700; margin-right: 0.5em; text-transform: uppercase; }
footer { margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--border); font-size: 0.8em; color: var(--muted); }
"""


def generate_report(data: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")

    sections = [
        ("Executive Summary", _section_executive(data)),
        ("Per-Fixture Breakdown", _section_fixture_cards(data)),
        ("Finding Classification", _section_classification(data)),
        ("Agent Behavior Analysis", _section_agent_behavior(data)),
        ("Tool Usage Analytics", _section_tool_usage(data)),
        ("Token Efficiency", _section_tokens(data)),
        ("Improvement Recommendations", _section_recommendations(data)),
    ]

    body = ""
    for title, content in sections:
        body += f"<h2>{title}</h2>\n{content}\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trace Agent Vulnerability Detection — Eval Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Trace Agent Vulnerability Detection — Eval Report</h1>
<p>Automated analysis of trace_agent vulnerability detection capabilities across benchmark fixtures.</p>
{body}
<footer>
Generated by <code>scripts/vuln_eval_report.py</code> · {data.get("timestamp","")[:19]}
</footer>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: vuln_eval_report.py <eval_results.json> [output.html]")
        sys.exit(1)
    results_path = Path(sys.argv[1])
    data = json.loads(results_path.read_text())
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent / "report.html"
    generate_report(data, output)
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
