import type {ReactNode} from "react";
import type {To} from "react-router";
import {Link} from "react-router";
import type {ProjectFindingSummary} from "../../api/audits";
import {findingStateLabel,severityLabel} from "../../app/vocabulary";
import {StatusChip} from "../../ui";
export function IssueRow({finding,to,selected=false,context,id,onSelect,trailing}:{finding:ProjectFindingSummary;to:To;selected?:boolean|undefined;context?:ReactNode;id?:string|undefined;onSelect?:(()=>void)|undefined;trailing?:ReactNode}) {
 const state=findingStateLabel(finding.state);
 return <article id={id} className="issues-row" data-selected={selected?"":undefined}><Link to={to} onClick={onSelect}>{finding.title}</Link><p>{context}</p><p><StatusChip tone={state.tone}>{state.label}</StatusChip> · Severity: {severityLabel(finding.analystSeverity)}</p>{trailing}</article>;
}
