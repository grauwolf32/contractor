import {useParams,useSearchParams} from "react-router";
import {useAllPossibleIssues} from "../../api/cross-project";
import type {FindingState} from "../../api/audits";
import {useDocumentTitle} from "../../app/document-title";
import {FINDING_STATE_LABELS} from "../../app/vocabulary";
import {Button} from "../../ui";
import {IssueDetail} from "./issue-detail";
import {IssueRow} from "./issue-row";
import "./issues.css";

export function IssuesRoute() {
 useDocumentTitle("Possible issues");const {projectId,findingId}=useParams();const [params,setParams]=useSearchParams();
 const selected=params.get("state")??"all";const state=Object.hasOwn(FINDING_STATE_LABELS,selected)?selected as FindingState:undefined;
 const issues=useAllPossibleIssues({...(state===undefined?{}:{states:[state]}),...(selected==="unreviewed"?{verdicts:["unreviewed"]}:{})});
 const projectFilter=params.get("project");const shown=issues.issues.filter(issue=>projectFilter===null || issue.project.projectId===projectFilter);
 return <main className="issues-destination"><header><h1>Possible issues</h1><p>One current assessment per Project issue, shared by every check.</p>
 <label>Status <select value={selected} onChange={event=>{const next=new URLSearchParams(params);next.set("state",event.target.value);void setParams(next);}}><option value="all">All</option><option value="unreviewed">Not assessed</option>{Object.entries(FINDING_STATE_LABELS).map(([value,label])=><option key={value} value={value}>{label.label}</option>)}</select></label>
 <Button size="xs" onClick={()=>void issues.refetch()}>Refresh</Button></header>
 {issues.error===null && !issues.partial?null:<p role="alert">Some possible issues could not be loaded. The list may be incomplete.</p>}
 {issues.isPending?<p role="status">Loading possible issues…</p>:null}
 <div className="project-finding-panes"><section aria-label="Possible issues"><ul>{shown.map(issue=><li key={issue.finding.findingId}><IssueRow finding={issue.finding} context={issue.project.name} selected={findingId===issue.finding.findingId} to={{pathname:`/issues/${encodeURIComponent(issue.finding.projectId)}/${encodeURIComponent(issue.finding.findingId)}`,search:params.toString()}}/></li>)}</ul>
 {!issues.isPending && shown.length===0?<p>No possible issues match this selection.</p>:null}</section>
 <section aria-label="Selected possible issue">{projectId===undefined || findingId===undefined?<p>Select a possible issue to assess it.</p>:<IssueDetail key={findingId} projectId={projectId} findingId={findingId}/>}</section></div></main>;
}
