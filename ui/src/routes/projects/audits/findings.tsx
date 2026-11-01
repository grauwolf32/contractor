import {useQuery} from "@tanstack/react-query";
import {useParams} from "react-router";
import {listProjectFindings} from "../../../api/audits";
import {usePublicAPI} from "../../../api/context";
import {loadAllPages} from "../../../api/pages";
import {IssueRow} from "../../issues/issue-row";
export function ProjectFindingsRoute() {
 const {projectId=""}=useParams();const api=usePublicAPI();
 const findings=useQuery({queryKey:["project-findings",projectId,"list"],queryFn:()=>loadAllPages(cursor=>listProjectFindings(api,projectId,cursor===undefined?{}:{cursor})),retry:false});
 return <section><h2>Possible issues</h2>{findings.error===null?null:<p role="alert">{findings.error.message}</p>}{findings.isPending?<p role="status">Loading possible issues…</p>:null}<ul>{findings.data?.items.map(finding=><li key={finding.findingId}><IssueRow finding={finding} to={`/issues/${encodeURIComponent(projectId)}/${encodeURIComponent(finding.findingId)}`}/></li>)}</ul></section>;
}
