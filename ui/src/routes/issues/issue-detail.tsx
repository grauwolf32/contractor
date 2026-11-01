import {useQuery} from "@tanstack/react-query";
import {getProjectFinding} from "../../api/audits";
import {usePublicAPI} from "../../api/context";
import {Button,TechnicalDetails} from "../../ui";
import {FindingDecision,FindingSummary} from "../decisions";
import {FindingSources} from "../projects/audits/finding-card";

export function IssueDetail({projectId,findingId}:{projectId:string;findingId:string}) {
 const api=usePublicAPI();const finding=useQuery({queryKey:["project-findings",projectId,findingId],queryFn:()=>getProjectFinding(api,projectId,findingId),retry:false});
 if(finding.error!==null)return <section role="alert">Possible issue could not be loaded: {finding.error.message} <Button onClick={()=>void finding.refetch()}>Retry</Button></section>;
 if(finding.data===undefined)return <p role="status">Loading possible issue…</p>;
 const value=finding.data;
 return <article><FindingSummary finding={value}/><FindingDecision finding={value}/><FindingSources key={value.findingId} finding={value}/><TechnicalDetails><p>Project {projectId} · Finding {findingId} · Revision {value.revision}</p></TechnicalDetails></article>;
}
