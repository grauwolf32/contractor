import { useMutation,useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { editProjectFinding,type ProjectFinding,type ProjectFindingEditResult,type FindingVerdict,type FindingSeverity } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { Button,DecisionBar } from "../../ui";
import { findingDecisionBody,findingDraftProblem,MAX_RATIONALE_BYTES,SEVERITY_OPTIONS,VERDICT_ACTIONS } from "./model";
import { DuplicatePicker } from "./duplicate-picker";
import "./decisions.css";

type EditOptions=Parameters<typeof editProjectFinding>[1];
interface Props {
 finding:ProjectFinding;
 auditId?:string;
 onDecided?:((result:ProjectFindingEditResult)=>void)|undefined;
 onRecording?:((pending:boolean)=>void)|undefined;
 next?:{label:string;onNext:()=>void}|undefined;
 autoFocus?:boolean|undefined;
 shortcuts?:boolean|undefined;
}
function Editor({finding,onDecided,onRecording,next,shortcuts=true}:Props) {
 const api=usePublicAPI();const client=useQueryClient();
 const [baseRevision,setBaseRevision]=useState(finding.revision);
 const [verdict,setVerdict]=useState<FindingVerdict|undefined>(undefined);
 const [severity,setSeverity]=useState<FindingSeverity|undefined>(finding.analystSeverity??undefined);
 const [rationale,setRationale]=useState("");const [target,setTarget]=useState("");
 const [notice,setNotice]=useState("");const [keys]=useState(()=>new MutationDraftKeyring<EditOptions>("project-finding"));
 const refresh=async()=>{await Promise.all([client.invalidateQueries({queryKey:["project-findings"]}),client.invalidateQueries({queryKey:["audits"]}),client.invalidateQueries({queryKey:["issues"]}),client.invalidateQueries({queryKey:["projects"]}),invalidateCrossProject(client)]);};
 const mutation=useMutation({mutationFn:(options:EditOptions)=>editProjectFinding(api,options),onSuccess:async result=>{
  setBaseRevision(result.finding.revision);setNotice("Assessment saved.");setRationale("");
  await refresh();onDecided?.(result);
 },onError:async()=>{await refresh();},onSettled:()=>{onRecording?.(false);}});
 const draft=verdict===undefined?undefined:{verdict,severity,rationale,duplicateTargetId:target};
 const problem=draft===undefined?undefined:findingDraftProblem(draft,finding.findingId);
 const stale=finding.revision!==baseRevision;
 const submit=()=>{
  if(draft===undefined || problem!==undefined || stale || mutation.isPending)return;
  const decision=findingDecisionBody(draft);
  const base={projectId:finding.projectId,findingId:finding.findingId,expectedRevision:baseRevision,decision,idempotencyKey:""};
  const options={...base,idempotencyKey:keys.keyFor(base)};
  onRecording?.(true);setNotice("");mutation.mutate(options);
 };
 return <section aria-label="Assess possible issue" className="decisions-finding">
 <p className="ui-caption">This assessment belongs to the Project and is shared by every check.</p>
 {notice===""?null:<p role="status">{notice}</p>}
 {stale?<p role="alert">This possible issue changed. Your draft is still here. <Button size="xs" onClick={()=>{setBaseRevision(finding.revision);mutation.reset();}}>Use current revision</Button></p>:null}
 <DecisionBar options={(Object.keys(VERDICT_ACTIONS) as FindingVerdict[]).map(id=>({id,label:VERDICT_ACTIONS[id],...(shortcuts?{shortcut:({true_positive:"c",false_positive:"r",needs_evidence:"e",duplicate:"d",reopen:"o"})[id]}:{})}))}
 selected={verdict} onSelect={value=>{setVerdict(value as FindingVerdict);setNotice("");}}
 severity={verdict==="true_positive"?{options:SEVERITY_OPTIONS,value:severity,onChange:value=>setSeverity(value as FindingSeverity),required:true}:undefined}
 rationale={{value:rationale,onChange:setRationale,maxLength:MAX_RATIONALE_BYTES,hint:"Required. Saved with your assessment."}}
 onSubmit={submit} submitLabel="Save assessment" pending={mutation.isPending} next={next}
 disabledReason={stale?"Review the current revision before saving.":problem}
 extra={verdict==="duplicate"?<DuplicatePicker projectId={finding.projectId} findingId={finding.findingId} value={target} onChange={setTarget}/>:undefined}
 error={mutation.error===null?undefined:<span>{mutation.error.message} {mutation.variables===undefined?null:<Button size="xs" onClick={()=>{onRecording?.(true);mutation.mutate(mutation.variables!);}}>Retry this save</Button>}</span>}/>
 </section>;
}
export function FindingDecision(props:Props) {return <Editor key={props.finding.findingId} {...props}/>;}
