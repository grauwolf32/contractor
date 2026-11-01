import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { listProjectFindings } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { loadAllPages } from "../../api/pages";
import { Button } from "../../ui";

export function DuplicatePicker({projectId,findingId,value,onChange,searchId}:{projectId:string;findingId:string;value:string;onChange:(id:string)=>void;searchId?:string}) {
 const api=usePublicAPI();const [search,setSearch]=useState("");
 const candidates=useQuery({queryKey:["project-findings",projectId,"duplicate-targets"],queryFn:()=>loadAllPages(cursor=>listProjectFindings(api,projectId,cursor===undefined?{}:{cursor})),retry:false});
 const choices=(candidates.data?.items??[]).filter(item=>item.findingId!==findingId && (item.title+" "+item.findingId).toLowerCase().includes(search.toLowerCase()));
 return <div className="decisions-duplicate-picker">
 <label htmlFor={searchId}>Original possible issue</label>
 <input id={searchId} value={search} onChange={event=>setSearch(event.target.value)} placeholder="Search this Project" />
 <label>Finding ID<input value={value} onChange={event=>onChange(event.target.value)} /></label>
 {candidates.error===null?null:<p role="alert">Candidates could not be loaded. You can enter an ID, or <Button size="xs" onClick={()=>void candidates.refetch()}>retry</Button>.</p>}
 <ul>{choices.slice(0,20).map(item=><li key={item.findingId}><Button size="xs" onClick={()=>onChange(item.findingId)}>{item.title}</Button></li>)}</ul>
 {choices.length>20?<p className="ui-caption">Refine the search to see more matches.</p>:null}
 </div>;
}
