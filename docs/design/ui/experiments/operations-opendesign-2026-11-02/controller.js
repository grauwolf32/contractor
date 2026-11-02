const DATA=JSON.parse(document.getElementById('fixture-data').textContent);
const params=new URLSearchParams(location.search),scenario=Object.hasOwn(DATA.fixtures,params.get('case'))?params.get('case'):'attention';
document.documentElement.dataset.theme=['light','dark','black'].includes(params.get('theme'))?params.get('theme'):'dark';
const fixture=DATA.fixtures[scenario],snapshot=fixture.snapshot;
const $=id=>document.getElementById(id);
const icon=name=>DATA.icons[name]??DATA.icons.docs;
const text=(tag,value,className)=>{const node=document.createElement(tag);node.textContent=String(value);if(className)node.className=className;return node};
const mark=(name,className='')=>{const node=document.createElement('span');node.className=className;node.setAttribute('aria-hidden','true');node.innerHTML=icon(name);return node};
const tones={idle:'neutral',busy:'progress',reserved:'progress',draining:'warning',fenced:'danger',active:'progress',prepared:'progress',preparing:'progress',aborting:'warning',absent:'neutral'};
const chip=(value,prefix='')=>{const node=text('span',prefix+value,'status-chip');node.dataset.tone=tones[value]??'neutral';return node};
const known=value=>value??'Not reported';
const mismatch=agent=>agent.reconciliationReason!==undefined||agent.currentAllocationId!==agent.authoritativeAllocationId;
const hasSnapshot=snapshot!==null;
const counts=hasSnapshot?Object.fromEntries(['idle','busy','reserved','draining','fenced'].map(state=>[state,snapshot.runtimeAgents.filter(agent=>agent.slotState===state).length])):null;
const mismatches=hasSnapshot?snapshot.runtimeAgents.filter(mismatch):[];
// A reserved assignment with an idle observation is explained as reserved,
// while reconciliation uses the same exact-ID comparison as current Overview.
function facts(items){const list=document.createElement('dl');list.className='fact-list';for(const [label,value] of items){const row=document.createElement('div');row.append(text('dt',label),text('dd',value));list.append(row)}return list}
function link(label,href,className=''){const node=text('a',label,className);node.href=href;return node}
function announce(message){$('action-status').textContent=message}
function copyButton(value,label){const button=document.createElement('button');button.className='icon-button';button.type='button';button.title=label;button.setAttribute('aria-label',label);button.append(mark('copy'));button.onclick=async()=>{try{if(!navigator.clipboard)throw Error('Unavailable');await navigator.clipboard.writeText(value);announce('Copied '+label.toLowerCase()+'.')}catch{announce(label+': '+value+'. Select the exact value in the details to copy it.')}};return button}
function fillSummary(){
 $('connection').dataset.tone=fixture.connection==='live'?'success':'warning';$('connection').append(mark(fixture.connection==='live'?'binding':'history'),text('span',fixture.connection==='live'?'Live updates':fixture.connection==='reconnecting'?'Reconnecting':'Read failed'));
 $('connection-note').textContent=fixture.connection==='reconnecting'?'Live updates are disconnected. Showing the last snapshot.':fixture.connection==='error'?'No current snapshot is available.':'Current-state snapshot · synthetic preview';
 for(const [id,value] of [['process-count',snapshot?.runtimeAgents.length],['allocation-count',snapshot?.allocations.length],['mismatch-count',hasSnapshot?mismatches.length:undefined]])$(id).textContent=value??'—';
 $('mismatch-card').dataset.tone=!hasSnapshot?'unknown':mismatches.length?'warning':'neutral';
 $('snapshot-content').hidden=!hasSnapshot;$('read-error').hidden=hasSnapshot;
 if(!hasSnapshot)return;
 $('slot-count').textContent=snapshot.runtimeAgents.length+' single-slot processes';
 const track=$('slot-track');track.setAttribute('aria-label',Object.entries(counts).map(([state,count])=>count+' '+state).join(', '));
 for(const [state,count]of Object.entries(counts)){
  const li=document.createElement('li');li.append(chip(state),text('strong',count));$('slot-legend').append(li);
  if(count){const part=document.createElement('span');part.dataset.state=state;part.style.flexGrow=String(count);part.title=count+' '+state;part.setAttribute('aria-hidden','true');track.append(part)}
 }
 if(!snapshot.runtimeAgents.length){track.hidden=true;$('empty-runtime').hidden=false;}
 const inspection=$('inspection-body');
 if(mismatches.length){for(const agent of mismatches){const item=document.createElement('div');item.className='inspection-item';item.append(mark('issues'),link(agent.instanceId,'#agent-'+agent.instanceId),text('p',agent.reconciliationReason?.code.replaceAll('_',' ')??(agent.slotState==='reserved'?'Reserved — awaiting a Runtime allocation observation.':'Observed allocation differs from its assignment.')));inspection.append(item)}}
 else inspection.append(text('p','No reconciliation mismatch in this snapshot.','quiet'));
 if(counts.draining)inspection.append(text('p',counts.draining+' draining slot'+(counts.draining===1?'':'s')+' · inspect its current allocation.','drain-note'));
 $('inspection').dataset.tone=mismatches.length?'warning':'neutral';
 $('agent-list-count').textContent=snapshot.runtimeAgents.length;$('allocation-list-count').textContent=snapshot.allocations.length;
 $('agent-list-empty').hidden=snapshot.runtimeAgents.length>0;$('allocation-list-empty').hidden=snapshot.allocations.length>0;
}
function renderAgents(){if(!hasSnapshot)return;for(const agent of snapshot.runtimeAgents){
 const details=document.createElement('details');details.className='agent-row';details.id='agent-'+agent.instanceId;details.dataset.warning=String(mismatch(agent));
 const summary=document.createElement('summary');const heading=document.createElement('div');heading.className='row-heading';heading.append(mark('operations','row-icon'),text('strong',agent.instanceId));const status=chip(agent.slotState);status.classList.add('row-state');
 const context=text('span',agent.lastAcceptedHeartbeat===undefined?'Heartbeat not reported':agent.authoritativeAllocationId?'Assigned '+agent.authoritativeAllocationId:'No allocation assigned','row-context');summary.append(heading,status,context,mark('open','row-open'));
 const body=document.createElement('div');body.className='row-details';
 const header=document.createElement('div');header.className='row-details-title';header.append(text('strong','Process diagnostics'),copyButton(agent.instanceId,'Copy process ID '+agent.instanceId));
 body.append(header,facts([['Process ID',agent.instanceId],['Observed process',agent.observedState],['Reconciled slot',agent.slotState],['Observed allocation',agent.currentAllocationId??'None reported'],['Authoritative allocation',agent.authoritativeAllocationId??'None assigned'],['Last accepted heartbeat',known(agent.lastAcceptedHeartbeat)],['Confirmed lease until',known(agent.confirmedLeaseUntil)],['Software',agent.softwareVersion],['Runtimes',agent.supportedRuntimes.join(', ')],['Sandbox profiles',agent.supportedSandboxProfiles.join(', ')],['Runtime adapters',agent.supportedRuntimeAdapters.join(', ')||'None reported'],['Toolsets',agent.supportedToolsets.map(set=>set.ref+': '+set.tools.join(', ')).join('; ')],['Reconciliation reason',agent.reconciliationReason?agent.reconciliationReason.code+' · '+(agent.reconciliationReason.retryable?'retryable':'not retryable'):'None reported']]));
 const allocation=snapshot.allocations.find(row=>row.allocationId===agent.authoritativeAllocationId);
 if(allocation)body.append(link('Inspect assigned allocation →','#allocation-'+allocation.allocationId,'details-link'));
 body.append(link('Open Runtime Agents →','/operations/runtime-agents','details-link'));details.append(summary,body);$('agent-list').append(details);
}}
function renderAllocations(){if(!hasSnapshot)return;for(const allocation of snapshot.allocations){
 const details=document.createElement('details');details.className='allocation-row';details.id='allocation-'+allocation.allocationId;
 const summary=document.createElement('summary');const heading=document.createElement('div');heading.className='row-heading';heading.append(mark('runs','row-icon'),text('strong',allocation.logicalWorker),text('code',allocation.allocationId));
 const states=document.createElement('div');states.className='allocation-states';states.append(chip(allocation.authoritativePhase,'Assigned: '),chip(allocation.observedPhase,'Runtime: '));summary.append(heading,states,mark('open','row-open'));
 const body=document.createElement('div');body.className='row-details';body.append(facts([['Allocation',allocation.allocationId],['Run',allocation.runId],['Stage execution',allocation.stageExecutionId],['Runtime process',allocation.runtimeAgentInstanceId],['Authoritative phase',allocation.authoritativePhase],['Observed phase',allocation.observedPhase],['Agent template',allocation.agentTemplate.templateId+'@'+allocation.agentTemplate.version],['Template digest',allocation.agentTemplate.digest],['Metrics',allocation.metrics.reportsComplete?'Final reports complete':'Pending final report; live counters are incomplete.']]));
 body.append(link('Open Run →','/runs/'+encodeURIComponent(allocation.runId),'details-link'),copyButton(allocation.allocationId,'Copy allocation ID '+allocation.allocationId));details.append(summary,body);$('allocation-list').append(details);
}}
function renderDiagnostics(){const body=$('snapshot-diagnostics-body');body.append(facts([['Generation',snapshot?.cursor.generation??'Unavailable'],['Snapshot revision',snapshot?.cursor.revision??'Unavailable'],['Live connection',fixture.connection],['Source','Synthetic OperationsSnapshot fixture']]));}
fillSummary();renderAgents();renderAllocations();renderDiagnostics();
for(const button of document.querySelectorAll('[data-refresh]'))button.onclick=()=>announce(hasSnapshot?'Preview refreshed the same synthetic snapshot; no deployment request was sent.':'Synthetic read failure remains selected. Choose another example in the preview toolbar.');
document.addEventListener('click',event=>{const anchor=event.target.closest('a');if(!anchor)return;event.preventDefault();const href=anchor.getAttribute('href');if(href.startsWith('#')){const target=document.getElementById(href.slice(1));if(target){if(target instanceof HTMLDetailsElement)target.open=true;target.scrollIntoView({block:'nearest',behavior:'smooth'});target.querySelector('summary')?.focus();return}}
 announce('Preview navigation: '+anchor.textContent.trim()+' → '+href+'. No deployment state changed.');});
const mobile=matchMedia('(max-width:820px)');function adapt(){const nav=$('sections-browser');nav.open=!mobile.matches;}$('mobile-menu').onclick=()=>{$('global-rail').dataset.open=String($('global-rail').dataset.open!=='true');$('mobile-menu').setAttribute('aria-expanded',$('global-rail').dataset.open);};mobile.addEventListener('change',adapt);adapt();
window.__operationsReview={scenario,summary:()=>({counts,mismatches:hasSnapshot?mismatches.length:null,processes:snapshot?.runtimeAgents.length??null,allocations:snapshot?.allocations.length??null,connection:fixture.connection}),snapshot:()=>snapshot};
