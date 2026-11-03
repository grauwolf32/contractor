// Synthetic read-only transport around the actual Operations route.
import {createRoot} from 'react-dom/client';
import {renderToStaticMarkup} from 'react-dom/server';
import {createMemoryRouter} from 'react-router';
import {Application} from '/src/app/application';
import {applicationRoutes} from '/src/app/router';
import {Icon} from '/src/app/icon';
import {PublicAPI} from '/src/api/client';
import {RunEventsManager} from '/src/events/run-events';
import {conforming} from '/src/test/builders';
import {operationsSnapshot} from '/src/test/builders-execution';
import {json,failure,emptyPageOrNotFound} from '/src/test/responses';
import config from '/@fs/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/valid/shared-config.json';
import '/src/app/theme.css';import '/src/styles.css';import '/src/app/ui-primitives.css';import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),scenario=params.get('case')??'attention';
const theme=params.get('theme')??'dark';localStorage.setItem('contractor.theme',theme);document.documentElement.dataset.theme=theme;
const digest='sha256:'+'1'.repeat(64),at='2026-11-02T12:00:00Z';
const agent=(number,slotState,observedState,extra={})=>{const value={
 instanceId:`runtime-${String(number).padStart(2,'0')}`,softwareVersion:'0.11.4',supportedRuntimes:['python@1','adk@1'],supportedToolsets:[{ref:'likec4@1',tools:['read_likec4','validate_likec4']}],supportedSandboxProfiles:['local-workdir@1'],supportedRuntimeAdapters:['http-capture@1'],slotState,observedState,lastAcceptedHeartbeat:at,confirmedLeaseUntil:'2026-11-02T12:01:00Z',...extra
};for(const key of Object.keys(value))if(value[key]===undefined)delete value[key];return conforming('RuntimeAgentObservation',value);};
const allocation=(number,authoritativePhase='active',observedPhase='prepared')=>conforming('AllocationObservation',{
 allocationId:`alloc-${String(number).padStart(2,'0')}`,runId:`run-analysis-${String(number).padStart(2,'0')}`,stageExecutionId:`stage-review-${number}`,runtimeAgentInstanceId:`runtime-${String(number).padStart(2,'0')}`,logicalWorker:number===4?'api-tracer':number===5?'evidence-analyst':'source-reviewer',agentTemplate:{templateId:'source-reviewer',version:'1',digest},executionConfig:{},authoritativePhase,observedPhase,metrics:{reportsComplete:false,modelCalls:0,inputTokens:0,outputTokens:0,totalTokens:0,toolCalls:0,toolFailures:0,errorCount:0,truncated:false}
});
const base=[agent(1,'idle','idle'),agent(2,'busy','allocated',{currentAllocationId:'alloc-02',authoritativeAllocationId:'alloc-02'}),agent(3,'busy','allocated',{currentAllocationId:'alloc-03',authoritativeAllocationId:'alloc-03'}),agent(4,'reserved','idle',{authoritativeAllocationId:'alloc-04',lastAcceptedHeartbeat:undefined,confirmedLeaseUntil:undefined})];
const normal=operationsSnapshot('42',{runtimeAgents:base.slice(0,3),allocations:[allocation(2),allocation(3)]});
const attention=operationsSnapshot('43',{runtimeAgents:[...base,agent(5,'draining','draining',{currentAllocationId:'alloc-05',authoritativeAllocationId:'alloc-05'}),agent(6,'fenced','fenced',{currentAllocationId:'alloc-observed-06',authoritativeAllocationId:'alloc-06',reconciliationReason:{code:'lease_confirmation_lost',retryable:true}})],allocations:[...normal.allocations,allocation(4,'preparing','absent'),allocation(5,'aborting','draining'),allocation(6,'active','fenced')]});
const fixtures={normal:{snapshot:normal,connection:'live'},attention:{snapshot:attention,connection:'live'},empty:{snapshot:operationsSnapshot('44'),connection:'live'},reconnecting:{snapshot:attention,connection:'reconnecting'},error:{snapshot:null,connection:'error'}};
const fixture=fixtures[scenario]??fixtures.attention;
const icons=Object.fromEntries(['operations','runs','settings','binding','projects','inbox','checks','issues','reports','library','evals','copy','history','target','docs','open','list'].map(name=>[name,renderToStaticMarkup(<Icon name={name} size={18}/>)]));
window.__operationsData={fixtures,icons,source:'actual public schema / OperationsSnapshot',capturedFixtureAt:at};
class DemoSocket{
 protocol='contractor.events.v1';readyState=0;onopen=null;onmessage=null;onerror=null;onclose=null;
 constructor(){queueMicrotask(()=>{this.readyState=1;this.onopen?.(new Event('open'))})}
 send(value){const frame=JSON.parse(value);if(frame.type==='subscribe')queueMicrotask(()=>{this.onmessage?.({data:JSON.stringify({version:'contractor.events.v1',type:'subscribed',subscriptionId:frame.subscriptionId,stream:frame.stream,cursor:frame.after})});if(scenario==='reconnecting')setTimeout(()=>{this.readyState=3;this.onclose?.({code:1006})},50)})}
 close(code=1000){this.readyState=3;this.onclose?.({code})}
}
const api=new PublicAPI({uiVersion:'0.11.4',supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url),path=url.pathname;
 if(request.method!=='GET')return failure(403,'forbidden','Read-only design reference');
 if(path==='/v1/config')return json(config);
 if(path==='/v1/auth/session')return json(conforming('AuthSession',{principal:{userId:'preview',username:'preview',capabilities:['user','operations']},csrfToken:'a'.repeat(43),idleExpiresAt:'2099-01-01T00:00:00Z',absoluteExpiresAt:'2099-01-02T00:00:00Z'}));
 if(path==='/v1/operations/snapshot')return fixture.snapshot===null?failure(503,'storage_unavailable','Synthetic Operations read failure'):json(fixture.snapshot);
 return emptyPageOrNotFound(request,url);
});
const events=new RunEventsManager(location.origin,{WebSocketImplementation:DemoSocket});
createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} runEvents={events} router={createMemoryRouter(applicationRoutes(),{initialEntries:['/operations']})}/>);
