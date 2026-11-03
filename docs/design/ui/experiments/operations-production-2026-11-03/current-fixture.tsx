// Synthetic in-memory transport around the actual Operations routes.
import {createRoot} from 'react-dom/client';
import {renderToStaticMarkup} from 'react-dom/server';
import {createMemoryRouter} from 'react-router';
import {Application} from '/src/app/application';
import {applicationRoutes} from '/src/app/router';
import {Icon} from '/src/app/icon';
import {initializeTheme} from '/src/app/theme';
import {PublicAPI} from '/src/api/client';
import {RunEventsManager} from '/src/events/run-events';
import {conforming} from '/src/test/builders';
import {operationsSnapshot} from '/src/test/builders-execution';
import {json,failure,emptyPageOrNotFound,page} from '/src/test/responses';
import packageMetadata from '/package.json';
import config from '/@fs/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/valid/shared-config.json';
import '/src/app/theme.css';import '/src/styles.css';import '/src/app/ui-primitives.css';import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),scenario=params.get('case')??'attention';
const theme=params.get('theme')??'dark';localStorage.setItem('contractor.theme',theme);initializeTheme();
const digest='sha256:'+'1'.repeat(64),at='2026-11-03T12:00:00Z';
const agent=(number,slotState,observedState,extra={})=>{const value={
 instanceId:`runtime-${String(number).padStart(2,'0')}`,softwareVersion:'0.11.4',supportedRuntimes:['python@1','adk@1'],supportedToolsets:[{ref:'likec4@1',tools:['read_likec4','validate_likec4']}],supportedSandboxProfiles:['local-workdir@1'],supportedRuntimeAdapters:['http-capture@1'],slotState,observedState,lastAcceptedHeartbeat:at,confirmedLeaseUntil:'2026-11-03T12:01:00Z',...extra
};for(const key of Object.keys(value))if(value[key]===undefined)delete value[key];return conforming('RuntimeAgentObservation',value);};
const allocation=(number,authoritativePhase='active',observedPhase='prepared')=>conforming('AllocationObservation',{
 allocationId:`alloc-${String(number).padStart(2,'0')}`,runId:`run-analysis-${String(number).padStart(2,'0')}`,stageExecutionId:`stage-review-${number}`,runtimeAgentInstanceId:`runtime-${String(number).padStart(2,'0')}`,logicalWorker:number===4?'api-tracer':number===5?'evidence-analyst':'source-reviewer',agentTemplate:{templateId:'source-reviewer',version:'1',digest},executionConfig:{},authoritativePhase,observedPhase,metrics:{reportsComplete:false,modelCalls:0,inputTokens:0,outputTokens:0,totalTokens:0,toolCalls:0,toolFailures:0,errorCount:0,truncated:false}
});
const base=[agent(1,'idle','idle'),agent(2,'busy','allocated',{currentAllocationId:'alloc-02',authoritativeAllocationId:'alloc-02'}),agent(3,'busy','allocated',{currentAllocationId:'alloc-03',authoritativeAllocationId:'alloc-03'}),agent(4,'reserved','idle',{authoritativeAllocationId:'alloc-04',lastAcceptedHeartbeat:undefined,confirmedLeaseUntil:undefined})];
const normal=operationsSnapshot('42',{runtimeAgents:base.slice(0,3),allocations:[allocation(2),allocation(3)]});
const attention=operationsSnapshot('43',{runtimeAgents:[...base,agent(5,'draining','draining',{currentAllocationId:'alloc-05',authoritativeAllocationId:'alloc-05'}),agent(6,'fenced','fenced',{currentAllocationId:'alloc-observed-06',authoritativeAllocationId:'alloc-06',reconciliationReason:{code:'lease_confirmation_lost',retryable:true}})],allocations:[...normal.allocations,allocation(4,'preparing','absent'),allocation(5,'aborting','draining'),allocation(6,'active','fenced')]});
const fixtures={normal:{snapshot:normal,connection:'live'},attention:{snapshot:attention,connection:'live'},empty:{snapshot:operationsSnapshot('44'),connection:'live'},reconnecting:{snapshot:attention,connection:'reconnecting'},error:{snapshot:null,connection:'error'}};
const fixture=fixtures[scenario]??fixtures.attention;
const paths={overview:'/operations','runtime-agents':'/operations/runtime-agents',allocations:'/operations/allocations',completed:'/operations/allocations/completed',performance:'/operations/performance',configuration:'/operations/configuration',llm:'/operations/configurations',credentials:'/operations/credentials',settings:'/operations/settings'};
const section=params.get('section')??'overview';
const principals=(fixture.snapshot?.runtimeAgents??[]).map((live,i)=>conforming('RuntimeAgentPrincipal',{runtimeAgentId:String(i+1).repeat(64),labels:i===0?['default']:i===1?['source-review']:i===3?['api']:[],revision:'1',availability:i===0?'available':i<3?'busy':'slot_unavailable',requiredRuntimeAdapters:[],missingRuntimeAdapters:[],createdBy:'demo-operator',createdAt:at,updatedBy:'demo-operator',updatedAt:at,live}));
if(scenario!=='empty')principals.push(conforming('RuntimeAgentPrincipal',{runtimeAgentId:'7'.repeat(64),labels:['staging'],revision:'1',availability:'offline',requiredRuntimeAdapters:[],missingRuntimeAdapters:[],createdBy:'demo-operator',createdAt:at,updatedBy:'demo-operator',updatedAt:at}));
const runtimeConfigs=['default','api-review','source-review'].map((name,i)=>conforming('RuntimeConfigResource',{ref:{name,version:'1',digest},document:{apiVersion:'contractor/v1alpha1',kind:'RuntimeConfig',metadata:{name,version:'1'},spec:{}},builtIn:i===0,createdBy:'demo-operator',createdAt:at}));
const bindings=['default','api'].map((label,i)=>conforming('RuntimeLabelBinding',{label,config:runtimeConfigs[i].ref,revision:'1',createdBy:'demo-operator',createdAt:at,updatedBy:'demo-operator',updatedAt:at}));
const runtimeCredentials=[['demo-telemetry','otlp-headers@1'],['demo-api-access','target-bearer@1']].map(([credentialId,kind])=>conforming('RuntimeCredentialMetadata',{credentialId,kind,createdBy:'demo-operator',createdAt:at}));
const configResources={
 'model-policies':['worker','planner'].map(name=>conforming('ConfigurationResource',{ref:{kind:'model-policies',name,version:'1',digest},body:{model:'demo-default',contextWindowTokens:131072,maxOutputTokens:8192,maxModelCalls:24,maxToolCalls:80},source:'operator'})),
 'llm-gateways':[conforming('ConfigurationResource',{ref:{kind:'llm-gateways',name:'local-litellm',version:'1',digest},body:{protocol:'openai-compatible@1',url:'https://gateway.example.invalid/v1',credentialManager:{implementation:'litellm-virtual-keys@1',managementUrl:'https://gateway.example.invalid'}},source:'operator'})],
 'agent-templates':[]
};
const credentials=['review-budget','api-budget'].map((credentialId,i)=>conforming('CredentialResource',{credentialId,label:i?'API analysis':'Source review',llmGateway:{gatewayId:'local-litellm',version:'1',digest},createdAt:at,effectivePolicy:{modelPolicies:[{policyId:'worker',version:'1',digest}],models:['demo-default']}}));
let scheduler=conforming('SchedulerSettings',{maxConcurrentRuns:1,maxConcurrentToolRuns:2,revision:'4',updatedAt:at});
let selection={defaultModel:'demo-default',roles:{}},modelRevision='4';
const modelInfo=()=>conforming('ModelSelectionInfo',{configured:scenario!=='empty',models:scenario==='empty'?[]:['demo-default','demo-alternative'].map(id=>({id,digest,upstreamModel:'test/'+id,contextWindowTokens:id==='demo-default'?131072:65536,maxOutputTokens:8192,policies:{planner:'planner@1',worker:'worker@1',summarizer:'summarizer@1',audit_completion_worker:'completion@1'}})),...(scenario==='empty'?{}:{selection,revision:modelRevision,catalogDigest:digest,updatedAt:at})});
const historyItems=['available','partial','disabled','unavailable','unsupported','pending'].map((status,i)=>conforming('AllocationResourceSummary',{allocationId:'alloc-history-'+(i+1),runId:'run-analysis-'+(i+1),stageExecutionId:'stage-history-'+i,stage:'review',logicalAgent:i%2?'api-tracer':'source-reviewer',outcome:i===1||i===3?'failed':'succeeded',finishedAt:at,collectionPolicy:status==='disabled'?'disabled':status==='unsupported'?'unsupported':'requested',status,...(status==='unavailable'?{reason:'report_missing'}:status==='unsupported'?{reason:'unsupported_platform'}:{}),...(['available','partial'].includes(status)?{resources:{version:1,scope:'runtime_process',status:status==='partial'?'partial':'complete',...(status==='partial'?{reason:'sampling_gap'}:{}),durationSeconds:92,cpuUserSeconds:12,cpuSystemSeconds:2.2,rssStartBytes:120*1024*1024,rssEndBytes:140*1024*1024,rssPeakObservedBytes:184*1024*1024,rssSampleCount:8,maxSampleGapSeconds:status==='partial'?40:15}}:{})}));
const freshness=(timestamp,intervalSeconds=15)=>({status:'ok',observedAt:timestamp,lastAttemptAt:timestamp,intervalSeconds,coverage:{startedAt:new Date(Date.parse(timestamp)-intervalSeconds*1000).toISOString(),endedAt:timestamp,durationSeconds:intervalSeconds,expectedSamples:1,observedSamples:1}});
const performanceSample=(timestamp,cpu=1.25,rss=256)=>conforming('PerformanceSample',{version:1,generation:'performance-demo-1',observedAt:timestamp,http:{freshness:freshness(timestamp),surfaces:[{surface:'public',inFlight:2,counts:Array.from({length:10},(_,i)=>i===0?[0,71,0,0,1,0]:[0,0,0,0,0,0]),duration:{count:72,sumSeconds:7.2,buckets:Array.from({length:14},(_,i)=>i<5?0:72)}},{surface:'private',inFlight:0,counts:Array.from({length:10},()=>[0,0,0,0,0,0]),duration:{count:0,sumSeconds:0,buckets:Array.from({length:14},()=>0)}}]},process:{freshness:freshness(timestamp),cpuCores:cpu,rssBytes:Math.round(rss*1024*1024),heapLiveBytes:96*1024*1024,goroutines:42,gcCycles:7,gcPauseSeconds:.003},pool:{freshness:freshness(timestamp),acquiredConnections:3,idleConnections:2,totalConnections:5,maxConnections:20,acquireCount:5,acquireDurationSeconds:.25,emptyAcquireWaitSeconds:.01,canceledAcquireCount:1},database:{freshness:freshness(timestamp,60),activeConnections:2,lockWaitingConnections:0,longestTransactionSeconds:.4,estimatedDeadTuples:12,vacuumCount:4,autovacuumCount:3},databaseSize:{freshness:freshness(timestamp,300),sizeBytes:2*1024**3}});
const performance=conforming('PerformanceSnapshot',{enabled:scenario!=='empty',generation:'performance-demo-1',observedAt:at,sampleIntervalSeconds:15,databaseIntervalSeconds:60,databaseSizeIntervalSeconds:300,...(scenario==='empty'?{}:{current:performanceSample(at)}),diagnostics:{skippedSamples:0,rejectedSamples:0,skippedMinutes:0,droppedMinutes:0,pendingMinutes:0}});
window.__operationsRequests=[];

const icons=Object.fromEntries(['operations','runs','settings','binding','projects','inbox','checks','issues','reports','library','evals','copy','history','target','docs','open','list'].map(name=>[name,renderToStaticMarkup(<Icon name={name} size={18}/>)]));
window.__operationsData={fixtures,icons,source:'actual public schema / OperationsSnapshot',capturedFixtureAt:at};
class DemoSocket{
 protocol='contractor.events.v1';readyState=0;onopen=null;onmessage=null;onerror=null;onclose=null;
 constructor(){queueMicrotask(()=>{this.readyState=1;this.onopen?.(new Event('open'))})}
 send(value){const frame=JSON.parse(value);if(frame.type==='subscribe')queueMicrotask(()=>{this.onmessage?.({data:JSON.stringify({version:'contractor.events.v1',type:'subscribed',subscriptionId:frame.subscriptionId,stream:frame.stream,cursor:frame.after})});if(scenario==='reconnecting')setTimeout(()=>{this.readyState=3;this.onclose?.({code:1006})},50)})}
 close(code=1000){this.readyState=3;this.onclose?.({code})}
}
const api=new PublicAPI({uiVersion:packageMetadata.version,supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url),path=url.pathname;
 window.__operationsRequests.push({method:request.method,path});
 if(request.method==='PUT'&&path==='/v1/operations/settings/scheduler'){
  if(scenario==='error')return failure(503,'storage_unavailable','Synthetic settings read failure');
  if(scenario==='attention'){scheduler=conforming('SchedulerSettings',{...scheduler,maxConcurrentToolRuns:3,revision:String(Number(scheduler.revision)+1)});return failure(412,'precondition_failed','Synthetic concurrent Scheduler update');}
  const limits=await request.json();scheduler=conforming('SchedulerSettings',{...limits,revision:String(Number(scheduler.revision)+1),updatedAt:at});return json(scheduler,200,{ETag:'"'+scheduler.revision+'"','Cache-Control':'no-store'});
 }
 if(request.method==='PUT'&&path==='/v1/operations/settings/models'){
  if(scenario==='attention'){modelRevision=String(Number(modelRevision)+1);return failure(412,'precondition_failed','Synthetic concurrent model update');}
  selection=await request.json();modelRevision=String(Number(modelRevision)+1);return json(modelInfo(),200,{ETag:'"'+modelRevision+':'+digest.slice(7)+'"','Cache-Control':'no-store'});
 }
 if(request.method!=='GET')return failure(403,'forbidden','Read-only synthetic inventory; settings writes stay in local fixture memory');
 if(path==='/v1/config')return json(config);
 if(path==='/v1/auth/session')return json(conforming('AuthSession',{principal:{userId:'preview',username:'preview',capabilities:['user','operations']},csrfToken:'a'.repeat(43),idleExpiresAt:'2099-01-01T00:00:00Z',absoluteExpiresAt:'2099-01-02T00:00:00Z'}));
 if(path==='/v1/operations/snapshot'&&section==='settings'&&scenario==='error')return json(fixtures.attention.snapshot);
 if(path==='/v1/operations/snapshot')return fixture.snapshot===null?failure(503,'storage_unavailable','Synthetic Operations read failure'):json(fixture.snapshot);
 if(scenario==='error'&&path.startsWith('/v1/operations/'))return failure(503,'storage_unavailable','Synthetic Operations read failure');
 if(path==='/v1/operations/runtime-agent-principals')return json(page(principals));
 if(path==='/v1/operations/runtime-labels')return json(page(scenario==='empty'?[]:bindings));
 if(path==='/v1/operations/runtime-configs')return json(page(scenario==='empty'?[]:runtimeConfigs));
 if(path.startsWith('/v1/operations/runtime-configs/')){const parts=path.split('/');const item=runtimeConfigs.find(c=>c.ref.name===parts[4]);if(item)return json(item);}
 if(path==='/v1/operations/runtime-credentials')return json(page(scenario==='empty'?[]:runtimeCredentials));
 if(path==='/v1/operations/credentials')return json(page(scenario==='empty'?[]:credentials));
 if(path.startsWith('/v1/operations/credentials/')){const item=credentials.find(c=>path.endsWith('/'+c.credentialId));if(item)return json(item);}
 if(path.startsWith('/v1/configurations/')){const parts=path.split('/'),items=configResources[parts[3]]??[];return parts.length===4?json(page(scenario==='empty'?[]:items)):items.find(c=>c.ref.name===parts[4])?json(items.find(c=>c.ref.name===parts[4])):failure(404,'not_found','No synthetic version');}
 if(path==='/v1/operations/settings/scheduler')return json(scheduler,200,{ETag:'"'+scheduler.revision+'"','Cache-Control':'no-store'});
 if(path==='/v1/operations/settings/models')return json(modelInfo(),200,{ETag:'"'+modelRevision+':'+digest.slice(7)+'"','Cache-Control':'no-store'});
 if(path==='/v1/settings/git-key')return json({configured:false});
 if(path==='/v1/operations/allocation-history')return json(page(scenario==='empty'?[]:historyItems.filter(r=>!url.searchParams.get('runId')||r.runId===url.searchParams.get('runId'))));
 if(path==='/v1/operations/performance')return json(performance);
 if(path==='/v1/operations/performance/history'){const from=url.searchParams.get('from'),to=url.searchParams.get('to'),step=url.searchParams.get('step');return json(conforming('PerformanceHistory',{from,to,step,points:step==='15s'&&scenario!=='empty'?Array.from({length:60},(_,i)=>({...performanceSample(new Date(Date.parse(to)-(60-i)*15000).toISOString(),[.5,.7,.68,.95,1.1,.75,.7,.95,1.18,1.02,.8,.9,1.05,.93,1.02,1.25][Math.floor(i/4)%16],236+i/59*20),kind:'sample'})):[]}));}
 return emptyPageOrNotFound(request,url);
});
const events=new RunEventsManager(location.origin,{WebSocketImplementation:DemoSocket});
createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} runEvents={events} router={createMemoryRouter(applicationRoutes(),{initialEntries:[paths[section]??paths.overview]})}/>);
