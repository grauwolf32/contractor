// Read-only synthetic transport around the actual production Application/route.
// Copied by verify-production.mjs into Vite's ignored test-results directory.
import { createRoot } from 'react-dom/client';
import { createMemoryRouter } from 'react-router';
import { Application } from '/src/app/application';
import { applicationRoutes } from '/src/app/router';
import { PublicAPI } from '/src/api/client';
import * as build from '/src/test/builders';
import { json, page, failure, emptyPageOrNotFound } from '/src/test/responses';
import {trace,top10,asvsReview,asvsPilot,checklist,profileFixture,projectFixture,materialFixture} from '/src/routes/checks/start/test-fixtures';
import config from '/@fs/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/valid/shared-config.json';
import engagementCases from '/@fs/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor/api/testdata/v1alpha1/engagement-cases.json';
import '/src/app/theme.css'; import '/src/styles.css'; import '/src/app/ui-primitives.css'; import '/src/ui/ui.css';
const params=new URLSearchParams(location.search),scenario=params.get('case')??'ready';
const theme=params.get('theme')??'dark';localStorage.setItem('contractor.theme',theme);document.documentElement.dataset.theme=theme;
const apiSource=`openapi: 3.0.3
info: {title: Shop API, version: '1'}
paths:
  /products:
    get: {responses: {'200': {description: OK}}}
    post: {responses: {'201': {description: Created}}}
  /orders:
    get: {responses: {'200': {description: OK}}}
    post: {responses: {'201': {description: Created}}}
  /orders/{id}:
    get: {responses: {'200': {description: OK}}}
    delete: {responses: {'204': {description: Deleted}}}
`;
const checklistSource=JSON.stringify({schema:'contractor.audit.checklist.v1',items:['Verify tenant isolation.','Verify object ownership.','Reject expired sessions.'].map((statement,i)=>({key:`check-000${i+1}`,version:'1',statement,applicability:'always',allowed_methods:['static'],required_evidence:[],review_policy:'automatic'}))});
const engagement=engagementCases.documents.find(doc=>doc.name==='anonymous-blackbox').text;
const mediaEngagement='application/vnd.contractor.engagement+json';
const profiles=[build.conforming('AuditProfile',{...trace,inputs:{source:{required:true,mediaTypes:['application/zip']},openapi:{required:true,mediaTypes:['application/yaml']},...(scenario==='engagement'?{engagement:{required:true,mediaTypes:[mediaEngagement]}}:{})},...(scenario==='engagement'?{serverCompatible:false,compatibilityReasons:['engagement_capability_unsupported']}: {})}),top10,asvsReview,asvsPilot,build.conforming('AuditProfile',{...checklist,inputs:{source:{required:true,mediaTypes:['application/zip']},checklist:{required:true,mediaTypes:['application/json']}}}),profileFixture('extended-source-review',{serverCompatible:false,compatibilityReasons:['multiple_rounds_unsupported']})];
const payloads=new Map([['shop-openapi',apiSource],['security-checklist',checklistSource],['engagement',engagement]]);
const size=(text)=>new TextEncoder().encode(text).byteLength;
const allMaterials=[materialFixture('shop-source','application/zip',{size:2384971,artifact:{namespace:'sources',name:'shop-source',revision:'source-r3'}}),materialFixture('shop-openapi','application/yaml',{size:size(apiSource),artifact:{namespace:'openapi',name:'shop-openapi',revision:'api-r2'}}),materialFixture('security-checklist','application/json',{size:size(checklistSource),artifact:{namespace:'docs',name:'security-checklist',revision:'checklist-r1'}}),...(scenario==='engagement'?[materialFixture('engagement',mediaEngagement,{size:size(engagement),artifact:{namespace:'scope',name:'engagement',revision:'engagement-r1'}})]:[])];
const materials=scenario==='missing'?allMaterials.filter(m=>m.mediaType!=='application/yaml'):allMaterials;
const project=projectFixture({name:'Shop API',description:'Synthetic design project · source code and API contract'});
const api=new PublicAPI({uiVersion:'0.11.4',supportedApiVersions:['contractor.public.v1'],apiBaseUrl:location.origin},async input=>{
 const request=input instanceof Request?input:new Request(input),url=new URL(request.url),path=url.pathname;
 if(request.method!=='GET')return failure(403,'forbidden','Read-only design preview — no material or check is created');
 if(path==='/v1/auth/session')return json(build.conforming('AuthSession',{principal:{userId:'preview',username:'preview',capabilities:['user']},csrfToken:'a'.repeat(43),idleExpiresAt:'2099-01-01T00:00:00Z',absoluteExpiresAt:'2099-01-02T00:00:00Z'}));
 if(path==='/v1/config')return json(config);
 if(path==='/v1/projects')return json(page([project]));
 if(path==='/v1/projects/project_shop')return json(project,200,{ETag:'"1"'});
 if(path==='/v1/projects/project_shop/artifacts')return scenario==='error'?failure(503,'storage_unavailable','Materials could not be loaded'):json(page(materials));
 const content=/^\/v1\/projects\/project_shop\/artifacts\/([^/]+)\/([^/]+)$/.exec(path);
 if(content){const m=materials.find(m=>m.artifact.name===content[2]&&m.artifact.namespace===content[1]&&m.artifact.revision===url.searchParams.get('revision'));return m&&payloads.has(m.artifact.name)?new Response(payloads.get(m.artifact.name),{headers:{'content-type':m.mediaType,'X-Contractor-API-Version':'contractor.public.v1'}}):failure(404,'not_found');}
 if(path==='/v1/audit-profiles')return json(page(profiles));
 const match=/^\/v1\/audit-profiles\/([^/]+)\/versions\/([^/]+)$/.exec(path);
 if(match){const p=profiles.find(p=>p.ref.name===match[1]&&p.ref.version===match[2]);return p?json(p,200,{ETag:'"'+p.ref.digest+'"'}):failure(404,'not_found');}
 return emptyPageOrNotFound(request,url);
});
const type=scenario==='unsupported'?'extended-source-review':scenario==='source'?'owasp-top10-2025-source-risk':['checklist','editor'].includes(scenario)?'source-checklist':scenario==='asvs'?'owasp-asvs-5-0-l1-source-review':'openapi-operation-trace';
createRoot(document.getElementById('root')!).render(<Application api={api} publicAPI={api} router={createMemoryRouter(applicationRoutes(),{initialEntries:[`/checks/new?project=project_shop&type=${type}&objective=${encodeURIComponent('Review the shop API for broken access control')}`]})}/>);
