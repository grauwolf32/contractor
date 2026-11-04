// Reuse the preserved, user-reviewed Issues examples; no invented source snippets.
import {issueReviewFixtures} from '/src/test/issue-review-fixtures';
import * as build from '/src/test/builders';
import {json,page,failure,bytes} from '/src/test/responses';

export function fullIssueFixtures(server,scenario){
 const examples=issueReviewFixtures(),mapping={finding_idor:'issue-debug-exposure',finding_sqli:'issue-product-search-sql'},sources=[];
 for(const [key,values] of Object.entries(server.findings))server.findings[key]=values.map(original=>{
  const sample=examples.findings.find(f=>f.findingId===mapping[original.findingId]);if(!sample)return original;
  const finding=build.finding(original.projectId,original.findingId,{createdAt:original.createdAt,updatedAt:original.updatedAt,revision:original.revision},sample.claim);
  const source=examples.sources.find(s=>s.findingId===sample.findingId);
  sources.push(build.findingSource(finding,{evidence:source.evidence,projectEvidence:source.projectEvidence},{...source.document,client_key:finding.findingId}));return finding;
 });
 const findings=Object.values(server.findings).flat();
 const evidence=examples.evidence.filter(file=>sources.some(s=>s.projectEvidence.some(e=>JSON.stringify(e.ref)===JSON.stringify(file.metadata.artifact))));
 return {sources,evidence,response(request,url){
  const path=url.pathname;
  if(scenario==='detail-error'&&/^\/v1\/projects\/[^/]+\/findings\/[^/]+$/.test(path))return failure(503,'storage_transaction_conflict','Synthetic issue read failure');
  const project=server.projects.find(p=>path===`/v1/projects/${p.projectId}`);if(project)return json(project,200,{ETag:'"1"'});
  for(const source of sources){
   const prefix=`/v1/projects/${source.projectId}/finding-sources`;
   if(path===prefix+'/'+source.receiptId)return scenario==='source-error'?failure(503,'storage_transaction_conflict','Synthetic retained-source read failure'):json(source);
   if(path===prefix+'/'+source.receiptId+'/history')return json({projectRevision:1,...page([])});
   if(path===prefix)return json({projectRevision:1,...page(sources.filter(s=>s.projectId===source.projectId&&(!url.searchParams.has('findingId')||s.findingId===url.searchParams.get('findingId'))).map(build.findingSourcePreview))});
  }
  const match=path.match(/^\/v1\/projects\/([^/]+)\/findings$/);if(match)return json({projectRevision:1,...page(findings.filter(f=>f.projectId===match[1]).map(build.findingSummary))});
  const file=evidence.find(e=>path===`/v1/projects/${e.projectId}/artifacts/${e.metadata.artifact.namespace}/${e.metadata.artifact.name}`+(path.endsWith('/metadata')?'/metadata':'')&&url.searchParams.get('revision')===e.metadata.artifact.revision);
  if(file){if(scenario==='evidence-error')return failure(404,'not_found','Synthetic retained evidence is unavailable');return path.endsWith('/metadata')?json(file.metadata):bytes(file.text,file.metadata.mediaType);}
  return undefined;
 }};
}
