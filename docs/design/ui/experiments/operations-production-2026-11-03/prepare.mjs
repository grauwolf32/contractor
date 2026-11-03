import {createHash} from 'node:crypto';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',root),target=new URL('ui/test-results/operations-production/',repo);await mkdir(target,{recursive:true});
const source=(await readFile(new URL('current-fixture.tsx',root),'utf8')).replaceAll('/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor',repo.pathname.replace(/\/$/,''));await writeFile(new URL('current.tsx',target),source);const revision=createHash('sha256').update(source).digest('hex').slice(0,12);
await writeFile(new URL('current.html',target),'<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Operations · actual UI preview</title><div id="root"></div><script type="module" src="./current.tsx?rev='+revision+'"></script></html>');
const [server]=await Promise.all([readFile(new URL('../operations-settings-opendesign-2026-11-03/serve.mjs',root),'utf8')]);
const match=server.match(/const wrapper=`([\s\S]*?)`;\nconst metrics=/);if(!match)throw Error('Reviewed wrapper unavailable');
let wrapper=match[1].replace('Contractor · Operations · OpenDesign review','Contractor · Operations · working UI review').replace('Operations · OpenDesign · synthetic read-only preview','Operations · working React UI · synthetic API').replace('<option value="new">OpenDesign</option>','<option value="new">Working UI</option>').replace('<option value="previous">Previous Settings</option>','<option value="approved">Approved design</option>').replace('<option value="seed">Functional seed</option>','').replace('Draft conflict','Needs inspection / write conflict').replace("value='settings'","value='overview'");
wrapper=wrapper.replace("frame.src='/'+(v==='new'?'prototype':v)+'.html?'+q;", "frame.src=v==='approved'?'http://127.0.0.1:17570/prototype.html?'+q:'./current.html?'+q;");
// Same-origin is needed for Vite module imports in the actual Application;
// the synthetic transport intercepts every API call before any network request.
wrapper=wrapper.replace('sandbox="allow-scripts allow-forms"','sandbox="allow-scripts allow-same-origin allow-forms"');
await writeFile(new URL('index.html',target),wrapper);console.log('Prepared actual React Operations preview at /test-results/operations-production/index.html');
