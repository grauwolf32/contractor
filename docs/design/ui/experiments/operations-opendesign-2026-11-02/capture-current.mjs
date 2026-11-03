import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {readFile,writeFile,mkdir} from 'node:fs/promises';import {gzipSync} from 'node:zlib';
const root=new URL('./',import.meta.url),repo=new URL('../../../../../',import.meta.url),target=new URL('ui/test-results/operations-design/',repo);
try{await readFile(new URL('provenance.json',root));throw Error('These native reference inputs are frozen; create a new experiment/revision')}catch(error){if(error.code!=='ENOENT')throw error}
await mkdir(target,{recursive:true});
const source=(await readFile(new URL('current-fixture.tsx',root),'utf8')).replaceAll('/Users/ruslan/.codex/worktrees/check-ui-hierarchy/contractor',repo.pathname.replace(/\/$/,''));
await writeFile(new URL('current.tsx',target),source);
await writeFile(new URL('current.html',target),'<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Operations · actual UI reference</title><div id="root"></div><script type="module" src="./current.tsx?rev=3"></script></html>');
const browser=await chromium.launch(),bodies={},styles={},errors=[];
try{
 for(const scenario of ['normal','attention','empty','reconnecting','error'])for(const width of [1440,390]){
  const page=await browser.newPage({viewport:{width,height:1000}});page.on('pageerror',error=>errors.push(error.message));
  await page.goto(`http://127.0.0.1:17560/test-results/operations-design/current.html?case=${scenario}&theme=dark`);
  await expect(page.getByRole('heading',{name:'Operations',exact:true})).toBeVisible();
  await expect(scenario==='error'?page.getByText('Public API is unavailable',{exact:true}):page.getByRole('heading',{name:'Execution readiness',exact:true})).toBeVisible();
  const data=await page.evaluate(()=>window.__operationsData);if(!data)throw Error('Schema-validated fixture inventory missing');
  if(scenario==='normal'&&width===1440)await writeFile(new URL('fixtures.json',root),JSON.stringify(data,null,2)+'\n');
  const snapshot=await page.evaluate(()=>{
   const clone=document.body.cloneNode(true);clone.querySelectorAll('script').forEach(el=>el.remove());
   return {body:clone.innerHTML,css:[...document.querySelectorAll('style')].map(style=>style.textContent).join('\n')};
  });
  bodies[`${scenario}-${width}`]=snapshot.body;styles[`${scenario}-${width}`]=snapshot.css;
  await page.screenshot({path:new URL(`current-${scenario}-${width}.png`,root).pathname,fullPage:true});await page.close();
 }
 if(errors.length)throw Error(errors.join('\n'));
 await writeFile(new URL('baseline.json.gz',root),gzipSync(JSON.stringify(bodies)));await writeFile(new URL('baseline.css.gz',root),gzipSync(JSON.stringify(styles)));
 console.log('Captured ten real Operations React references; every populated snapshot validates against the public schema.');
}finally{await browser.close()}
