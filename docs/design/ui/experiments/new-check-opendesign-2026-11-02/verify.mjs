import {chromium,expect} from '../../../../../ui/node_modules/@playwright/test/index.mjs';
import {readFile,mkdir,writeFile} from 'node:fs/promises';import {createHash} from 'node:crypto';import vm from 'node:vm';
const base='http://127.0.0.1:17567',root=new URL('./',import.meta.url),capture=new URL('../../../../../ui/test-results/new-check-design/candidate-captures/',import.meta.url);
const sha=s=>createHash('sha256').update(s).digest('hex'),script=html=>{const m=html.match(/<script>([\s\S]*?)<\/script>/);if(!m)throw Error('Missing local controller');new vm.Script(m[1]);return m[1]};
const seed=await readFile(new URL('seed.html',root),'utf8'),original=await readFile(new URL('original.html',root),'utf8'),candidate=await readFile(new URL('preview.html',root),'utf8');
if(sha(script(seed))!==sha(script(original))||sha(script(seed))!==sha(script(candidate)))throw Error('Generation changed the preserved local controls/fixtures');
const polish=await readFile(new URL('opendesign.css',root),'utf8')+await readFile(new URL('review-objective.css',root),'utf8');if(/@import|@font-face|url\s*\(|#[\da-f]{3,8}\b|rgba?\s*\(/i.test(polish))throw Error('Design CSS contains remote assets, fonts or literal colors');
const declared=new Set([...candidate.matchAll(/(--[\w-]+)\s*:/g)].map(m=>m[1]));const unknown=[...candidate.matchAll(/var\((--[\w-]+)/g)].map(m=>m[1]).filter(v=>!declared.has(v));if(unknown.length)throw Error('Undefined tokens: '+[...new Set(unknown)]);
await mkdir(capture,{recursive:true});const browser=await chromium.launch(),errors=[],outbound=[],metrics=[];
async function page(width=1440,scenario='ready',theme='dark'){
 const p=await browser.newPage({viewport:{width,height:1000}});p.setDefaultTimeout(10000);p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(new URL(r.url()).origin!==base)outbound.push(r.url());if(r.method()!=='GET')outbound.push(r.method()+' '+r.url())});await p.goto(`${base}/prototype.html?case=${scenario}&theme=${theme}`);await expect(p.getByRole('heading',{name:'Start a check',exact:true})).toBeVisible();return p;
}
async function choose(p,name){const browser=p.locator('#types-browser');if(await browser.getAttribute('open')===null)await browser.locator('summary').click();await p.getByRole('button',{name:new RegExp(name)}).filter({has:p.locator('[class="type-copy"]')}).click()}
try{
 for(const scenario of ['ready','missing','unsupported'])for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320]){
  const p=await page(width,scenario,theme),start=p.getByRole('button',{name:'Start check',exact:true});
  if(scenario==='ready')await expect(start).toBeEnabled();else await expect(start).toBeDisabled();
  await expect(p.locator('#readiness')).toHaveAttribute('data-tone',scenario==='ready'?'ready':scenario==='missing'?'missing':'blocked');
  const objective=await p.evaluate(()=>{
   const rect=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
   return {card:rect('.objective'),workbench:rect('.workbench'),label:rect('#objective-label'),input:rect('#objective'),hint:rect('#suggestion')};
  });
  expect(Math.abs(objective.card.left-objective.workbench.left)).toBeLessThanOrEqual(1);
  expect(Math.abs(objective.card.right-objective.workbench.right)).toBeLessThanOrEqual(1);
  for(const child of [objective.label,objective.input,objective.hint]){
   expect(child.left).toBeGreaterThanOrEqual(objective.card.left);
   expect(child.right).toBeLessThanOrEqual(objective.card.right);
  }
  expect(objective.label.bottom).toBeLessThanOrEqual(objective.input.top);
  expect(objective.input.bottom).toBeLessThanOrEqual(objective.hint.top);
  expect(objective.hint.bottom).toBeLessThanOrEqual(objective.card.bottom);
  await start.scrollIntoViewIfNeeded();const measured=await p.evaluate(()=>({width:innerWidth,documentWidth:document.documentElement.scrollWidth,button:document.getElementById('start').getBoundingClientRect().toJSON(),font:getComputedStyle(document.body).fontFamily}));
  expect(measured.documentWidth).toBeLessThanOrEqual(width);expect(measured.button.y).toBeGreaterThanOrEqual(-1);expect(measured.button.bottom).toBeLessThanOrEqual(1001);
  metrics.push({scenario,theme,...measured});await p.evaluate(()=>scrollTo(0,0));
  if(width===1440||width===390)await p.screenshot({path:new URL(`${scenario}-${theme}-${width}.png`,capture).pathname,fullPage:false});await p.close();
 }
 const p=await page();await p.getByRole('button',{name:'Start check',exact:true}).click();await expect(p.locator('#action-status')).toHaveText('Preview only — no check was started.');await p.getByRole('button',{name:'Save as draft',exact:true}).click();await expect(p.locator('#action-status')).toHaveText('Preview only — no draft was created.');
 await p.locator('#slot-source').selectOption('');await expect(p.locator('#start')).toBeDisabled();await expect(p.locator('#readiness')).toHaveText('1 material missing');await p.locator('#slot-source').selectOption({index:1});await expect(p.locator('#start')).toBeEnabled();
 await choose(p,'OWASP ASVS');await p.getByRole('radio',{name:'Pilot · 5 requirements'}).check();await expect(p.locator('#summary-scope')).toHaveText('5 requirements');await p.locator('#advanced summary').click();await expect(p.locator('#exact-type')).toHaveText('owasp-asvs-5-0-l1-source-pilot@1');await p.getByRole('radio',{name:'Full review · 70 requirements'}).check();await expect(p.locator('#exact-type')).toHaveText('owasp-asvs-5-0-l1-source-review@1');
 await p.getByRole('button',{name:'Custom',exact:true}).click();await p.locator('#custom-hours').fill('0');await expect(p.locator('#start')).toBeDisabled();await p.locator('#custom-hours').fill('72');await expect(p.locator('#start')).toBeEnabled();await expect(p.locator('#summary-time')).toHaveText('72 hours');await p.getByRole('button',{name:'No time limit',exact:true}).click();await expect(p.locator('#summary-time')).toHaveText('No time limit');
 await p.locator('#type-search').fill('no-such-check');await expect(p.getByText('No matching check types.',{exact:true})).toBeVisible();await p.locator('#type-search').fill('');await choose(p,'Custom checklist');await expect(p.locator('#slot-checklist')).toHaveValue('docs/security-checklist@checklist-r1');await expect(p.locator('#start')).toBeEnabled();
 await p.setViewportSize({width:390,height:1000});await expect(p.locator('#types-browser')).not.toHaveAttribute('open');await p.locator('#types-browser summary').click();await expect(p.locator('#type-search')).toBeVisible();await choose(p,'API endpoint trace');await expect(p.locator('#types-browser')).not.toHaveAttribute('open');await p.setViewportSize({width:1440,height:1000});await expect(p.locator('#types-browser')).toHaveAttribute('open');await p.close();
 const missing=await page(390,'missing');await missing.getByRole('button',{name:'Add API spec',exact:true}).click();await expect(missing.locator('#start')).toBeEnabled();await expect(missing.locator('#notice')).toContainText('No upload was sent.');await missing.close();
 const keyboard=await page(390);await keyboard.locator('#types-browser summary').focus();await keyboard.keyboard.press('Enter');await expect(keyboard.locator('#type-search')).toBeVisible();await keyboard.locator('#type-search').focus();await keyboard.keyboard.type('ASVS');await expect(keyboard.locator('#type-list .type-row')).toHaveCount(1);await keyboard.close();
 const parent=await browser.newPage({viewport:{width:850,height:1000}});await parent.goto(base+'/?version=new&case=ready&theme=dark&width=1440');await parent.frameLocator('#design').getByRole('button',{name:'Start check',exact:true}).waitFor();await parent.locator('#width').selectOption('390');await expect(parent.frameLocator('#design').locator('#types-browser')).not.toHaveAttribute('open');await parent.locator('#width').selectOption('1440');await expect(parent.frameLocator('#design').locator('#types-browser')).toHaveAttribute('open');await parent.close();
 for(const method of ['POST','PUT','DELETE']){const response=await fetch(base+'/',{method});expect(response.status).toBe(405)}
 expect(outbound).toEqual([]);expect(errors).toEqual([]);await writeFile(new URL('metrics.json',capture),JSON.stringify(metrics,null,2)+'\n');console.log('Passed 36 objective alignment/containment and readiness/theme/width layouts; local type/material/scope/time/draft/start/search, keyboard and live iframe resize; exact original script, shared tokens, no mutations/outbound requests.');
}finally{await browser.close()}
