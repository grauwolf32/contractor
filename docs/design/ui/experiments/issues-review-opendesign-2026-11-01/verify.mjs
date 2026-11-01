import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const require = createRequire(new URL('../../../../../ui/package.json',import.meta.url));
const { chromium } = require('@playwright/test');
const output = fileURLToPath(new URL('../../../../../ui/test-results/issues-review-opendesign/',import.meta.url));
await mkdir(output,{recursive:true});
const review=JSON.parse(await readFile(new URL('./review.json',import.meta.url),'utf8'));
for(const [file,key] of [['original.html','originalSha256'],['preview.html','previewSha256']]){
  assert.equal(createHash('sha256').update(await readFile(new URL(file,import.meta.url))).digest('hex'),review[key],file+' provenance hash');
}
assert.equal(review.generationOutcome,'completed');
assert.notEqual(review.originalSha256,createHash('sha256').update(await readFile(new URL(review.previous,import.meta.url))).digest('hex'),'New source is a separate generated composition');
const child=spawn(process.execPath,[fileURLToPath(new URL('./serve.mjs',import.meta.url)),'0'],{stdio:['ignore','pipe','pipe']});
const url=await new Promise((resolve,reject)=>{let pending='';child.stdout.on('data',d=>{pending+=d;const n=pending.indexOf('\n');if(n>=0){try{resolve(JSON.parse(pending.slice(0,n)).url)}catch(e){reject(e)}}});child.once('error',reject);child.once('exit',code=>reject(new Error('Preview server exited '+code)));});
let browser;
const results=[];
try{
  browser=await chromium.launch({headless:true});
  for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320]){
    const context=await browser.newContext({viewport:{width:Math.max(430,width+32),height:1100}});
    const page=await context.newPage();const failures=[];const external=[];
    page.on('pageerror',e=>failures.push(e.message));
    await page.route('**/*',route=>{const target=route.request().url();if(target.startsWith(url)||target.startsWith('data:')||target.startsWith('blob:'))return route.continue();external.push(target);return route.abort();});
    await page.goto(url+'?theme='+theme+'&width='+width);
    const frame=page.frameLocator('#design');
    await frame.locator('#issueTitle').waitFor({state:'visible'});
    assert.equal(await frame.locator('html').getAttribute('data-theme'),theme);
    const metrics=await frame.locator('html').evaluate(el=>({width:innerWidth,documentWidth:el.scrollWidth,font:getComputedStyle(document.body).fontFamily}));
    assert.equal(metrics.width,width);assert.equal(metrics.documentWidth,width);assert.match(metrics.font,/system-ui/);assert.doesNotMatch(metrics.font,/Manrope|JetBrains/);
    const layout=await frame.locator('#tabSummary').evaluate(panel=>{
      const rect=el=>{const b=el.getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,width:b.width,height:b.height}};
      const lede=panel.querySelector('.lede'),workbench=panel.querySelector('.workbench'),side=panel.querySelector('.col-side');
      const style=el=>({border:getComputedStyle(el).borderTopWidth,background:getComputedStyle(el).backgroundColor});
      return {main:rect(panel.querySelector('.col-main')),side:rect(side),lede:style(lede),workbench:style(workbench),detail:rect(panel.closest('.detail'))};
    });
    if(width>900){
      assert.ok(layout.main.width>=380,'Main narrative/evidence column must remain readable at medium desktop widths');
      assert.ok(layout.side.right<=layout.detail.right+1);
      if(width===1440)assert.ok(layout.side.left>layout.main.left+layout.main.width,'Wide desktop retains a supporting column');
    }else{
      assert.ok(layout.main.width>=width*.8,'Mobile narrative fills its single column');
      assert.ok(Math.abs(layout.main.left-layout.side.left)<=1,'Supporting context stacks on mobile');
    }
    assert.equal(layout.lede.border,'0px','Narrative is unboxed');
    assert.notDeepEqual(layout.lede,layout.workbench,'Narrative and evidence must have distinct visual roles');
    assert.equal(await frame.locator('#draft').isVisible(),false,'Decision draft starts compact');
    assert.match(await frame.locator('#statePill').innerText(),/Proposed/);
    assert.match(await frame.locator('#sevSuggest').innerText(),/AI suggests High/);
    assert.match(await frame.locator('#verifyPill').innerText(),/Inconclusive/);
    assert.equal(await frame.locator('#nextBtn').isDisabled(),true);
    assert.equal(await frame.locator('[data-verdict="reopen"]').isVisible(),false);
    assert.equal(await frame.locator('#recordBtn').isDisabled(),true);
    await page.screenshot({path:output+`issue-${theme}-${width}.png`,fullPage:true});
    await frame.getByRole('tab',{name:'Evidence',exact:true}).click();
    assert.equal(await frame.locator('#tabEvidence').isVisible(),true);
    assert.equal(await frame.locator('#tabSummary').isVisible(),false);
    assert.match(await frame.locator('#tabEvidence').innerText(),/No HTTP evidence captured|No HTTP exchange/i);
    const preview=frame.locator('#tabEvidence [data-preview]');
    await preview.click();assert.equal(await frame.locator('#tabEvidence .preview').first().isVisible(),true);
    await preview.click();assert.equal(await frame.locator('#tabEvidence .preview').first().isVisible(),false);
    await frame.getByRole('tab',{name:'History',exact:true}).click();
    assert.match(await frame.locator('#tabHistory').innerText(),/No decision recorded yet/);
    await frame.getByRole('tab',{name:'Summary',exact:true}).click();
    await frame.getByRole('button',{name:'Confirm issue',exact:true}).click();
    await frame.locator('#whyInput').fill('Synthetic design review only.');
    assert.equal(await frame.locator('#recordBtn').isDisabled(),true);
    await frame.getByRole('button',{name:'High',exact:true}).click();
    assert.equal(await frame.locator('#recordBtn').isDisabled(),false);
    await frame.locator('#recordBtn').click();
    assert.match(await frame.locator('#demoNote').innerText(),/Demo only — no decision was sent/);
    assert.match(await frame.locator('#statePill').innerText(),/Proposed/);
    await frame.getByRole('button',{name:'Duplicate…',exact:true}).click();
    assert.equal(await frame.locator('#recordBtn').isDisabled(),true,'Duplicate requires an explicit target');
    await frame.locator('[data-dup]').click();
    assert.equal(await frame.locator('#recordBtn').isDisabled(),false);
    await frame.locator('#whyInput').fill('   ');
    assert.equal(await frame.locator('#recordBtn').isDisabled(),true,'Whitespace is not a rationale');
    await frame.locator('#whyInput').fill('Local synthetic duplicate choice.');
    if(width<900){
      await frame.locator('#backLink').click();
      assert.equal(await frame.locator('.listpane').isVisible(),true);
      await frame.locator('#issueList').getByRole('option',{name:/Order lookup/}).click();
      assert.equal(await frame.locator('.detail').isVisible(),true);
    }
    const finalWidth=await frame.locator('html').evaluate(el=>el.scrollWidth);
    assert.equal(finalWidth,width,'No horizontal overflow after opening the draft and previews');
    assert.deepEqual(failures,[]);assert.deepEqual(external,[]);
    results.push({theme,width,...metrics,layout,passed:true});await context.close();
  }
  await writeFile(output+'report.json',JSON.stringify({cases:results,passed:results.length},null,2)+'\n');
  const response=await fetch(url+'prototype.html');
  assert.match(response.headers.get('content-security-policy'),/connect-src 'none'/);
  assert.match(response.headers.get('content-security-policy'),/form-action 'none'/);
  assert.match(response.headers.get('content-security-policy'),/sandbox allow-scripts allow-downloads/);
  assert.equal((await fetch(url+'prototype.html',{method:'POST'})).status,405);
  assert.equal((await fetch(url+'original.html')).status,404);
  const comparison=await browser.newPage({viewport:{width:1480,height:1100}});
  await comparison.goto(url+'?version=previous&theme=dark&width=1440');
  assert.match(await comparison.frameLocator('#design').locator('#issueTitle').innerText(),/Order lookup/);
  await comparison.close();
  console.log(`${results.length} Issue review cases passed: distinct composition, common fonts, themes, widths, truthful states, tabs, explicit preview, required rationale/severity/duplicate target, local-only recording, mobile navigation, prior-version comparison and isolated GET-only preview.`);
}finally{
  await browser?.close();child.kill('SIGTERM');await new Promise(resolve=>{if(child.exitCode!==null||child.signalCode!==null)resolve();else child.once('exit',resolve)});
}
