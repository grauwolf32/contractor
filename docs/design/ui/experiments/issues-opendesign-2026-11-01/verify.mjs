import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const require = createRequire(new URL('../../../../../ui/package.json',import.meta.url));
const { chromium } = require('@playwright/test');
const output = fileURLToPath(new URL('../../../../../ui/test-results/issues-opendesign/',import.meta.url));
await mkdir(output,{recursive:true});
const child=spawn(process.execPath,[fileURLToPath(new URL('./serve.mjs',import.meta.url)),'0'],{stdio:['ignore','pipe','pipe']});
const url=await new Promise((resolve,reject)=>{let pending='';child.stdout.on('data',d=>{pending+=d;const n=pending.indexOf('\n');if(n>=0){try{resolve(JSON.parse(pending.slice(0,n)).url)}catch(e){reject(e)}}});child.once('error',reject);child.once('exit',code=>reject(new Error('Preview server exited '+code)));});
let browser;
const results=[];
try{
  browser=await chromium.launch({headless:true});
  for(const theme of ['light','dark','black'])for(const width of [1440,390,320]){
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
    if(width===1440){
      const grid=await frame.locator('.detail').evaluate(detail=>{
        const box=detail.getBoundingClientRect();
        const cards=Array.from(detail.querySelectorAll('.card')).map(card=>card.getBoundingClientRect()).filter(card=>card.width>0&&card.height>0);
        return {available:box.width,span:Math.max(...cards.map(card=>card.right))-Math.min(...cards.map(card=>card.left)),columns:new Set(cards.map(card=>Math.round(card.left))).size};
      });
      assert.ok(grid.span>=grid.available*.82,'Cards must use the available desktop canvas, not a narrow isolated strip');
      assert.ok(grid.columns>=2,'Wide Issue detail should have a main and a supporting column');
    }
    if(width<900){
      const cards=await frame.locator('#tabSummary').evaluate(panel=>Array.from(panel.querySelectorAll('.card')).map(card=>{const box=card.getBoundingClientRect();return {width:box.width,right:box.right}}));
      assert.ok(cards.every(card=>card.width>width*.65&&card.right<=width+1),'Mobile cards must remain readable in one column');
    }
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
    if(width<900){
      await frame.locator('#backLink').click();
      assert.equal(await frame.locator('.listpane').isVisible(),true);
      await frame.locator('#issueList').getByRole('option',{name:/Order lookup/}).click();
      assert.equal(await frame.locator('.detail').isVisible(),true);
    }
    assert.deepEqual(failures,[]);assert.deepEqual(external,[]);
    results.push({theme,width,...metrics,passed:true});await context.close();
  }
  await writeFile(output+'report.json',JSON.stringify({cases:results,passed:results.length},null,2)+'\n');
  console.log(`${results.length} Issue design cases passed: themes, widths, truthful states, tabs, explicit preview, decision validation, local-only recording and mobile navigation.`);
}finally{
  await browser?.close();child.kill('SIGTERM');await new Promise(resolve=>{if(child.exitCode!==null||child.signalCode!==null)resolve();else child.once('exit',resolve)});
}
