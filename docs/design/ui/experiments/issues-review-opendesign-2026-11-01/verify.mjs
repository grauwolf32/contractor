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
const fixtureData=JSON.parse(await readFile(new URL('./examples.json',import.meta.url),'utf8'));
const examples=fixtureData.examples;
assert.equal(fixtureData.synthetic,true);assert.equal(examples.length,6);
const source=await readFile(new URL('./preview.html',import.meta.url),'utf8');
assert.deepEqual(JSON.parse(source.match(/<script id="findingExamples" type="application\/json">([\s\S]*?)<\/script>/)[1]),fixtureData,'Preview embeds the exact fixture source');
const {dereference,validate}=require('@cfworker/json-schema');
const schemas=JSON.parse(await readFile(new URL('../../../../../ui/src/api/generated/public-schemas.json',import.meta.url),'utf8'));
const lookup=dereference(schemas);
for(const example of examples){
  const result=validate(example.proposal,{$ref:schemas.$id+'#/$defs/FindingProposalDocument'},'2020-12',lookup,true);
  assert.equal(result.valid,true,example.id+' proposal contract: '+JSON.stringify(result.errors));
  assert.deepEqual(example.proposal.evidence_ids,example.files.map(file=>file.id));
}
assert.equal(createHash('sha256').update(await readFile(new URL('./'+review.acceptedComposition,import.meta.url))).digest('hex'),review.acceptedCompositionSha256,'Accepted composition snapshot is preserved');
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
    assert.equal(await frame.locator('#nextBtn').isDisabled(),false);
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
    await frame.getByRole('button',{name:'Mark as duplicate',exact:true}).click();
    assert.equal(await frame.locator('#recordBtn').isDisabled(),true,'Duplicate requires an explicit target');
    await frame.locator('[data-dup]').first().click();
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
    const exampleResults=[];
    for(const example of examples){
      if(width<900)await frame.locator('#backLink').click();
      await frame.locator('#issueList [data-select="'+example.id+'"]').click();
      await frame.getByRole('tab',{name:'Summary',exact:true}).click();
      assert.equal(await frame.locator('#detail').getAttribute('data-finding-id'),example.id);
      assert.equal(await frame.locator('#issueTitle').innerText(),example.proposal.title);
      assert.equal(await frame.locator('#projectContext').innerText(),example.project);
      assert.equal(await frame.locator('#checkContext').innerText(),'Check: '+example.check);
      assert.deepEqual(await frame.locator('.statusline').evaluate(el=>Array.from(el.children).map(child=>child.id)),['statePill','verifyPill','sevSuggest'],'Verification precedes AI/analyst severity');
      const summary=await frame.locator('#tabSummary').innerText();
      assert.ok(summary.includes(example.proposal.description));
      for(const limitation of example.proposal.limitations)assert.ok(summary.includes(limitation));
      for(const other of examples.filter(item=>item!==example))assert.ok(!summary.includes(other.proposal.description),'No narrative leaks between examples');
      if(example.proposal.subject===null)assert.match(await frame.locator('#subjectLine').innerText(),/Subject not identified/);
      else assert.ok((await frame.locator('#subjectLine').innerText()).includes(example.proposal.subject.key.replace(/^\S+ /,example.proposal.subject.kind==='operation'?'':'')));
      assert.match(await frame.locator('#verifyPill').innerText(),new RegExp(({supported:'Supported',inconclusive:'Inconclusive',blocked:'Blocked','not-tested':'Not tested'})[example.assessment]));
      assert.equal(await frame.locator('#decidedSummary').isVisible(),!!example.analyst);
      assert.equal(await frame.locator('[data-verdict=reopen]').isVisible(),example.actions.includes('reopen'));
      if(example.analyst){
        const recorded=await frame.locator('#decidedSummary').evaluate(el=>{const box=el.getBoundingClientRect(),main=document.querySelector('#tabSummary .col-main').getBoundingClientRect();return {height:box.height,left:box.left,mainLeft:main.left,font:parseFloat(getComputedStyle(el.querySelector('.recorded-reason')).fontSize)}});
        assert.ok(Math.abs(recorded.left-recorded.mainLeft)<=1,'Recorded decision aligns with the content gutter');assert.ok(recorded.font<=14);
        if(width===1440)assert.ok(recorded.height<=120,'Recorded review is compact at desktop width');
      }
      if(example.analyst){assert.match(await frame.locator('#sevSuggest').innerText(),/Analyst severity/);assert.ok((await frame.locator('#decidedSummary').innerText()).includes(example.analyst.reason))}
      else assert.match(await frame.locator('#sevSuggest').innerText(),/^AI /);
      if(example.proposal.http_exchange){
        assert.equal(await frame.locator('#tabSummary .exchange').isVisible(),true);
        assert.ok(!(await frame.locator('#tabSummary').innerText()).includes('Bearer synthetic-demo-token'));
        await frame.locator('#tabSummary').getByRole('button',{name:'Show Authorization value',exact:true}).click();
        assert.ok((await frame.locator('#tabSummary').innerText()).includes('Bearer synthetic-demo-token'));
        assert.ok(!(await frame.locator('#tabSummary').innerText()).includes('demo-session=synthetic-only'));
        await frame.locator('#tabSummary').getByRole('button',{name:'Hide Authorization value',exact:true}).click();
      }else assert.match(summary,/No HTTP exchanges were captured/);
      for(const file of example.files){
        assert.ok(summary.includes(file.ref));
        const preview=frame.locator('#tabSummary [data-preview="'+file.id+'"]');
        await preview.click();assert.equal(await frame.locator('#tabSummary [data-file-preview="'+file.id+'"]').isVisible(),true);
        assert.equal(await frame.locator('#tabSummary [data-file-preview="'+file.id+'"] pre').textContent(),file.text);
        await preview.click();
        if(theme==='light'&&width===1440){
          const pending=page.waitForEvent('download');
          await frame.locator('#tabSummary [data-download="'+file.id+'"]').click();
          const download=await pending;assert.equal(await readFile(await download.path(),'utf8'),file.text,'Download uses the selected evidence bytes');
        }
      }
      await frame.getByRole('tab',{name:'Evidence',exact:true}).click();
      for(const file of example.files)assert.ok((await frame.locator('#tabEvidence').innerText()).includes(file.ref));
      if(!example.files.length)assert.match(await frame.locator('#tabEvidence').innerText(),/No locations or retained evidence/);
      await frame.getByRole('tab',{name:'History',exact:true}).click();
      assert.ok((await frame.locator('#tabHistory').innerText()).includes(example.analyst?example.analyst.reason:'No decision recorded yet.'));
      await frame.getByRole('tab',{name:'Summary',exact:true}).click();
      const widths=await frame.locator('html').evaluate(el=>({viewport:innerWidth,page:el.scrollWidth,duplicates:Array.from(document.querySelectorAll('[id]')).map(el=>el.id).filter((id,index,ids)=>ids.indexOf(id)!==index)}));
      assert.equal(widths.page,width);assert.deepEqual(widths.duplicates,[]);
      if(width===1440&&example.actions.includes('duplicate')){
        const positions=await frame.locator('#decision').evaluate(el=>{const a=el.querySelector('[data-verdict=needsevidence]').getBoundingClientRect(),b=el.querySelector('[data-verdict=duplicate]').getBoundingClientRect();return {gap:b.left-a.right,dy:Math.abs(a.top-b.top),icons:el.querySelectorAll('[data-verdict=duplicate] svg').length}});
        assert.ok(positions.gap>=0&&positions.gap<=12&&positions.dy<=1,'Duplicate is adjacent to the main decisions');assert.equal(positions.icons,1);
      }
      if(example.actions.includes('duplicate')){
        await frame.getByRole('button',{name:'Mark as duplicate',exact:true}).click();
        await frame.locator('#whyInput').fill('Synthetic target eligibility check.');
        const eligible=examples.filter(target=>target.id!==example.id&&target.project===example.project&&target.check===example.check);
        assert.deepEqual(await frame.locator('[data-dup]').evaluateAll(buttons=>buttons.map(b=>b.dataset.dup)),eligible.map(target=>target.id));
        assert.equal(await frame.locator('#recordBtn').isDisabled(),true,'Target selection remains mandatory');
        if(eligible.length){await frame.locator('[data-dup]').first().click();assert.equal(await frame.locator('#recordBtn').isDisabled(),false)}
      }else{
        await frame.getByRole('button',{name:'Reopen',exact:true}).click();
        await frame.locator('#whyInput').fill('Synthetic local reopen draft.');
        await frame.locator('#recordBtn').click();
        assert.match(await frame.locator('#statePill').innerText(),/Confirmed/);
        assert.ok((await frame.locator('#decidedSummary').innerText()).includes(example.analyst.reason));
      }
      if(example.actions.includes('duplicate'))await frame.getByRole('button',{name:'Needs evidence',exact:true}).click();
      await page.screenshot({path:output+`example-${example.id}-${theme}-${width}.png`,fullPage:true});
      assert.equal(await frame.locator('html').evaluate(el=>el.scrollWidth),width,'Draft remains contained for every example');
      exampleResults.push({id:example.id,passed:true});
    }
    assert.deepEqual(failures,[]);assert.deepEqual(external,[]);
    results.push({theme,width,...metrics,layout,examples:exampleResults,passed:true});await context.close();
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
  await comparison.goto(url+'?version=accepted&theme=dark&width=1440');
  assert.equal(await comparison.frameLocator('#design').getByRole('button',{name:'Duplicate…',exact:true}).isVisible(),true);
  await comparison.goto(url+'?theme=dark&width=1440&example=issue-debug-exposure');
  assert.match(await comparison.frameLocator('#design').locator('#issueTitle').innerText(),/Debug information/);
  await comparison.close();
  console.log(`${results.length} layouts / ${results.reduce((total,result)=>total+result.examples.length,0)} finding examples passed: distinct composition, common fonts, themes, widths, truthful states, tabs, explicit preview, required rationale/severity/duplicate target, local-only recording, mobile navigation, prior-version comparison and isolated GET-only preview.`);
}finally{
  await browser?.close();child.kill('SIGTERM');await new Promise(resolve=>{if(child.exitCode!==null||child.signalCode!==null)resolve();else child.once('exit',resolve)});
}
