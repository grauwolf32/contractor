// Verify the isolated review artifact, not a production route or live service.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFile, mkdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import {once} from 'node:events';
import path from 'node:path';

const folder=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(folder,'../../../../..');
const require=createRequire(path.join(root,'ui/package.json'));
const {chromium}=require('@playwright/test');
const read=name=>readFile(path.join(folder,name),'utf8');
const hash=value=>createHash('sha256').update(value).digest('hex');
const examples=JSON.parse(await read('examples.json'));
const preview=await read('preview.html'),original=await read('original.html');
const review=JSON.parse(await read('review.json'));
assert.equal(hash(original),review.sha256.original);
assert.equal(hash(preview),review.sha256.preview);
assert.equal(hash(await read('seed.html')),review.sha256.seed);
assert.notEqual(preview,await read('seed.html'),'The OpenDesign candidate must differ from the seed');
assert.equal(review.model,'openrouter/z-ai/glm-5.3-flash@cc78cc13596ab7d7/worker-model');
assert.equal(review.preservation.postBackup.sqliteIntegrity,'ok');
assert.equal(review.relay.stopped,true);
assert.equal(review.relay.upstreamCredentialRemoved,true);
for(const html of [preview,original]){
  assert.doesNotMatch(html,/<script[^>]+src\s*=|@import|<link[^>]+(?:href|rel)\s*=|\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon/iu);
  const embedded=html.match(/<script id="materialExamples" type="application\/json">(.*?)<\/script>/s);
  assert.ok(embedded,'Preserved synthetic fixtures');
  assert.deepEqual(JSON.parse(embedded[1]),examples);
}

// Reuse the production contract owner, including the no-response variant.
const {createServer}=await import(path.join(root,'ui/node_modules/vite/dist/node/index.js'));
const vite=await createServer({root:path.join(root,'ui'),server:{middlewareMode:true},appType:'custom'});
try{
  const {parseHTTPExchange}=await vite.ssrLoadModule('/src/api/http-exchange.ts');
  for(const e of examples.examples){
    assert.equal(Buffer.byteLength(e.source),e.sizeBytes);
    if(e.id==='http'){
      assert.ok(parseHTTPExchange(e.source),'Valid saved exchange');
      const absent=JSON.parse(e.source),a=absent.capture.attempts[0];
      delete a.response;a.error='response_unavailable';
      assert.ok(parseHTTPExchange(JSON.stringify(absent)),'Valid exchange without response');
    }
  }
}finally{await vite.close()}

const child=spawn(process.execPath,[path.join(folder,'serve.mjs'),'0'],{stdio:['ignore','pipe','inherit']});
let browser;
try{
  const [chunk]=await once(child.stdout,'data');
  const base=JSON.parse(chunk.toString().trim()).url;
  const response=await fetch(base+'prototype.html');
  assert.match(response.headers.get('content-security-policy'),/connect-src 'none'/);
  assert.match(response.headers.get('content-security-policy'),/sandbox allow-scripts allow-downloads/);
  assert.equal((await fetch(base+'private.txt')).status,404);
  assert.equal((await fetch(base,{method:'POST'})).status,405);
  browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1480,height:1100},acceptDownloads:true});
  const errors=[],external=[];
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>{if(!request.url().startsWith(base)&&!request.url().startsWith('blob:'))external.push(request.url())});
  const screenshots=path.join(root,'ui/test-results/material-detail-opendesign');
  await mkdir(screenshots,{recursive:true});
  let layouts=0;
  async function open(file='source',state='normal',theme='dark',width=390,version='new'){
    await page.goto(base+'?'+new URLSearchParams({case:file,state,theme,width:String(width),version}));
    const frame=page.frameLocator('#design');
    await frame.locator('#viewer').waitFor();
    return frame;
  }
  async function layout(frame,width){
    const dimensions=await frame.locator('html').evaluate(n=>({width:innerWidth,doc:n.scrollWidth,font:getComputedStyle(document.body).fontFamily,reader:document.querySelector('.reader').getBoundingClientRect().toJSON(),side:document.querySelector('.side').getBoundingClientRect().toJSON(),header:document.querySelector('.header').getBoundingClientRect().toJSON()}));
    assert.equal(dimensions.width,width);
    assert.equal(dimensions.doc,width,'No horizontal document overflow');
    assert.match(dimensions.font,/ui-sans-serif.*system-ui/);
    assert.ok(dimensions.reader.width>=width*(width<821?.85:.5),'The content owns the main reading area');
    if(width<821){
      assert.ok(dimensions.reader.y<390,'Phone chrome leaves content visible');
      assert.ok(dimensions.side.y>=dimensions.reader.bottom,'Phone metadata follows content');
      assert.ok(dimensions.header.height<300,'Compact phone header');
    }
    const actions=await frame.locator('.actions button').evaluateAll(nodes=>nodes.map(n=>({rect:n.getBoundingClientRect().toJSON(),name:n.getAttribute('aria-label')||n.textContent})));
    for(const a of actions){assert.ok(a.name.trim());assert.ok(a.rect.x>=0&&a.rect.right<=width);assert.ok(a.rect.width>=30&&a.rect.height>=30)}
    layouts++;
  }
  for(const theme of ['light','dark','black'])for(const width of [1440,1100,390,320])for(const example of examples.examples){
    const frame=await open(example.id,'normal',theme,width);
    await layout(frame,width);
    assert.equal(await frame.locator('#materialTitle').textContent(),example.name);
    assert.equal(await frame.locator('html').getAttribute('data-theme'),theme);
    assert.match(await frame.locator('#badges').innerText(),/Current revision/);
    assert.equal(await frame.locator('#newVersion').isVisible(),example.id==='report');
    assert.equal(await frame.locator('#viewer').locator('script,img,iframe').count(),0);
    await page.screenshot({path:path.join(screenshots,`${example.id}-${theme}-${width}.png`),fullPage:true});
    if(example.id==='report'){
      assert.match(await frame.locator('#viewer').innerText(),/Order access review/);
      await frame.getByRole('tab',{name:'Source',exact:true}).click();
      assert.equal(await frame.locator('#viewer pre').textContent(),example.source);
    }else if(example.id==='json')assert.equal(await frame.locator('#viewer pre').textContent(),example.source);
    else if(example.id==='source')assert.equal(await frame.locator('.source-lines li').count(),4);
    else{
      assert.doesNotMatch(await frame.locator('#viewer').innerText(),/demo_only_not_a_real_credential/);
      await frame.getByRole('tab',{name:'Response',exact:true}).click();
      assert.match(await frame.locator('#viewer').innerText(),/demo-customer/);
      assert.doesNotMatch(await frame.locator('#viewer').innerText(),/demo_session=not_a_real_cookie/);
    }
  }
  for(const state of ['historical','no-response','incomplete','unavailable','oversize'])for(const width of [1440,390,320]){
    const frame=await open('http',state,'black',width);
    await layout(frame,width);
    const content=await frame.locator('#viewer').innerText();
    if(state==='historical')assert.match(await frame.locator('#badges').innerText(),/Historical revision/);
    if(state==='no-response'){
      assert.doesNotMatch(content,/HTTP 200/);
      await frame.getByRole('tab',{name:'Response',exact:true}).click();
      assert.match(await frame.locator('#viewer').innerText(),/No response captured/);
      const saved=await frame.locator('html').evaluate(()=>source());
      assert.equal(JSON.parse(saved).capture.attempts[0].error,'response_unavailable');
    }
    if(state==='incomplete')assert.match(content,/Incomplete/);
    if(state==='unavailable'){
      assert.match(content,/Material unavailable/);
      assert.equal(await frame.locator('#download').isDisabled(),true);
      assert.equal(await frame.locator('#showVersions').isDisabled(),true);
      await frame.getByRole('button',{name:'Retry',exact:true}).click();
      assert.match(await frame.locator('#status').textContent(),/No network request/);
    }
    if(state==='oversize')assert.match(content,/Inline preview unavailable/);
  }
  let frame=await open('http');
  await frame.getByRole('button',{name:'Show Authorization value',exact:true}).click();
  assert.match(await frame.locator('#viewer').innerText(),/demo_only_not_a_real_credential/);
  await frame.locator('#showVersions').click();
  await frame.locator('#versionList button').nth(1).click();
  assert.doesNotMatch(await frame.locator('#viewer').innerText(),/demo_only_not_a_real_credential/);
  assert.match(await frame.locator('#exact').textContent(),/evidence\/order-exchange@order-exchange-r1/);
  assert.equal(await frame.locator('#back').getAttribute('href'),'#source-issue');
  await frame.getByRole('tab',{name:'Source',exact:true}).click();
  assert.doesNotMatch(await frame.locator('#viewer').innerText(),/demo_only_not_a_real_credential|demo_session=not_a_real_cookie/);
  await frame.locator('html').evaluate(()=>Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('Deterministic permission denial')}},configurable:true}));
  await frame.locator('#copy').click();
  await frame.locator('#status').filter({hasText:'Copy is unavailable'}).waitFor();
  assert.match(await frame.locator('#status').textContent(),/Copy is unavailable/);
  assert.equal(await frame.locator('#technical').getAttribute('open'),'');
  await frame.locator('html').evaluate(()=>Object.defineProperty(navigator,'clipboard',{value:{writeText:async value=>{window.testCopied=value}},configurable:true}));
  await frame.locator('#copy').click();
  await frame.locator('#status').filter({hasText:'Copied identifier'}).waitFor();
  assert.equal(await frame.locator('#status').textContent(),'Copied identifier');
  assert.match(await frame.locator('html').evaluate(()=>window.testCopied),/order-exchange-r1/);
  for(const example of examples.examples){
    frame=await open(example.id);
    const download=page.waitForEvent('download');await frame.locator('#download').click();
    const artifact=await download;
    assert.match(artifact.suggestedFilename(),new RegExp('^'+example.name+'-r2\\.'));
    assert.deepEqual(await readFile(await artifact.path()),Buffer.from(example.source));
  }
  frame=await open('source','oversize');
  const binary=page.waitForEvent('download');await frame.locator('#download').click();
  assert.equal((await readFile(await (await binary).path())).length,examples.binaryExample.sizeBytes);
  frame=await open('source');
  await frame.locator('#wrap').click();
  assert.equal(await frame.locator('.source-lines').getAttribute('data-wrap'),'false');
  assert.equal(await frame.locator('.source-lines li').first().evaluate(n=>getComputedStyle(n).whiteSpace),'pre');
  await frame.locator('#menu').click();
  assert.equal(await frame.locator('#menu').getAttribute('aria-expanded'),'true');
  assert.equal(await frame.locator('#rail a svg').count(),7,'Labelled rail uses real icons');
  frame=await open('report');
  await frame.locator('#newVersion summary').click();
  await frame.locator('#uploadFile').setInputFiles({name:'local-demo.md',mimeType:'text/markdown',buffer:Buffer.from('# Demo')});
  await frame.locator('#upload').click();
  assert.match(await frame.locator('#status').textContent(),/no upload was sent/);
  await frame.locator('#showVersions').click();await frame.locator('#versionList button').nth(1).click();
  assert.equal(await frame.locator('#newVersion').isVisible(),false);
  for(const file of ['source','http','report','json']){
    await page.goto(base+'?'+new URLSearchParams({version:'current',case:file,theme:'dark',width:'390'}));
    const reference=page.frameLocator('#design');await reference.locator('#root').waitFor();
    assert.equal(await reference.locator('#root').getAttribute('inert'),'');
    assert.match(await reference.locator('#root').innerText(),/Download this revision/);
  }
  assert.deepEqual(errors,[],'No browser runtime errors');
  assert.deepEqual(external,[],'No external requests');
  console.log(JSON.stringify({passed:true,layouts,formats:4,themes:3,widths:[1440,1100,390,320],savedBytes:'exact',httpContract:'production parser',isolated:true,screenshots:path.relative(root,screenshots)}));
}finally{
  if(browser)await browser.close();
  child.kill('SIGTERM');
  await once(child,'exit');
}
