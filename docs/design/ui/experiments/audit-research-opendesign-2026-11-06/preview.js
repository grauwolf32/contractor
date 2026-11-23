// Host-owned synthetic interactions. No API client, credentials or target activity.
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const params = new URLSearchParams(location.search);
document.documentElement.dataset.theme = ['light','dark','black'].includes(params.get('theme')) ? params.get('theme') : 'dark';
const pill = (label, tone = 'neutral') => `<span class="pill" data-tone="${tone}">${escape(label)}</span>`;
const action = (label, name, primary = false) => `<button type="button" class="btn" data-action="${name}" ${primary ? 'data-primary' : ''}>${escape(label)}</button>`;
const data = {
  order: {statement:'A customer can read another customer’s order', origin:'Fixed checklist · order ownership', verdict:'Supported', tone:'warning', scope:'GET /api/orders/{id} · customer A and customer B · Shop API baseline r7', grounds:'Order lookup accepts a customer-supplied identifier. Ownership enforcement needs a direct comparison.', result:'Customer A received order_b42 owned by customer B. The response included order contents and delivery details.', conditions:'Two authenticated customers; one endpoint; one recorded request pair.', limitation:'Other order endpoints and account roles were not tested.', sources:['Order API observations','Order ownership evidence'], task:'Order ownership check', revision:1},
  search: {statement:'Product search input can change the SQL query', origin:'Fixed checklist · query construction', verdict:'Refuted', tone:'success', scope:'src/routes/products/search.ts · pinned source revision', grounds:'The product search route passes user-provided search terms to a query builder.', result:'The selected route uses bound parameters for the tested input. The query structure did not change.', conditions:'Pinned source baseline; the selected search route and query builder.', limitation:'This result does not describe other database access paths.', sources:['Product search route','Product search check result'], task:'Product search query check', revision:1},
  refresh: {statement:'A refresh token can be reused across sessions', origin:'Generated · Research hypotheses', verdict:'Inconclusive', tone:'warning', scope:'POST /api/auth/refresh · selected baseline', grounds:'The observed token flow lacks a recorded comparison between independent authenticated sessions.', result:'The required second session was unavailable. The observed response cannot establish cross-session reuse.', conditions:'One authenticated session; only the supplied refresh request.', limitation:'A second independent authenticated session is required.', sources:['Authentication observations','Prior check result'], task:'Refresh-token reuse check', revision:1},
  invoice: {statement:'An invoice export can cross account boundaries', origin:'Generated · Research hypotheses', verdict:'Checking', tone:'info', scope:'GET /api/invoices/{id}/export · customer A / customer B', grounds:'Order access observations suggest invoice exports may reuse an identifier-based lookup.', result:'The accepted check has one active Run. No scoped result has been collected yet.', conditions:'The check will use the exact selected hypothesis revision.', limitation:'Run execution is not a substantive result.', sources:['Order API observations','Order ownership result'], task:'Check invoice export access', revision:1},
  tenant: {statement:'A tenant can enumerate another tenant’s exports', origin:'Generated · Research hypotheses', verdict:'Not checked', tone:'neutral', scope:'GET /api/exports · tenant A and tenant B', grounds:'The API specification declares tenant selectors. Cross-tenant filtering has not been observed.', result:'This hypothesis is registered. No checking task has been accepted for it.', conditions:'No tested conditions yet.', limitation:'Registration does not schedule a check.', sources:['Shop API specification'], task:undefined, revision:1},
  session: {statement:'A session can be used after sign out', origin:'Fixed checklist · session lifecycle', verdict:'Not checked', tone:'neutral', scope:'POST /api/logout and GET /api/me', grounds:'The fixed checklist requires a session invalidation comparison.', result:'A checking task is accepted and awaits dispatch.', conditions:'No collected check yet.', limitation:'An accepted task is not a result.', sources:[], task:'Review session isolation', revision:1},
  admin: {statement:'A customer can access administrator functions', origin:'Fixed checklist · role enforcement', verdict:'Not checked', tone:'neutral', scope:'Selected administrative API endpoints', grounds:'The fixed checklist includes role boundary checks.', result:'No check result has been collected for this fixed hypothesis.', conditions:'No tested conditions yet.', limitation:'The unchecked entry stays in the original denominator.', sources:[], task:undefined, revision:1},
};
let selectedHypothesis = 'order';
const hypothesisHistory = {};
let extraTasks = 0;
let paused = params.get('state') === 'paused';
let cancelled = false;
let tab = 'overview';
let toastTimer;
function toast(message) {
  const box = $('#status'); box.textContent = message; box.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => box.hidden = true, 6500);
}
function selectTab(next, update = true) {
  if (!$(`[data-panel="${next}"]`)) next = 'overview';
  tab = next;
  $$('[data-panel]').forEach(p => {p.hidden = p.dataset.panel !== next; p.setAttribute('role','tabpanel'); p.setAttribute('aria-labelledby',`tab-${p.dataset.panel}`);});
  $$('.section-tabs [data-tab]').forEach(b => {b.setAttribute('aria-selected',String(b.dataset.tab === next)); b.tabIndex = b.dataset.tab === next ? 0 : -1;});
  if (next === 'hypotheses') renderHypothesis(selectedHypothesis);
  if (update) {params.set('tab',next); history.replaceState(null,'','?'+params);}
  window.parent.postMessage({type:'audit-preview-tab',tab:next},location.origin);
}
function renderHypothesis(key) {
  selectedHypothesis = key; const h = data[key]; if (!h) return;
  $$('.hypothesis-row').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.hypothesis === key)));
  $('#hypothesis-detail').innerHTML = `<header><div class="inline">${pill(h.origin.startsWith('Fixed')?'Fixed':h.origin.startsWith('Manual')?'Manual':'Generated',h.origin.startsWith('Generated')?'info':'neutral')}${pill(h.verdict,h.tone)}</div><h2>${escape(h.statement)}</h2><p class="caption">Exact assertion · revision ${h.revision}</p></header><section><h3>Scope &amp; conditions</h3><p>${escape(h.scope)}</p></section><section><h3>Why investigate this</h3><p>${escape(h.grounds)}</p></section><section class="scoped-result" data-tone="${h.tone}"><div class="inline between"><h3>${h.verdict === 'Not checked' || h.verdict === 'Checking' ? 'Checking work' : 'Recorded result'}</h3>${pill(h.verdict,h.tone)}</div><p>${escape(h.result)}</p><dl class="result-facts"><div><dt>Tested conditions</dt><dd>${escape(h.conditions)}</dd></div><div><dt>Limitations</dt><dd>${escape(h.limitation)}</dd></div></dl>${key==='order'?action('Inspect possible issue','issue-open'):''}</section><section><h3>Grounds · exact materials</h3>${h.sources.length?h.sources.map(s=>`<button class="source-link" data-action="material-report"><span>${escape(s)}</span><small>Retained exact revision</small><span aria-hidden="true">↗</span></button>`).join(''):'<p class="caption">No exact source material was selected.</p>'}</section><section class="origin-note"><h3>Origin</h3><p>${escape(h.origin)}</p><p class="caption">${h.origin.startsWith('Generated')?'Producer: Research hypotheses → ordinary Run → accepted proposal output.':h.origin.startsWith('Manual')?'Explicit analyst input; origin remains unchanged across edits.':'Frozen source: Order access checklist, version 1.'}</p><p class="caption">${h.task ? 'Checking task: '+escape(h.task)+'. Exact hypothesis association is system-derived.' : 'No checking task. A check can be requested explicitly.'}</p></section><footer class="hypothesis-actions">${action('Edit hypothesis','edit-hypothesis')}${h.verdict==='Checking'?'':action('Check hypothesis','check-hypothesis',true)}<button class="quiet-link" data-action="hypothesis-history">Check history →</button></footer><details class="technical"><summary>Technical details</summary><p>Hypothesis <code>hyp_${escape(key)}…1a3f</code> · revision ${h.revision}</p><p>Origin, pinned sources and exact checked snapshot are recorded by the owner.</p></details>`;
}
const dialog = $('#dialog');
function openDialog(title, eyebrow, content, actions = action('Close','dialog-close')) {
  $('#dialog-title').textContent = title; $('#dialog-eyebrow').textContent = eyebrow;
  $('#dialog-content').innerHTML = content; $('#dialog-actions').innerHTML = actions;
  if (!dialog.open) dialog.showModal();
  $('input,textarea,select', $('#dialog-content'))?.focus();
}
function closeDialog() {dialog.close(); $('#dialog-content').replaceChildren();}
function controlState() {
  const b = $('[data-action="pause"], [data-action="resume"]');
  if (!b) return;
  b.dataset.action = paused?'resume':'pause'; b.textContent = paused?'Resume accepted work':'Pause new Audit Runs';
  b.disabled = cancelled;
  const p = $('.check-header .pill'); p.textContent = cancelled?'Cancelling':paused?'Paused':'Active'; p.dataset.tone = cancelled?'neutral':paused?'warning':'info';
  const notice = $('#state-notice'); notice.hidden = !paused && !cancelled;
  notice.dataset.tone = cancelled?'neutral':'warning';
  notice.textContent = cancelled?'Preview: cancellation requested. No new work will be dispatched or admitted. One active Run is still draining; terminal cancellation has not been reported.':paused?'New Audit Runs are paused. The current Run can finish and its outputs can still be collected. Accepted work is retained for Resume.':'';
}
function hypothesisForm(edit = false) {
  const h = edit ? data[selectedHypothesis] : undefined;
  openDialog(edit?'Edit hypothesis':'Add a hypothesis','Assertion · scope · grounds',`<form id="hypothesis-form"><label for="statement">Statement</label><textarea id="statement" name="statement" required rows="3" placeholder="A customer can … under these conditions">${escape(h?.statement??'')}</textarea><label for="conditions">Scope &amp; conditions</label><textarea id="conditions" name="conditions" rows="2" required placeholder="Endpoint, actors and selected baseline">${escape(h?.scope??'')}</textarea><label for="grounds">Why investigate this</label><textarea id="grounds" name="grounds" rows="2" required>${escape(h?.grounds??'')}</textarea><label>Checking method <span class="caption">optional</span><select id="method" name="method"><option value="">Save without scheduling a check</option><option value="access-check">HTTP access check · pinned Program</option><option value="source-check">Source assertion check · pinned Program</option></select></label><p class="caption">The system assigns identity and revision and pins the assertion to any accepted check.</p><div id="form-message" role="status"></div></form>`,action('Cancel','dialog-close')+action('Save hypothesis',edit?'save-edit':'save-hypothesis',true));
  $('#hypothesis-form').addEventListener('submit',e=>e.preventDefault());
  $('#method').addEventListener('change',()=>{$('#dialog-actions [data-primary]').textContent=$('#method').value?'Save & check':'Save hypothesis';});
}
function saveHypothesis(edit) {
  if (!$('#hypothesis-form').reportValidity()) return;
  if (params.get('state')==='conflict' && !$('#form-message').dataset.reloaded) {
    $('#form-message').className='conflict-notice'; $('#form-message').innerHTML='<strong>The saved revision changed.</strong><p>Your text is kept here. Reload the current assertion before another write. No save was simulated.</p>'+action('Reload current assertion','reload-assertion');
    $('#dialog-actions [data-primary]').disabled=true; return;
  }
  const key=edit?selectedHypothesis:'manual-'+Object.keys(data).length;
  const method=$('#method').value;
  if(edit)(hypothesisHistory[key]??=[]).push({...data[key]});
  const h={statement:$('#statement').value.trim(),scope:$('#conditions').value.trim(),grounds:$('#grounds').value.trim(),origin:edit?data[key].origin:'Manual · demo analyst',verdict:'Not checked',tone:'neutral',result:method?'Checking task accepted in this preview. No result has been collected.':'Hypothesis saved in this preview. No check was scheduled.',conditions:'No tested conditions yet.',limitation:'This is a local simulation, not a server result.',sources:edit?data[key].sources:[],task:method?'Explicit manual check':undefined,revision:edit?data[key].revision+1:1};
  data[key]=h;
  if (!edit) {const row=document.createElement('button');row.className='hypothesis-row';row.dataset.hypothesis=key;row.innerHTML=`<span class="inline">${pill('Manual')}${pill('Not checked')}</span><strong>${escape(h.statement)}</strong><small>${method?'Check accepted':'Saved · no check scheduled'}</small>`;$('.hypothesis-list').append(row);}
  $('#tab-hypotheses .count').textContent=String(Object.keys(data).length);
  if(method)addPendingTask(h.statement);
  closeDialog(); selectTab('hypotheses');renderHypothesis(key);toast('Preview only: '+(method?'hypothesis and checking task accepted.':'hypothesis saved without scheduling a check.'));
}
function addPendingTask(statement){
  extraTasks++;
  const row=document.createElement('li');row.className='work-row';row.dataset.group='pending';row.innerHTML=`<span class="timeline-icon" data-tone="neutral">○</span><div class="row-copy"><strong>${escape(statement)}</strong><p>Accepted · exact manual hypothesis snapshot · not dispatched</p></div>${pill('Accepted')}`;$('#work-list').append(row);
  $('.task-progress>.inline>span').textContent=`${8+extraTasks} accepted tasks`;
  $('#tab-work .count').textContent=String(8+extraTasks);$('.work-section [data-tab=work]').textContent=`All ${8+extraTasks} tasks →`;$('#work-filter option[value=all]').textContent=`All tasks · ${8+extraTasks}`;$('#work-filter option[value=pending]').textContent=`Pending · ${3+extraTasks}`;
  const legend=$$('.progress-legend>span');legend[2].querySelector('strong').textContent=String(3+extraTasks);
  $('.progress-track').setAttribute('aria-label',`4 completed, 1 technical gap, ${3+extraTasks} pending`);
  for(const [cls,count] of [['completed',4],['gap',1],['pending',3+extraTasks]])$('.progress-track .'+cls).style.width=(100*count/(8+extraTasks))+'%';
}
function previewMaterial(kind) {
  const source = kind==='source';
  openDialog(source?'Product search route':kind==='api'?'Shop API specification':kind==='result'?'Order ownership result':'Order API observations','Exact retained material · synthetic',`<div class="inline">${pill(source?'TypeScript':'Accepted material')}${pill('Revision 1')}</div><pre class="material-preview">${escape(source?'// Synthetic example: parameterized lookup\nconst orders = await db.query(\n  "SELECT * FROM products WHERE name = $1",\n  [searchTerm]\n);':kind==='api'?'openapi: 3.1.0\ninfo:\n  title: Shop API\npaths:\n  /api/orders/{id}:\n    get:\n      summary: Read an order':kind==='result'?'Verdict: supported\nChecked hypothesis: fixed order ownership, revision 1\nTested conditions: customer A / customer B\nObservation: A received the order owned by B\nEvidence: request pair and redacted response':'# Order API observations\n\nThe API exposes order lookup and invoice export.\nThe order response includes an owner identifier.\n\n## Observed conditions\nPinned API baseline; two synthetic customer accounts.\n\n## Gaps\nCross-tenant export filtering has not been checked.')}</pre><details class="technical"><summary>Technical details</summary><p>Audit-owned exact revision · source Task and Run retained in lineage.</p><p>The preview renders text. It does not execute material content.</p></details>`,action('Close','dialog-close')+action('Publish to Project','publish-'+kind,true));
}
document.addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b) return;
  if(b.dataset.tab) {selectTab(b.dataset.tab);return;}
  if(b.dataset.hypothesis) {renderHypothesis(b.dataset.hypothesis);return;}
  const a=b.dataset.action; if(!a) return;
  if(a==='dialog-close') {closeDialog();return;}
  if(a==='pause') {paused=true;controlState();return;}
  if(a==='resume') {paused=false;controlState();return;}
  if(a==='cancel') {openDialog('Cancel this check?','Final action','<p>No new Runs will start. Active work will be cancelled and drained. Collected results remain available.</p><p>Cancellation cannot be resumed. This preview simulates the state only.</p>',action('Keep check','dialog-close')+action('Cancel check','confirm-cancel'));return;}
  if(a==='confirm-cancel') {cancelled=true;closeDialog();controlState();$$('[data-action="cancel"]').forEach(x=>x.disabled=true);toast('Preview only: cancellation requested; active work is still draining.');return;}
  if(a==='context-toggle') {const open=b.getAttribute('aria-expanded')==='true';b.setAttribute('aria-expanded',String(!open));$('#context-pane').classList.toggle('context-open',!open);return;}
  if(a.startsWith('hypothesis-')&&a!=='hypothesis-history') {const key=a.slice(11);selectTab('hypotheses');renderHypothesis(key);return;}
  if(a==='issue-open'||a==='issue-origins') {selectTab('issues');return;}
  if(a==='add-hypothesis'||a==='edit-hypothesis') {hypothesisForm(a==='edit-hypothesis');return;}
  if(a==='save-hypothesis'||a==='save-edit') {saveHypothesis(a==='save-edit');return;}
  if(a==='reload-assertion') {const h=data[selectedHypothesis];$('#statement').value=h.statement;$('#conditions').value=h.scope;$('#grounds').value=h.grounds;$('#form-message').dataset.reloaded='true';$('#form-message').textContent='Current assertion reloaded. You can edit it again.';$('#dialog-actions [data-primary]').disabled=false;return;}
  if(a==='check-hypothesis') {openDialog('Check this hypothesis','Explicit checking task',`<p>${escape(data[selectedHypothesis].statement)}</p><label>Checking method<select id="checking-method"><option>HTTP access check · pinned Program</option><option>Source assertion check · pinned Program</option></select></label><p class="caption">The exact assertion and available materials are pinned automatically. This preview starts no Run.</p>`,action('Cancel','dialog-close')+action('Accept checking task','accept-check',true));return;}
  if(a==='accept-check') {addPendingTask(data[selectedHypothesis].statement);data[selectedHypothesis].task='Explicit manual check';data[selectedHypothesis].result='A checking task is accepted; no result has been collected.';closeDialog();renderHypothesis(selectedHypothesis);toast('Preview only: checking task accepted. No result has been collected.');return;}
  if(a==='hypothesis-history') {openDialog('Check history','Exact assertion revisions',`<p><strong>${escape(data[selectedHypothesis].verdict)}</strong> · ${escape(data[selectedHypothesis].conditions)}</p><p>${escape(data[selectedHypothesis].result)}</p><p class="caption">Negative rechecks remain results. They do not change the Project issue assessment.</p>`);return;}
  if(a.startsWith('material-')) {previewMaterial(a.slice(9));return;}
  if(a.startsWith('publish-')) {const key=a.slice(8);openDialog('Publish exact material to Project','Independent Project retention',`<p>The selected exact revision will remain available in Shop API even after the source check is removed.</p><label>Material name<input id="publish-name" required value="${key==='report'?'Order API observations':'Selected check material'}"></label><p class="caption">No recursive publication of links inside the payload. This is a preview-only action.</p>`,action('Cancel','dialog-close')+action('Publish exact revision','confirm-publish',true));return;}
  if(a==='confirm-publish') {if(!$('#publish-name').reportValidity())return;closeDialog();toast('Preview only: exact material published to Project with source lineage.');return;}
  if(a==='proposal-open') {openDialog('Saved proposal output','Producer still active',`<p>The Workflow saved a typed proposal artifact. Its Run has not completed successfully and no semantic intake has registered a hypothesis from this output.</p>${pill('Saved proposal','info')}<p>It may be inspected as diagnostic material. It is not an approval request or an accepted check.</p>`);return;}
  if(a.startsWith('task-')) {selectTab('work');const box=$('#task-detail');box.hidden=false;box.innerHTML=a==='task-gap'?'<h3>Run succeeded · collection rejected</h3><p>The proposal output referred to a missing exact ground. No hypothesis or checking task was accepted from this batch.</p><p class="caption">Technical execution remains Succeeded. The collection gap is retained separately.</p>':'<h3>'+escape(a==='task-invoice'?'Invoice export check':a==='task-tenant'?'Tenant boundary comparison':'Session isolation check')+'</h3><p>The accepted task keeps its exact materials and hypothesis snapshot, where applicable.</p><details class="technical"><summary>Attempts &amp; technical details</summary><p>Ordinary Run attempt 1 · pinned Workflow revision.</p><p>Deleted Runs remain marked as historical provenance.</p></details>';box.scrollIntoView({block:'nearest'});return;}
  if(a==='load-older') {$('#older-status').textContent='All synthetic events are loaded. Production loads the next bounded cursor page.';b.disabled=true;return;}
  if(a==='copy-id') {navigator.clipboard?.writeText('audit_order_authorization_a7c2').then(()=>toast('Synthetic Audit ID copied.')).catch(()=>toast('Clipboard unavailable.'));return;}
  if(a==='download') {const blob=new Blob(['Synthetic Contractor design evidence. No live data.\n'],{type:'text/plain'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='synthetic-evidence.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);return;}
  if(a==='evidence-preview'||a==='evidence-details') {openDialog('Cross-account response','Exact retained Project evidence','<pre class="material-preview">GET /api/orders/order_b42\nAuthorization: [redacted]\n\n200 OK\nowner: customer_b</pre><p class="caption">Synthetic redacted capture · exact source revision. No target traffic.</p>');return;}
  if(a==='issue-decision') {openDialog(b.textContent.trim(),'Canonical Project decision','<p>The production UI reuses the existing Issues decision owner, its required fields, conflict handling and history.</p><label>Why<textarea rows="3" placeholder="Explain the evidence and conditions"></textarea></label><p class="caption">This review does not record a real analyst decision.</p>',action('Cancel','dialog-close')+action('Simulate decision','demo-decision',true));return;}
  if(a==='demo-decision') {closeDialog();toast('Preview only: decision simulated; the real Project issue was not changed.');return;}
  if(a==='scope') {openDialog('Scope & selected materials','Pinned baseline','<p>Shop API · exact baseline r7</p><p>Selected order and invoice endpoints; customer A and customer B.</p><p>Materials: API specification, source baseline, access notes and the fixed checklist.</p><p>Allowed operations, identities and HTTP limits come from the selected engagement material.</p>');return;}
  if(a==='new-check') {openDialog('Start a check','Existing check setup', '<label>Your objective<textarea rows="2">Investigate access across customer accounts.</textarea></label><label>Check type<select><option>Order access research</option><option>Source checklist</option></select></label><p>Selected materials: Shop API specification, source baseline and access notes.</p><details class="technical"><summary>Initial hypotheses & optional continuation</summary><p>The frozen checklist provides fixed seeds. An optional ordinary Workflow can propose new hypotheses using selected exact materials.</p></details><p class="caption">This design review starts no Audit.</p>',action('Close','dialog-close'));return;}
  if(a==='add-work') {openDialog('Add work','Ordinary workflow · selected exact materials',`<label>What should this work investigate?<textarea id="work-objective" rows="2" required placeholder="Explore invoice export access"></textarea></label><label>Method<select id="work-method"><option>Explore API surface</option><option>HTTP access check</option><option>Source assertion check</option><option>Research hypotheses</option></select></label><fieldset class="material-picker"><legend>Exact materials</legend>${['Shop API specification · r7','Source baseline · r3','Order API observations · r1','Order ownership result · r1'].map((m,i)=>`<label><input type="checkbox" name="work-material" value="${i}" ${i===0?'checked':''}><span>${escape(m)}</span></label>`).join('')}</fieldset><p class="caption">Selected as one material set. Only methods and inputs allowed by the pinned Program are offered.</p><p class="caption">Exploration needs no hypothesis. To check an assertion, use Check hypothesis so its exact snapshot is attached automatically.</p>`,action('Cancel','dialog-close')+action('Accept work','accept-work',true));return;}
  if(a==='accept-work') {const field=$('#work-objective');if(!field.reportValidity())return;addPendingTask(field.value.trim());closeDialog();toast('Preview only: ordinary work accepted with the selected exact materials.');return;}
  toast('Design preview: this destination uses the existing Contractor screen.');
});
$('.section-tabs').addEventListener('keydown',e=>{const tabs=$$('.section-tabs [role="tab"]'),i=tabs.indexOf(document.activeElement);if(i<0)return;let j;if(e.key==='ArrowRight')j=(i+1)%tabs.length;else if(e.key==='ArrowLeft')j=(i-1+tabs.length)%tabs.length;else if(e.key==='Home')j=0;else if(e.key==='End')j=tabs.length-1;else return;e.preventDefault();tabs[j].focus();selectTab(tabs[j].dataset.tab);});
$('#check-search').addEventListener('input',e=>$$('.context-row').forEach(b=>b.hidden=!b.textContent.toLowerCase().includes(e.target.value.toLowerCase())));
$$('#work-list .work-row').forEach((r,i)=>r.dataset.group=i<3?'pending':i===7?'gap':'collected');
$('#work-filter').addEventListener('change',e=>$$('#work-list .work-row').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.group!==e.target.value));
$('#material-filter').addEventListener('change',e=>$$('#material-list .material-row').forEach((r,i)=>r.hidden=e.target.value==='outputs'?i>=2:e.target.value==='inputs'?i<2:false));
$('#panel-work>.section-heading').insertAdjacentHTML('beforeend',action('Add work','add-work',true));
selectTab(params.get('tab')??'overview',false);controlState();renderHypothesis(params.get('hypothesis') in data?params.get('hypothesis'):'order');
if(params.get('state')==='stale-read') {const n=$('#state-notice');n.hidden=false;n.dataset.tone='warning';n.innerHTML='<strong>Could not refresh this check.</strong> Previously loaded data is kept. '+action('Try again','retry-read');document.addEventListener('click',e=>{if(e.target.closest('[data-action="retry-read"]')){n.hidden=true;toast('Preview: refresh recovered.');}});}
if(params.get('state')==='gap') selectTab('work',false);
if(params.get('state')==='conflict') {selectTab('hypotheses',false);hypothesisForm(true);}
window.AUDIT_PREVIEW_READY = true;
