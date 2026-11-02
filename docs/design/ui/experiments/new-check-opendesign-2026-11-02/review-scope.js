// Candidate-only review extension. The model's original controller is retained.
(() => {
  const clone=value=>JSON.parse(JSON.stringify(value)), bytes=value=>new TextEncoder().encode(value).length;
  paths.trash='<path d="M3 6h18M9 6V4h6v2M5 6l1 14h12l1-14M10 10v6M14 10v6"/>';
  paths.plus='<path d="M12 5v14M5 12h14"/>';
  const contents=new Map(), scopeCard=$('scope-copy').closest('.scope');
  const newEntry=key=>({key,version:'1',statement:'',applicability:'always',allowed_methods:['static'],required_evidence:[],review_policy:'automatic'});
  const ui={mode:'material',dirty:false,draft:[newEntry('check-0001')],applied:[],revision:0,nextKey:2};
  const itemCount=count=>`${count} ${count===1?'item':'items'}`;
  for(const [name,kind] of [['openapi','application/yaml'],['checklist','application/json']]) {
    const fixture=SCOPE_FIXTURES[name], original=FIXTURES.materials.find(material=>material.mediaType===kind);
    Object.assign(original,{artifact:clone(fixture.artifact),size:bytes(JSON.stringify(fixture.document))});
    contents.set(key(original),clone(fixture.document));
  }
  const isChecklist=()=>type().id==='source-checklist',isAPI=()=>type().id==='openapi-operation-trace';
  const scopeBody=document.createElement('div');scopeBody.id='material-scope';scopeCard.append(scopeBody);
  const variants=document.createElement('section');variants.className='review-coverage';variants.id='review-coverage';
  variants.append(text('h4','Review coverage'),$('scope-options'));$('advanced').querySelector('.advanced-content').prepend(variants);
  const checklistDefaults=document.createElement('section');checklistDefaults.className='review-coverage';checklistDefaults.id='checklist-defaults';checklistDefaults.append(text('h4','Checklist defaults'),text('p','Static code review · always applicable · automatic review policy · no required evidence types. Each item keeps its own key and version.'));$('advanced').querySelector('.advanced-content').append(checklistDefaults);
  const listDocument=()=>({schema:'contractor.audit.checklist.v1',items:clone(ui.draft)});
  function error(){
    if(!ui.draft.length)return 'Add at least one item to define the check.';
    if(ui.draft.length>SCOPE_FIXTURES.limits.maxItems)return 'This check type accepts up to 128 items.';
    for(const [index,item] of ui.draft.entries()) {
      if(!item.statement.trim())return `Item ${index+1} needs a statement.`;
      if(item.statement.includes('\0')||bytes(item.statement)>SCOPE_FIXTURES.limits.maxStringBytes)return `Item ${index+1} exceeds the text limit or contains an invalid character.`;
    }
    if(bytes(JSON.stringify(listDocument()))>SCOPE_FIXTURES.limits.maxDocumentBytes)return 'The checklist exceeds the document size limit.';
    return '';
  }
  function currentDocument(){return contents.get(state.selected[isAPI()?'openapi':'checklist'])}
  function operations(document){return Object.entries(document?.paths??{}).flatMap(([path,item])=>['get','put','post','delete','options','head','patch','trace'].filter(method=>item[method]).map(method=>({path,method:method.toUpperCase()})))}
  function facts(entries){const dl=document.createElement('dl');dl.className='scope-facts';for(const [label,value] of entries){const row=document.createElement('div');row.append(text('dt',label),text('dd',value));dl.append(row)}return dl}
  function renderScope(){
    const shown=isAPI()||isChecklist();scopeCard.hidden=!shown||isChecklist()&&ui.mode==='write'&&ui.dirty;variants.hidden=!type().id.includes('asvs');checklistDefaults.hidden=!isChecklist()||ui.mode!=='write';scopeBody.replaceChildren();
    if(!shown){if(!type().id.includes('asvs'))$('summary-scope').textContent='Whole source archive';return}
    const doc=currentDocument(),editing=isChecklist()&&ui.mode==='write',count=isAPI()?operations(doc).length:editing?ui.draft.filter(item=>item.statement.trim()).length:doc?.items?.length;
    const label=isAPI()?'API spec':'Checklist';
    $('scope-copy').textContent=isAPI()?(doc?`All ${count} endpoints in the selected API spec.`:'Choose an API spec to define the scope.'):(editing&&ui.dirty?(count?`${itemCount(count)} drafted. Apply the checklist to set the scope.`:'Add your first item to define the scope.'):doc?`All ${itemCount(count)} in the selected checklist.`:'Choose or write a checklist to define the scope.');
    $('scope-copy').className='coverage-heading';
    $('summary-scope').textContent=doc&&(!editing||!ui.dirty)?`All ${isAPI()?count+' endpoints':itemCount(count)}`:editing?(count?`${itemCount(count)} drafted`:'Checklist draft'):'Not selected';
    const ref=state.selected[isAPI()?'openapi':'checklist'];
    if(ref){const source=text('p',`From ${label==='API spec'?label:'checklist'} · ${ref}`);source.className='scope-source';scopeBody.append(source)}
    if(isAPI()&&doc){
      const details=document.createElement('details');details.className='scope-note';details.append(text('summary','View endpoints'));
      const list=document.createElement('ul');list.className='endpoint-list';
      for(const operation of operations(doc)){const row=document.createElement('li'),method=text('span',operation.method);method.className='endpoint-method';row.append(method,text('code',operation.path));list.append(row)}
      details.append(list);scopeBody.append(details);
      const note=text('p',`The check reads every endpoint. Profile limit: ${type().maxItems}.`);note.className='scope-note';scopeBody.append(note);
    }
    if(isAPI()&&new URLSearchParams(location.search).get('case')==='engagement')renderEngagement();
  }
  function renderEngagement(){
    const fixture=SCOPE_FIXTURES.engagement,engagement=fixture.document,details=document.createElement('details');details.className='engagement-scope';details.id='engagement-scope';details.open=true;details.append(text('summary','Engagement'));
    const ref=text('p',`${fixture.artifact.namespace}/${fixture.artifact.name}@${fixture.artifact.revision}`);ref.className='scope-source';details.append(ref);
    details.append(facts([
      ['Target',engagement.target.origins.join(', ')],
      ['Allowed',engagement.scope.routes.map(route=>route.pathPrefixes.join(', ')+' · '+route.methods.join(', ')).join('; ')],
      ['Excluded',engagement.scope.exclusions.map(route=>route.pathPrefixes.join(', ')).join('; ')||'None'],
      ['Access',engagement.accessMode==='source-assisted'?'Source-assisted':'Blackbox'],
      ['Identities',engagement.identities.map(identity=>identity.role?`${identity.role}${identity.tenant?' · tenant '+identity.tenant:''}`:identity.kind).join('; ')]
    ]));
    const technical=document.createElement('details');technical.className='scope-note';technical.append(text('summary','Limits and technical details'),facts([
      ['Active time',`${engagement.budgets.activeSeconds/3600} hour`],['Rounds',String(engagement.budgets.rounds)],['Child runs',String(engagement.budgets.submittedRuns)],
      ['Addresses',engagement.scope.routes.flatMap(route=>route.addressRanges).join(', ')],['Policy',engagement.policy.activeChecks]
    ]));details.append(technical);scopeBody.append(details);
  }
  function updateEditorStatus(){
    const feedback=$('checklist-feedback');if(!feedback)return;
    const invalid=error(),empty=ui.draft.every(item=>!item.statement.trim()),applied=!invalid&&!ui.dirty;
    feedback.dataset.tone=empty||invalid.includes('needs a statement')?'draft':invalid?'error':ui.dirty?'draft':'ready';
    feedback.textContent=empty?'Add at least one item to define your checklist.':invalid||(ui.dirty?`${itemCount(ui.draft.length)} · unapplied changes`:`${itemCount(ui.draft.length)} · ready to use`);
    const apply=$('apply-checklist');apply.disabled=!!invalid||!ui.dirty;apply.classList.toggle('is-applied',applied);apply.textContent=applied?'Checklist ready':'Use checklist';
    $('editor-state').textContent=applied?'Ready':'Draft';$('editor-state').dataset.tone=applied?'ready':'draft';
    $('add-checklist-item').disabled=ui.draft.length>=SCOPE_FIXTURES.limits.maxItems;
  }
  function dirty(){ui.dirty=true;state.selected.checklist='';updateEditorStatus();renderReadiness()}
  function renderEditor(row){
    row.replaceChildren();row.className='checklist-editor';row.id='checklist-editor';
    const head=document.createElement('header');head.className='checklist-editor-head';const title=text('h4','Your checklist');title.insertAdjacentHTML('afterbegin',icon('file'));
    const editorState=text('span','Draft');editorState.id='editor-state';editorState.className='editor-state';title.append(editorState);
    const existing=text('button','Use existing');existing.setAttribute('aria-label','Use a project material');existing.type='button';existing.onclick=()=>{ui.mode='material';const material=FIXTURES.materials.find(m=>m.artifact.name==='security-checklist');state.selected.checklist=key(material);render()};head.append(title,existing);row.append(head);
    const intro=text('p','One clear check per item.');intro.className='checklist-intro';row.append(intro);
    const list=document.createElement('ol');list.className='checklist-entries';row.append(list);
    for(const [index,item] of ui.draft.entries()){
      const li=document.createElement('li');li.className='checklist-entry';li.dataset.key=item.key;
      const ordinal=text('span',String(index+1).padStart(2,'0'));ordinal.className='checklist-ordinal';ordinal.setAttribute('aria-hidden','true');
      const body=document.createElement('div');body.className='checklist-entry-body';const label=text('label',`Item ${index+1}`);label.htmlFor='statement-'+item.key;label.className='checklist-entry-label';
      const remove=document.createElement('button');remove.type='button';remove.className='checklist-remove';remove.setAttribute('aria-label',`Remove item ${index+1}`);remove.innerHTML=icon('trash');remove.onclick=()=>{ui.draft.splice(index,1);dirty();renderMaterials();const next=$('checklist-editor').querySelectorAll('textarea')[Math.min(index,ui.draft.length-1)];(next||$('add-checklist-item')).focus()};
      const input=document.createElement('textarea');input.id=label.htmlFor;input.className='checklist-statement';input.rows=2;input.maxLength=SCOPE_FIXTURES.limits.maxStringBytes;input.value=item.statement;input.placeholder='Describe what should be verified…';input.oninput=()=>{item.statement=input.value;dirty()};body.append(label,input);li.append(ordinal,body,remove);list.append(li);
    }
    const controls=document.createElement('div');controls.className='checklist-editor-actions';
    const add=text('button','Add item');add.insertAdjacentHTML('afterbegin',icon('plus'));add.type='button';add.id='add-checklist-item';add.onclick=()=>{
      if(ui.draft.length>=SCOPE_FIXTURES.limits.maxItems)return;
      ui.draft.push(newEntry('check-'+String(ui.nextKey++).padStart(4,'0')));dirty();renderMaterials();$('checklist-editor').querySelector('li:last-child textarea').focus();
    };
    const apply=text('button','Use checklist');apply.type='button';apply.id='apply-checklist';apply.onclick=()=>{
      if(error()||!ui.dirty)return;
      for(const item of ui.draft){const previous=ui.applied.find(known=>known.key===item.key);if(previous&&previous.statement!==item.statement)item.version=String(Number(previous.version)+1)}
      const document=listDocument();ui.applied=clone(ui.draft);ui.dirty=false;ui.revision++;
      let material=FIXTURES.materials.find(m=>m.artifact.name==='inline-checklist');
      if(!material){material=clone(FIXTURES.materials.find(m=>m.artifact.name==='security-checklist'));material.artifact={namespace:'docs',name:'inline-checklist',revision:''};FIXTURES.materials.push(material)}
      const oldKey=key(material);contents.delete(oldKey);material.artifact.revision='local-'+ui.revision;material.size=bytes(JSON.stringify(document));contents.set(key(material),document);state.selected.checklist=key(material);
      render();announce('Checklist applied to this preview. No material was saved to the project.');
    };
    controls.append(add,apply);row.append(controls);
    const feedback=document.createElement('p');feedback.id='checklist-feedback';feedback.className='checklist-feedback';feedback.setAttribute('role','status');row.append(feedback);
    updateEditorStatus();
  }
  const originalMaterials=renderMaterials;
  renderMaterials=function(){
    originalMaterials();if(!isChecklist())return;
    const row=$('material-slots').lastElementChild;
    if(ui.mode==='write'){renderEditor(row);return}
    const create=text('button','Write checklist');create.type='button';create.className='checklist-create';create.id='write-checklist';create.onclick=()=>{ui.mode='write';if(!ui.revision)ui.dirty=true;state.selected.checklist=ui.dirty?'':key(FIXTURES.materials.find(m=>m.artifact.name==='inline-checklist'));render();$('checklist-editor').querySelector('textarea')?.focus()};row.append(create);
  };
  const originalReadiness=renderReadiness;
  renderReadiness=function(){
    originalReadiness();renderScope();
    if(isChecklist()&&ui.mode==='write'&&(ui.dirty||error())){
      const empty=ui.draft.every(item=>!item.statement.trim()),incomplete=ui.draft.some(item=>!item.statement.trim());
      $('start').disabled=true;$('save').disabled=true;$('readiness').dataset.tone='missing';$('readiness').textContent=empty||incomplete?'Checklist draft':error()?'Check the checklist':'Unapplied changes';$('readiness-copy').textContent=empty?'Add your first item, then apply the checklist.':incomplete?'Complete each item, then apply the checklist.':error()||'Apply your checklist changes before starting.';
    }
    if(isAPI()&&new URLSearchParams(location.search).get('case')==='engagement'){
      $('start').disabled=true;$('save').disabled=true;$('readiness').dataset.tone='missing';$('readiness').textContent='Engagement preview';$('readiness-copy').textContent='Placement example. This engagement is not attached; compatibility has not been checked.';
    }
    updateEditorStatus();
  };
  const originalInitialize=initialize;
  initialize=function(){originalInitialize();if(isChecklist()){
    if(ui.mode==='write')state.selected.checklist=ui.dirty||!ui.revision?'':key(FIXTURES.materials.find(m=>m.artifact.name==='inline-checklist'));
    else state.selected.checklist=key(FIXTURES.materials.find(m=>m.artifact.name==='security-checklist'));
  }};
  const requested=new URLSearchParams(location.search).get('type');if(FIXTURES.types.some(t=>t.id===requested))state.type=requested;
  window.__scopeReview={document:()=>clone(contents.get(state.selected.checklist)),draft:listDocument,dirty:()=>ui.dirty};
  initialize();render();
})();
