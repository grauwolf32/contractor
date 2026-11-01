import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const dir = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.argv[2] ?? 17562);
const childPolicy = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; sandbox allow-scripts allow-downloads";
const shim = `<script>Object.defineProperty(window,'localStorage',{value:{getItem:()=>null,setItem:()=>{},removeItem:()=>{},clear:()=>{}},configurable:false});</script>`;
const metrics = `<script>function issueMetrics(){parent.postMessage({type:'issue-design-metrics',width:innerWidth,documentWidth:document.documentElement.scrollWidth},'*')}addEventListener('load',issueMetrics);addEventListener('resize',issueMetrics);new MutationObserver(()=>requestAnimationFrame(issueMetrics)).observe(document.documentElement,{childList:true,subtree:true,attributes:true});</script>`;
const wrapper = `<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Contractor · Issues · Design review</title><style>
:root{color-scheme:light;--bg:#eceef0;--text:#24262a;--line:#ccd0d5;--surface:#fff}html[data-theme=dark]{color-scheme:dark;--bg:#101216;--text:#e4e6ea;--line:#343940;--surface:#191d23}html[data-theme=black]{color-scheme:dark;--bg:#000;--text:#e4e6ea;--line:#292d34;--surface:#0d0f12}*{box-sizing:border-box}body{margin:0;padding:14px 16px;background:var(--bg);font:13px ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--text)}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}nav strong{margin-right:auto}label{display:flex;align-items:center;gap:6px}select{font:inherit;padding:5px 8px;border:1px solid var(--line);border-radius:5px;background:var(--surface);color:var(--text)}#stage{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:5px}iframe{position:absolute;top:0;left:0;transform-origin:top left;display:block;border:0}a{color:inherit}#metrics{font-size:11px;opacity:.7;margin:10px 0}
</style><nav><strong>Issues · OpenDesign</strong><label>Version<select id="version"><option value="new">New composition</option><option value="previous">Previous design</option></select></label><label>Theme<select id="theme"><option>light</option><option>dark</option><option>black</option></select></label><label>Width<select id="width"><option>1440</option><option>1100</option><option>390</option><option>320</option></select></label><a href="http://127.0.0.1:17560/test-results/issue-design/index-resumed.html?width=1440&theme=dark" target="_blank" rel="noopener">Current UI</a></nav><div id="stage"><iframe id="design" title="OpenDesign Issue review" sandbox="allow-scripts allow-downloads" height="1000"></iframe></div><p id="metrics">Synthetic demo · no decisions are sent</p><script>
const query=new URLSearchParams(location.search),frame=document.getElementById('design');for(const key of ['version','theme','width']){const select=document.getElementById(key);if([...select.options].some(o=>o.value===query.get(key)))select.value=query.get(key);select.onchange=()=>{const params=new URLSearchParams;for(const k of ['version','theme','width'])params.set(k,document.getElementById(k).value);location.search=params.toString()}}const width=+document.getElementById('width').value,theme=document.getElementById('theme').value;document.documentElement.dataset.theme=theme;frame.width=width;frame.src=(document.getElementById('version').value==='previous'?'/previous.html':'/prototype.html')+'?theme='+theme+'&width='+width;function fit(){const scale=Math.min(1,(innerWidth-34)/width),stage=document.getElementById('stage');stage.style.width=width*scale+2+'px';stage.style.height=Number(frame.height)*scale+2+'px';frame.style.transform='scale('+scale+')'}fit();addEventListener('resize',fit);addEventListener('message',e=>{if(e.source!==frame.contentWindow||e.data?.type!=='issue-design-metrics')return;document.getElementById('metrics').textContent=e.data.width+' px viewport / '+e.data.documentWidth+' px document · synthetic demo; no decisions are sent'});
</script></html>`;
const server=http.createServer(async(req,res)=>{
  if(!/^(127\.0\.0\.1|localhost):[0-9]+$/.test(req.headers.host??'')){res.writeHead(403).end();return}
  if(req.method!=='GET'){res.writeHead(405).end();return}
  const url=new URL(req.url,'http://127.0.0.1:'+port);
  res.setHeader('Cache-Control','no-store');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('X-Content-Type-Options','nosniff');
  if(url.pathname==='/'){
    res.setHeader('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'");res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'}).end(wrapper);return;
  }
  if(url.pathname==='/prototype.html'||url.pathname==='/previous.html'){
    try{
      const file=url.pathname==='/previous.html'?new URL('../issues-opendesign-2026-11-01/preview.html',import.meta.url):new URL('preview.html',import.meta.url);
      const html=(await readFile(file,'utf8')).replace(/<head[^>]*>/i,head=>head+shim).replace(/<\/body>/i,metrics+'</body>');
      res.setHeader('Content-Security-Policy',childPolicy);res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'}).end(html);
    }catch{res.writeHead(503,{'Content-Type':'text/plain'}).end('Design is still being prepared.')}
    return;
  }
  res.writeHead(404).end();
});
server.listen(port,'127.0.0.1',()=>console.log(JSON.stringify({url:'http://127.0.0.1:'+server.address().port+'/'})));
