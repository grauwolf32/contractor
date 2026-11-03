import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=new URL('./',import.meta.url),pass=new URL('checklist-opendesign-pass-2026-11-02/',root);
const read=(folder,name)=>readFile(new URL(name,folder),'utf8'),sha=text=>createHash('sha256').update(text).digest('hex');
const [input,native,css,candidate,record]=await Promise.all([
 read(pass,'input-v4.html'),read(pass,'index-checklist-v1.html'),read(pass,'checklist-polish-v1.css'),read(root,'preview.html'),read(pass,'provenance.json')
]);
const head=input.indexOf('</head>'),prefix=input.slice(0,head),suffix=input.slice(head);
if(!native.startsWith(prefix)||!native.endsWith(suffix))throw Error('Native output changed the input outside the style insertion');
const inserted=native.slice(prefix.length,native.length-suffix.length),style=inserted.match(/^\s*<style(?:\s+id="[\w-]+")?>\s*([\s\S]*?)\s*<\/style>\s*$/);
if(!style||style[1].trim()!==css.trim())throw Error('Native HTML does not contain the exact generated CSS as its only insertion');
const scripts=html=>[...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map(match=>match[1]);
if(JSON.stringify(scripts(input))!==JSON.stringify(scripts(candidate)))throw Error('Refinement changed a local controller or fixture');
if(/@import|@font-face|url\s*\(|#[\da-f]{3,8}(?![\w-])|rgba?\s*\(/i.test(css))throw Error('Generated CSS adds external assets, fonts or literal colors');
const declared=new Set([...input.matchAll(/(--[\w-]+)\s*:/g)].map(match=>match[1]));
const unknown=[...css.matchAll(/var\((--[\w-]+)/g)].map(match=>match[1]).filter(name=>!declared.has(name));
if(unknown.length)throw Error('Generated CSS uses unknown theme tokens: '+[...new Set(unknown)]);
const provenance=JSON.parse(record);
if(provenance.status!=='succeeded'||provenance.inputSha256!==sha(input))throw Error('Missing successful native run or exact input identity');
if(!provenance.cleanup.stopped||!provenance.cleanup.upstreamCredentialRemoved||provenance.cleanup.requestsUsed>24)throw Error('Relay cleanup/budget did not pass');
if(provenance.preservation.priorFilesChanged||provenance.preservation.priorFilesMissing||!Object.values(provenance.preservation.tables).every(value=>value.allPriorRowsIdentical))throw Error('Prior OpenDesign data changed');
console.log('Verified genuine OpenDesign output, exact CSS-only insertion, all original scripts/fixtures, shared tokens, native preservation and relay cleanup.');
