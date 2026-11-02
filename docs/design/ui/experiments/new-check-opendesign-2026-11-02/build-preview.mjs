import {readFile, writeFile} from 'node:fs/promises';
const root=new URL('./',import.meta.url),read=name=>readFile(new URL(name,root),'utf8');
const [original,objective,scope,fixtures,extension]=await Promise.all(['original.html','review-objective.css','review-scope.css','scope-fixtures.json','review-scope.js'].map(read));
const styles=`<style id="review-objective">\n${objective}</style>\n<style id="review-scope">\n${scope}</style>\n`;
const script=`<script id="scope-review">\nconst SCOPE_FIXTURES=${JSON.stringify(JSON.parse(fixtures)).replaceAll('<','\\u003c')};\n${extension}\n</script>`;
await writeFile(new URL('preview.html',root),original.replace('</head>',styles+'</head>').replace('</body>',script+'</body>'));
