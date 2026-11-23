---
description: Language-specific dangerous function/method names for source code scanning — organized by vulnerability class with safe vs vulnerable patterns.
---

# Sink Patterns for Code Review

Grep for these patterns, then read surrounding code to confirm.
Vulnerable = user input reaches the function without validation.

## SQL Injection

### Python
- VULNERABLE: `cursor.execute("SELECT " + user_input)`, `f"SELECT {user_input}"`, `.extra(where=[user_input])`, `.raw(user_query)`
- SAFE: `cursor.execute("SELECT ... WHERE x=%s", [param])`, ORM `.filter(field=value)`

### Java
- VULNERABLE: `stmt.executeQuery("SELECT " + input)`, `createNativeQuery(sql + input)`, `createQuery(hql + input)`
- SAFE: `PreparedStatement` with `?` placeholders, JPA `@Query` with `:param` binding

### Go
- VULNERABLE: `db.Query("SELECT " + input)`, `db.Exec(fmt.Sprintf("DELETE ... %s", input))`
- SAFE: `db.Query("SELECT ... WHERE x=$1", param)`, parameterized

### Node.js
- VULNERABLE: `db.query("SELECT " + req.body.id)`, template literals in SQL
- SAFE: `db.query("SELECT ... WHERE x=$1", [param])`, query builders

### Ruby / Rails
- VULNERABLE: `Model.where("name = '#{params[:name]}'")`, `find_by_sql("... #{input}")`, interpolation `#{...}` inside `where(`/`order(`/`pluck(`/`group(`
- SAFE: `where("name = ?", value)`, `where(name: value)`, hash conditions

### .NET / C#
- VULNERABLE: `new SqlCommand("SELECT ... " + input)`, `string.Format`/`$"...{input}"` into `CommandText`, then `ExecuteReader`/`ExecuteNonQuery`/`ExecuteScalar`
- SAFE: `SqlParameter` / `cmd.Parameters.Add(...)` / `cmd.Parameters.AddWithValue("@x", value)`

### MongoDB (NoSQL Injection)
- VULNERABLE: `collection.find(JSON.parse(userInput))`, `bson.M` from `json.Unmarshal(body, &bsonMap)`, `{$where: userInput}`
- SAFE: explicit field construction `{field: value}`, schema validation before query

## Command Injection

### Python
- VULNERABLE: `os.system(cmd + input)`, `subprocess.call(cmd, shell=True)` with user input, `eval(input)`, `exec(input)`
- SAFE: `subprocess.run([binary, arg1, arg2], shell=False)`, shlex.quote

### Java
- VULNERABLE: `Runtime.getRuntime().exec("cmd " + input)`, `ProcessBuilder(Arrays.asList("sh", "-c", userCmd))`
- SAFE: `ProcessBuilder(Arrays.asList(binary, arg1))` without shell, input allowlisting

### Go
- VULNERABLE: `exec.Command("sh", "-c", userInput)`, `exec.Command(userBinary)`
- SAFE: `exec.Command(fixedBinary, sanitizedArg)` with allowlist

### Node.js
- VULNERABLE: `child_process.exec(cmd + input)` / `execSync(...)` with concatenation (spawns a shell), `eval(input)`, `new Function(input)`, `vm.runInContext(input)`, `require(varName)` with a non-literal, `setTimeout(stringCode)`
- SAFE: `execFile(binary, [arg1, arg2])` / `spawn(binary, [args])` with an argument array and no shell, fixed `require` literals

### Ruby / Rails
- VULNERABLE: `system(cmd + input)`, `exec`, `spawn`, `\`#{input}\``, `eval(input)`, `Open3.capture2`/`Open3.popen3`/`IO.popen`/`PTY.spawn` with user input
- REFLECTION: `input.constantize`/`safe_constantize`, `obj.send(input)`/`public_send`/`__send__`, `obj.try(input)` — user-controlled method/class name = RCE
- SAFE: array-form `system(binary, arg1)`, allowlist of permitted methods/classes

### .NET / C#
- VULNERABLE: `Process.Start(input)`, `ProcessStartInfo{ UseShellExecute=true, Arguments=input }`, `cmd /c` with concatenation
- SAFE: `ProcessStartInfo{ FileName=binary, ArgumentList={...}, UseShellExecute=false }`, allowlist

## SSRF

### All languages
- VULNERABLE: `requests.get(user_url)`, `http.Get(userURL)`, `fetch(req.body.url)`, `file_get_contents($url)`, `HttpClient.send(req)` where URL from user
- .NET: `HttpClient.GetAsync(userUrl)`, `WebRequest.Create(userUrl)`; Ruby: `Net::HTTP.get(URI(userUrl))`, `open(userUrl)`
- SAFE: URL parsed + host checked against allowlist, schema restricted to https, no redirect following to internal nets

### Bypass indicators in code
- Denylist only (blocklist for `127.0.0.1` / `localhost`) — bypassable via `0x7f000001`, `[::1]`, DNS rebinding
- `follow_redirects=True` / default redirect following — redirect to `file://` or internal host

## Path Traversal

### Python
- VULNERABLE: `open(os.path.join(base, user_filename))` without normalization check, `send_file(user_path)`
- SAFE: `os.path.realpath(path).startswith(allowed_dir)`, filename allowlist

### Java
- VULNERABLE: `new File(basePath + userInput)`, `Paths.get(base, userInput)` without canonical check
- SAFE: `path.toRealPath().startsWith(allowedBase)`, filename regex allowlist

### Go
- VULNERABLE: `filepath.Join(base, userInput)` without `filepath.Rel` check, `os.Open(userPath)`
- SAFE: `filepath.Rel(base, full)` returns no `..`, `filepath.Clean` + prefix check

### Node.js
- VULNERABLE: `fs.readFile(req...)` / `fs.createReadStream(req...)` with a user-derived path, `res.sendFile(userPath)` without a `root` option / normalization check
- SAFE: `path.resolve(base, name)` then verify it `startsWith(base)`, `res.sendFile(name, { root: base })`, filename allowlist

## Deserialization

### Python
- DANGEROUS: `pickle.loads(user_data)`, `yaml.load(data)` (without SafeLoader), `yaml.unsafe_load(data)`
- SAFE: `yaml.safe_load(data)`, `json.loads(data)`, `pickle` only on trusted data

### Java
- DANGEROUS: `ObjectInputStream.readObject()`, `XMLDecoder.readObject()`, `XStream.fromXML(userInput)`
- SAFE: `ObjectInputFilter` (Java 9+), allowlisted classes, JSON instead of Java serialization
- MAGIC BYTES: `AC ED 00 05` (binary), `rO0` (base64) — presence indicates Java serialization

### PHP
- DANGEROUS: `unserialize($userInput)` without `allowed_classes => false`
- SAFE: `json_decode($input)`, `unserialize($data, ['allowed_classes' => ['Safe']])`

### Ruby
- DANGEROUS: `Marshal.load(user_data)`, `YAML.load(data)` (pre-Psych-4), `Oj.load`/`Oj.object_load` with default mode
- SAFE: `YAML.safe_load(data)`, `JSON.parse(data)`, `Oj.load(data, mode: :strict)`

### .NET / C#
- DANGEROUS: `BinaryFormatter`, `LosFormatter`, `SoapFormatter`, `NetDataContractSerializer`, `JavaScriptSerializer` with a resolver, Newtonsoft `JsonConvert` with `TypeNameHandling` != `None`
- SAFE: `System.Text.Json`, Newtonsoft with default `TypeNameHandling.None`, `DataContractSerializer` with known types

### Node.js
- DANGEROUS: `node-serialize.unserialize(userInput)`, `funcster.deepDeserialize(...)`, `js-yaml` `load(userInput)` (legacy default schema executes types)
- SAFE: `js-yaml` `safeLoad` / `load(..., { schema: JSON_SCHEMA })`, `JSON.parse(...)`

## Prototype Pollution (Node.js)

Untrusted keys written into objects can poison `Object.prototype`, affecting every object in the process (DoS, property injection, sometimes RCE).

- VULNERABLE: `obj[userKey] = val` where `userKey` comes from request data, recursive `merge` / `_.merge` / `Object.assign` / `$.extend(true, ...)` applied to untrusted JSON, query-string parsers that build nested objects
- DANGER KEYS to flag in input or merge logic: `__proto__`, `constructor`, `prototype`
- SAFE: reject/strip those keys, `Object.create(null)` maps, `Map` instead of plain objects, schema-validate before merge, `_.merge` only over allowlisted keys

## Server-Side Template Injection (SSTI) — CRITICAL

User input must reach the template **string** itself, not just the data context. Passing user data as a *variable* to a precompiled template is safe; building the template source from user input is RCE.

### Python (Flask/Jinja2)
- VULNERABLE: `render_template_string(user_input)`, `Template(user_input).render(...)`, `Environment(...).from_string(user_input)`, `autoescape=False`
- SAFE: `render_template("file.html", var=user_input)` — user data bound as a context variable only

### Go
- VULNERABLE: `text/template` rendering user data (no auto-escaping), `template.HTML(userInput)` (bypasses escaping)
- SAFE: `html/template` with user data as a typed context value

### Node.js
- VULNERABLE: mustache/handlebars with escaping disabled, eval-based engines, `dot`/`ejs` `<%- userInput %>` (unescaped), template source from user input
- SAFE: precompiled templates with user data passed as the data object, escaped interpolation (`<%= %>`)

## XXE (XML External Entity)

VULNERABLE when these parse **user-supplied XML** without disabling DOCTYPE/external entities:

### Java
- `DocumentBuilderFactory` / `SAXParserFactory` / `XMLInputFactory` / `TransformerFactory` used without `setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)` or `FEATURE_SECURE_PROCESSING`
- SAFE: doctype disabled, `XMLConstants.ACCESS_EXTERNAL_DTD`/`SCHEMA` set to `""`

### .NET / C#
- `XmlTextReader` (DTD enabled by default pre-4.5), `XmlReaderSettings { DtdProcessing = DtdProcessing.Parse }`
- SAFE: `DtdProcessing = DtdProcessing.Prohibit`, `XmlResolver = null`

### PHP
- `simplexml_load_string`/`simplexml_load_file`/`DOMDocument->load*` with `LIBXML_NOENT`, or `libxml_disable_entity_loader(false)`
- SAFE: default flags (no `LIBXML_NOENT`), entity loader left disabled

### Node.js
- `libxmljs` parse with `{ noent: true }`, `xml2json` and similar that expand entities
- SAFE: entity expansion disabled / not enabled

## LDAP / XPath Injection

### .NET / C# — LDAP
- VULNERABLE: `DirectorySearcher`/`DirectoryEntry` with `Filter` built by concatenating user input
- SAFE: `Encoder.LdapFilterEncode(input)` before building the filter

### .NET / C# — XPath
- VULNERABLE: `SelectNodes`/`SelectSingleNode`/`XPathNavigator.Compile` with concatenated user input
- SAFE: parameterized XPath (`XPathExpression` with variables), input allowlist

## Cross-Site Scripting (XSS) — output sinks

### Ruby / Rails
- VULNERABLE: `input.html_safe`, `raw(input)`, `<%== input %>` — all bypass ERB auto-escaping
- SAFE: default `<%= input %>` (auto-escaped), `sanitize(input)` with an allowlist

### .NET / Razor
- VULNERABLE: `@Html.Raw(input)` — bypasses Razor auto-encoding
- SAFE: `@input` (auto-encoded), `@Html.Encode(input)`

## Render / Local File Inclusion

### Ruby / Rails
- VULNERABLE: `render file: params[:f]`, `render inline: params[:t]`, `render template:`/`render action:` with user-controlled path (inline → SSTI/RCE, file → LFI)
- SAFE: render fixed template names; never pass user input to `file:`/`inline:`/`template:`

## JWT Vulnerabilities

- ALG:NONE: `PlainJWT.parse()` fallback, `jwt.decode(verify=False)`, `algorithms` param missing
- KEY CONFUSION: HS256 verification with RSA public key as secret
- KID INJECTION: `kid` header used in `SQL query` / `file path` / `command`
- WEAK SECRET: `SECRET_KEY = "secret"` / short string / dictionary word
- NO EXPIRY CHECK: token accepted without `exp` claim validation

## Authentication Bypass

- MISSING AUTH: handler without `@login_required` / `@jwt_auth_required` / `@PreAuthorize` / `ensureAuthenticated` middleware
- 2FA RACE: session created BEFORE MFA check (`session['user'] = user` then `if user.mfa_enabled`)
- RESET TOKEN: generated from `time.time()` / `random.random()` (predictable), or no expiry
- HOST HEADER: `request.host` / `$_SERVER['HTTP_HOST']` used to build password reset URL

## Broken Access Control

- IDOR: `Model.findById(req.params.id)` without `WHERE owner = currentUser`
- BFLA: admin endpoint without role check (`@PreAuthorize("hasRole('ADMIN')")` missing)
- MASS ASSIGNMENT: `Model.update(req.body)` / `@RequestBody Entity` without DTO / `@JsonIgnore`
- MASS ASSIGNMENT (Rails): `params.permit!` (blanket), `attr_accessible`, `Model.new(params[:model])` without strong-param allowlist
- PRIVILEGE FIELD: `role`, `is_admin`, `balance`, `credit` accepted from user input

## CORS / Header Misconfig

The high-severity case is a permissive origin **paired with credentials** — any website can then read authenticated responses (CWE-942).

- VULNERABLE: `Access-Control-Allow-Origin: *` (or reflecting the request `Origin` header) **together with** `Access-Control-Allow-Credentials: true`
- Grep: `Allow-Origin.*\*`, request-origin reflection (`Allow-Origin.*Origin`), `setAllowedOrigins?\(["']?\*`, Express `cors\(\)` with no options, `origin:\s*true`, `CORS_ORIGIN_ALLOW_ALL\s*=\s*True` / `CORS_ALLOW_ALL_ORIGINS\s*=\s*True`
- SAFE: explicit origin allowlist, credentials disabled when origin is wildcard, no echoing of arbitrary `Origin`

## CSRF Protection Disabled / Missing

For cookie/session-authenticated **state-changing** endpoints, missing or disabled CSRF protection is a finding (CWE-352). Token-only (Authorization-header) APIs are generally not affected.

- VULNERABLE (Python/Django): `@csrf_exempt` / `csrf_exempt(...)` on a state-changing view
- VULNERABLE (Rails): `skip_before_action :verify_authenticity_token`, `protect_from_forgery ... except: [...]`
- VULNERABLE (.NET): `[IgnoreAntiforgeryToken]` on a POST/PUT/DELETE action
- VULNERABLE (Express): cookie-session app with no `csurf` (or equivalent) middleware on POST/PUT/PATCH/DELETE routes
- SAFE: framework CSRF middleware active app-wide, per-request anti-forgery token validated, or a pure bearer-token API with no ambient cookie auth

## Go-specific sinks

Beyond SQLi / command injection (above), flag these Go patterns:
- SSTI / XSS: `text/template` used to produce HTML output (no contextual auto-escaping — use `html/template`), `template.HTML(userInput)` (explicitly marks user input as safe HTML)
- OPEN REDIRECT: `http.Redirect(w, r, userURL, ...)` where the target comes from user input without host allowlisting
- DEBUG EXPOSURE: `import _ "net/http/pprof"` or a `/debug/pprof` route registered on a publicly reachable mux — leaks goroutines/heap/env
- ZIP SLIP: `filepath.Join(dest, f.Name)` during archive (zip/tar) extraction without rejecting entries whose cleaned path escapes `dest` (`..`)

## Race Conditions

- NON-ATOMIC: `if balance >= amount: balance -= amount` without lock
- TOCTOU: separate read + write without `SELECT FOR UPDATE` / `F()` expression / mutex
- MISSING UNIQUE: redemption table without `UNIQUE(coupon_id, user_id)` constraint
