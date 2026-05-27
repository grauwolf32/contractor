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

## SSRF

### All languages
- VULNERABLE: `requests.get(user_url)`, `http.Get(userURL)`, `fetch(req.body.url)`, `file_get_contents($url)`, `HttpClient.send(req)` where URL from user
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
- PRIVILEGE FIELD: `role`, `is_admin`, `balance`, `credit` accepted from user input

## Race Conditions

- NON-ATOMIC: `if balance >= amount: balance -= amount` without lock
- TOCTOU: separate read + write without `SELECT FOR UPDATE` / `F()` expression / mutex
- MISSING UNIQUE: redemption table without `UNIQUE(coupon_id, user_id)` constraint
