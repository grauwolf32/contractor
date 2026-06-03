---
description: Business-logic / abuse-of-functionality detection for the vuln_scan skill. Flaws where every per-endpoint control (auth, authz, validation) is present but the operation still violates an application invariant. Scanners miss these because nothing matches a dangerous-sink grep — you must reason about state and sequence.
---

# Business-Logic / Abuse-of-Functionality

These flaws have **no dangerous sink and no missing auth/authz row** — the
handler can be fully authenticated, authorized, and input-validated and
still be broken, because it violates a rule the framework cannot enforce.
Grep alone will not find them; check each pattern on every handler that
changes a resource whose correctness depends on application rules:
**money, balances, inventory/quantity, credits/points/votes, quotas/limits,
ownership transfer, or a multi-step workflow.**

Report these as their own class (do not force them into SQLi/IDOR). Name the
invariant that breaks and the request sequence that breaks it.

## 1. Check-then-act without atomicity (race / TOCTOU) → CWE-362

A balance/quota/stock value is read, validated, then mutated in **separate
statements** with no lock or atomic operation. Two concurrent requests both
pass the check, then both mutate → negative balance, over-redeem, oversell,
double-withdraw.

Vulnerable shape:
```python
acct = Account.get(id)
if acct.balance >= amount:          # check
    acct.balance -= amount          # ...then act (separate write)
    acct.save()
```
**Safe:** single atomic UPDATE with a guard, a transaction + row lock
(`SELECT … FOR UPDATE`), or an atomic decrement
(`UPDATE accounts SET balance = balance - :amt WHERE id=:id AND balance >= :amt`).

**How to check:** for each money/quantity/stock/quota mutation, confirm the
read and the write are one atomic step. `grep -n "FOR UPDATE\|with_for_update\|select_for_update\|transaction(\|atomic(\|F(" ` to find the *protected* ones; handlers that mutate without any of these are candidates.

## 2. Missing idempotency / replay → CWE-837

A charge / transfer / redeem / submit / vote applies cumulatively when the
same request is sent twice — no idempotency key, nonce, or once-only flag.

**How to check:** for each value-moving POST, look for an idempotency key in
the request contract or a uniqueness guard before the effect
(`grep -ni "idempoten\|nonce\|once\|already_(processed|redeemed|used)\|unique"`).
Absent on a charge/transfer/redeem → finding.

## 3. Unbounded / negative amount or quantity → CWE-20

Amount/quantity/price validated for *type* but not for *sign and range*. A
negative transfer reverses the flow (credit the attacker); a huge value
overflows or drains.

```python
amount = int(request.json["amount"])   # type-checked, sign unchecked
transfer(src, dst, amount)             # amount = -100 → reverse transfer
```
**How to check:** trace each numeric amount/quantity from the request to its
use; confirm a `> 0` / range check rejects (not coerces) out-of-range
values before the effect.

## 4. Client-trusted security-critical value → CWE-602

Price, total, discount, role, status, balance, tier, or another user's id is
taken from the **request body** and used as truth instead of being recomputed
or looked up server-side.

```python
order.total = request.json["total"]    # attacker sets total = 0.01
user.role  = request.json.get("role")  # privilege escalation
```
**How to check:** for any field that affects money, access, or state, confirm
the server derives it (catalog price, session role) rather than reading it
from the request.

## 5. Workflow / state-machine bypass → CWE-841

A step that must follow a prior one is independently reachable or can be
reordered: ship before pay, activate before verify, confirm without the
preceding challenge, set-new-password without a valid reset token, mark-paid
without a payment.

**How to check:** for multi-step flows, list the steps and ask whether each
later endpoint verifies the prior step's completion (a server-side state
field / token), or whether it can be called directly.

## 6. Per-user limit / quota not enforced server-side → CWE-840

A cap (free trials, invites, votes, withdrawals/day, coupon uses) is enforced
in the client/UI or optimistically but not atomically on the server.

**How to check:** find the limit constant; confirm the server counts and
enforces it transactionally on every path, not just the happy one.

## Reporting

- Class → CWE: race/TOCTOU CWE-362 · replay/idempotency CWE-837 · workflow
  bypass CWE-841 · client-trusted value CWE-602 · unbounded amount CWE-20 ·
  general business-logic CWE-840.
- In `details`: name the invariant, the violating sequence (e.g. "two
  concurrent POST /transfer with the same balance"), and the evidence lines.
- Severity: a financial invariant break exploitable by any user (double-spend,
  negative transfer, balance/quota race) is **high**; a quota/workflow bypass
  without direct financial loss is **medium**.
- Do not require a proof-of-exploit — a visible non-atomic mutation, a
  trusted client value, or a reachable out-of-order step is the finding.
