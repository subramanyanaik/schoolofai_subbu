# Pushing this to GitHub

> Delete this file before or after pushing — it's setup scaffolding, not part of the submission.

## 1. Create the repo

On GitHub: **New repository** → name it something like `erav5-mixture-curriculum` → **Public** (the reviewer must be able to see it) → **do not** initialise with a README, licence, or .gitignore (they already exist here).

## 2. Fix the badge URL

`README.md` line 3 has a placeholder:

```
https://github.com/USERNAME/REPO/actions/workflows/validate.yml
```

Replace `USERNAME` and `REPO` with your actual GitHub username and repo name — two occurrences on that line. If you'd rather not bother, delete that one badge line; the other three badges are static and work anywhere.

## 3. Push

```bash
cd <this-folder>
git init
git add .
git commit -m "ERA V5 mixture and curriculum specification"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git push -u origin main
```

## 4. Verify before submitting

- [ ] Open the repo URL in a **logged-out browser window** — if it 404s, it's private. Settings → General → Danger Zone → Change visibility.
- [ ] README renders: tables have borders, the `<details>` blocks in §10 expand when clicked.
- [ ] Actions tab shows a green check (the workflow runs `run_plan` and fails the build if any of the 49 assertions fail).
- [ ] Links to `results/computed_output.txt` and `src/erav5/*.py` resolve.
- [ ] Badge is green, not "workflow not found".

## 5. Submit

The link to submit is the repo root:

```
https://github.com/<USERNAME>/<REPO>
```

GitHub renders `README.md` automatically at that URL, which is what the reviewer evaluates.

## What the reviewer sees

The README is self-contained — every table, number, and decision rule is inline, so nothing needs to be cloned or run to grade it. The code is there for anyone who wants to check that the numbers are computed rather than asserted:

| Claim in README | Verify by |
|---|---|
| 49/49 checks pass | Actions tab (green check), or `cd src && python3 -m erav5.run_plan` |
| Stage×lane integrates at 0.00pp drift | §4.1 table, or `results/plan.json` → `integration` |
| Every number is computed | `src/erav5/config.py` holds the inputs; nothing else is hardcoded |
| Supply figures are honest | `config.py` → `INVENTORY`, with `[V4]/[SESS]/[PUB]/[EST]` provenance tags |
