---
name: appbuilder
description: >-
  Plan and build a complete app from a one-line idea. Interviews briefly, researches the
  real products in the category and saves the sources, inventories every feature with
  stable IDs (U01, U02, ...), gates the plan, then builds it unattended in measured waves
  on a git branch with a locked scorer, a keep-or-undo loop and a ledger. Use for
  "build me an app", "$appbuilder", "plan and build X", "clone Y for Z", or any new
  product that should be feature-complete rather than a demo. Not for small fixes or
  single features inside an existing codebase.
metadata:
  author: rafik
  version: "2.0"
  origin: "Rebuilt for Codex from the appBuilder Claude Code skill and the appBuilder Upgrade Map"
---

# appBuilder

You turn a one-line idea into a working, feature-complete app in two halves.
First you find out what to build: interview, research, inventory, gates.
Then you build it in a loop that measures every attempt and keeps only what
scores better. You wait for a human exactly twice: the interview answers and
scope approval. After approval you run until nothing is left or you are stopped.

## Rules that never change

1. **One number.** `projectmds/score.py` prints `defect_score`. Lower is better.
   It is the only definition of "better" in this run.
2. **The ruler is locked.** After the baseline commit never edit `score.py`,
   the gates, or the file templates in this skill. Change the work, never the measure.
3. **Research is evidence.** Once `research.md`, `units.tsv` and `sources/` pass
   the gates they are frozen. The plan may change; the evidence may not.
   Every unit carries a saved source, so padding the list cannot raise the score.
4. **One change per commit.** Build it, check it, score it. Better: keep.
   Worse, or equal after tie-breaks: revert to the last keep.
5. **Every attempt gets a ledger row.** Keeps, discards and crashes alike.
   Read the ledger before proposing anything; a dead end stays dead.
6. **Smaller can win.** Same coverage with fewer chunks is a keep, not a tie.
   Ties go to the smaller plan.
7. **Checks run; nobody reads them.** Every "done when" is a command. Prose
   criteria are allowed only when nothing can be scripted, and they count against you.
8. **Stop asking.** After scope approval do not ask questions. Decide, log, move on.
9. **The scope veto stays.** Never add a stage the human did not approve.
   Never exceed `max_rounds_per_wave`.

## Files

Everything lives in `projectmds/` at the repo root.

| File | Role | Mutable? |
|---|---|---|
| `program.md` | Per-project rules: priorities, budget, what "better" means, stages to chase | Until approval |
| `research.md` | Reference products, journeys, size class, gate results | Frozen after gates |
| `units.tsv` | Feature inventory with stable IDs and sources | Frozen after gates |
| `sources/<product>/` | Raw saved pages the units point to | Frozen after gates |
| `stage-NN.md` | The plan: one file per stage, chunks inside | Yes, one change per commit |
| `score.py` | The ruler | Locked after baseline |
| `ledger.tsv` | The notebook, one row per attempt | Append only |
| `logs/` | Subagent output; read at most the last 50 lines of any file | Scratch |
| `handoff.md` | Where a fresh session resumes from | Rewritten at every wave end |

## Phase 1: Interview

Ask at most three questions, in one message, together with a proposed project
name and run tag (`<name>-<yyyymmdd>`). Skip any question the idea already answers.

1. What is it and who uses it? Name the primary user and the job they come to do.
2. Which size class fits? Propose one: **S** tool (8 to 20 units),
   **M** product (20 to 60 units), **L** platform (60 to 150 units).
   Ask for a stack preference only if it is not obvious from the repo.
3. What must be true for version one to be worth using? Their answer becomes
   the `priorities` in `program.md`.

Wait for the answers. This is the first of the two waits.

## Phase 2: Setup ritual

Do all six, in order, before any research. Most hour-three failures die here.

1. Cut a fresh branch `appbuilder/<run tag>` from the current head.
2. Read what already exists: README, package manifests, `projectmds/` from an
   earlier run, and the ledger if there is one. Note anything reusable in `handoff.md`.
3. Prove the toolchain: scaffold the chosen stack, install, run one trivial test,
   and run the `fast_suite` command. Commit `chore: toolchain proof`.
   If install fails, fix it now; do not plan on a stack that does not install.
4. Write `program.md` from the template below.
5. Write `score.py` from the scorer spec below, run it against the empty plan,
   confirm it prints `defect_score` first. Commit `chore: scorer`. It is now locked.
6. Create `ledger.tsv` with the header row only.

### program.md template

```
# program — <project name> · <run tag>
size_class: S | M | L
priorities:
  1. <most important outcome>
  2. <second>
  3. <third>
better_means: lower defect_score; coverage of the spine units before anything else
stages_to_chase: 1 2 3
budget: <wall-clock cap for this run, e.g. 6h>
max_rounds_per_wave: 3
max_parallel: 4
stack: <language, framework, database, test runner>
fast_suite: <one command that finishes in under two minutes>
```

## Phase 3: Research the real product

You are better than any loop at one thing: finding out what actually exists. Do it well.

1. Pick two to four real products in the category: the category leader, a lean
   alternative, and anything the user named.
2. Spawn one subagent per product, in parallel, and wait for all of them. Each
   brief says: save the raw pages you rely on (feature pages, docs, help center,
   changelog, pricing) under `projectmds/sources/<product>/`; write
   `projectmds/sources/<product>/findings.md` as a flat list of user-facing
   capabilities, each with the saved file it came from; send long output to
   `projectmds/logs/research-<product>.log`; return exactly one line:
   the findings path and the count. An empty return is a crash: read the last
   50 lines of its log, then re-run or drop the product.
3. Merge the findings into `units.tsv`. One row per distinct user-facing
   capability. A unit is something a user would notice missing, not a code module.
4. Walk the journeys. For each primary user write the numbered steps from first
   open to job done in `research.md`. Tag every step with the unit ids it needs.
   A step with no unit means a missing unit: find a saved page that shows it,
   add the row. If no source shows it, it goes under "Unsourced ideas", not in `units.tsv`.
5. Assign ids `U01`, `U02`, ... in journey order. Ids are permanent. Never
   renumber, never reuse, never delete; a unit you later drop keeps its row.

### units.tsv template

```
id	name	journey	source
U01	Create a quote from a job	owner:job-to-invoice	sources/jobber/quotes.html
U02	Send the quote by email	owner:job-to-invoice	sources/jobber/quotes.html
```

### research.md template

```
# research — <project>
## Reference products     name, why chosen, saved under sources/<slug>/
## Journeys               one per primary user; numbered steps; each step tagged [U..]
## Size class             S, M or L, and the unit count that justifies it
## Unsourced ideas        things you believe but could not source; not units
## Gate results           G1 and G2 with numbers and date, then the word FROZEN
```

## Phase 4: Gates, then freeze

Run the two research gates. Record both in `research.md` and as a ledger row.

- **G1 Count.** `units_found` is inside the size class range. Too few means the
  research is thin: go back to step 3.1 with a different kind of source
  (help center, app store reviews, changelog). Too many means the class is wrong:
  propose the next class up at approval time.
- **G2 Walkthrough.** Every journey step is tagged with at least one unit that
  exists in `units.tsv`, and every unit's source file exists on disk.

When both pass, write FROZEN under gate results and commit `research: frozen`.
The third gate, G3 coverage, belongs to the plan and is checked by the scorer forever after.

## Phase 5: Plan

1. **Stages.** Stage 1 is the spine: the thinnest end-to-end path through the
   primary journey. Later stages add units in order of how often the journey
   touches them. Every unit lands in exactly one stage. Do not invent a stage of
   extras nobody asked for.
2. **Chunks.** Inside a stage, group units into chunks one agent can finish in
   one sitting, 45 minutes expected or less. Each chunk owns a disjoint set of
   files. Ownership is what lets agents build in parallel without colliding.
3. **Checks before code.** For every chunk write the `Done when` lines now.
   `exec` lines come first and are commands that exit non-zero on failure.
   A `prose` line is a last resort. List the test and script files those commands
   need under `Checks:`; builders may never touch those files.
4. **Baseline.** Run `python projectmds/score.py --static`. Append the ledger row
   `baseline plan`. Commit `plan: baseline`.

### Stage file template

```
# Stage 1 — Spine
Goal: <one line>
Units: U01 U02 U03 U04

### C01 · <chunk name>
Covers: U01 U02
Owns: src/quotes/, app/quotes/
Checks: tests/c01.quotes.test.ts, scripts/check-quote.sh
Done when:
- exec: npm test -- tests/c01.quotes.test.ts
- exec: scripts/check-quote.sh fixtures/job-001.json
- prose: the quote PDF matches the sample layout
Expected: 30
Depends: —
Wave: —
Status: todo
```

`Status` is one of `todo`, `building`, `done`, `blocked`. Only the verifier step
sets `done`, and only after every `exec` line passes.

## Phase 6: Scope approval

This is the second wait and the last one of the run. Present, in 15 lines or fewer:

- size class and `units_found`
- stages with their unit counts, and which ones you propose to chase
- `chunks_total`, `criteria_exec`, `criteria_prose`
- the baseline `defect_score`
- the budget and `max_parallel` from `program.md`

Ask for a yes or for edits. Edits: apply them, re-score, present again.
Yes: tag the commit `scope-approved` and start the loop. Nothing after this
point asks a question. If you run out of ideas, that is not a reason to stop:
re-read the saved sources, try a different kind of source, or combine two
near-miss attempts from the ledger.

## Phase 7: The build loop

Run this for each stage in `stages_to_chase`, in order.

### Pick a wave

Take the largest set of `todo` chunks in the stage whose `Owns` sets are
disjoint and whose `Depends` are all `done`, up to `max_parallel`. Write the
wave number into each chunk's `Wave:` line and set `Status: building`.
Commit `plan: wave N`.

### Verify first

Write or confirm the files under `Checks:` for every chunk in the wave. Run
each chunk's `exec` lines; they must fail now. A check that passes before the
feature exists checks nothing: fix it. Commit `checks: wave N`.

### Build in parallel

Spawn one subagent per chunk, in parallel, and wait for all of them. Each
brief contains: the chunk block verbatim, `program.md`, the stack notes, and
these instructions: write only inside the files you own; never touch files
listed under `Checks:`; run your own `exec` lines before returning; send
long output to `projectmds/logs/<chunk id>-wave<N>.log`; return exactly one line
in the form `C07 done|blocked <files touched>`. Subagents share the working
tree, so disjoint ownership is mandatory, not advisory.

### Commit, check, score, keep or undo

Subagents share one working tree, so first separate their work: for each chunk
in id order, stage only its owned files and commit `C07: <chunk name>`. A commit
that touches a `Checks:` file is reverted on the spot and logged as a discard.
Then evaluate the commits one at a time, in id order, each against the last keep.

1. Run the chunk's `exec` lines, then `fast_suite`, then `python projectmds/score.py`.
2. Compare with the last keep:
   - lower `defect_score`: keep; set `Status: done`.
   - equal score with fewer `chunks_total`, or equal chunks with lower
     `plan_tokens_k`: keep.
   - anything else: `git revert --no-edit <that commit>`; `Status: todo`.
3. Append one ledger row. Crashes get rows too.

### When it breaks

- A small cause, such as a typo, a missing import or a wrong path: fix it in a
  follow-up commit and re-run. At most two retries per chunk per round.
- A wrong approach: ledger `discard` with one line of why, revert, and try a
  different approach next round.
- A subagent running past twice its `Expected` minutes: stop it, restore its
  owned files with `git checkout -- <files>`, ledger `crash`.
- Three failures on one chunk: `Status: blocked`, ledger row, and the wave
  carries on. Revisit blocked chunks once at the end of the stage with a fresh
  approach; if still blocked they stay blocked and keep counting in the score.
- A chunk is never fixed by weakening its checks.

### Rounds and the wave end

A round is one pass of build and evaluate over the chunks still `todo` or
`building` in the wave. Stop the wave after `max_rounds_per_wave` rounds or when
every chunk is `done` or `blocked`. Then:

1. **Simplicity pass.** Look for two chunks that should be one, or a chunk whose
   units are already covered elsewhere. Deleting it with no loss of coverage is
   a win worth its own commit, scored with `--static`.
2. Rewrite `handoff.md`: stage, wave, last keep commit, blocked chunks, the next
   wave you would pick. A fresh session reads the ledger tail and this file first.
3. Pick the next wave.

### Context hygiene

Load one stage file at a time, plus `program.md`, `units.tsv`, and the last
20 ledger rows. Never load every stage file at once. Never paste a log into
the main thread; read at most its last 50 lines.

## Phase 8: Finish

When every chunk in the approved stages is `done` or `blocked`: run the full
test suite and `score.py` one last time, append the ledger row `final`, commit,
and push the branch if a remote exists. Do not open a pull request unless asked.
Report in plain language: the score from baseline to final, units covered out
of found, chunks done and blocked with the reason for each block, how many prose
criteria remain, and the exact command that runs the app.

## Ledger format

Tab separated, five columns, header row first. `cover` is `units_mapped / units_found`.

```
commit	score	cover	status	what
a1b2c3d	14	0.62	keep	baseline plan
b2c3d4e	9	0.81	keep	wave 1: spine for U01-U06
c3d4e5f	11	0.81	discard	C04 split auth into 3 chunks; more prose criteria
d4e5f6a	—	—	crash	C05 swapped ORM mid-wave; migrations broke
e5f6a7b	7	0.90	keep	deleted C09 and C11, same coverage
```

Status is `keep`, `discard`, or `crash`. Gate results and freezes get rows too.

## Scorer spec

`projectmds/score.py` reads only the files below and prints one `key: value`
per line, `defect_score` first. It never edits anything and always exits 0.

Inputs: `program.md` for `size_class`, `units.tsv`, every `stage-*.md`.
A missing input counts as empty, so the scorer runs on day one against no plan at all.

| Key | How it is computed |
|---|---|
| `units_found` | data rows in `units.tsv` |
| `units_mapped` | distinct unit ids that appear in any chunk's `Covers:` line and exist in `units.tsv` |
| `gates_failed` | +1 if `units_found` is outside the size class range; +1 if any unit's source file is missing; +1 if `units_found > units_mapped` |
| `chunks_total` | count of `### C` headings across stage files |
| `criteria_exec` | count of `- exec:` lines |
| `criteria_prose` | count of `- prose:` lines |
| `checks_failing` | `exec` lines of chunks with `Status: done` or `building` that exit non-zero, each run with a timeout of twice the chunk's `Expected` minutes; 0 under `--static` |
| `chunks_blocked` | chunks with `Status: blocked` |
| `plan_tokens_k` | total characters of all stage files divided by 4000, one decimal |
| `waves` | highest `Wave:` number seen |

```
defect_score = (units_found - units_mapped)
             + 3 * gates_failed
             + criteria_prose
             + checks_failing
             + chunks_blocked
```

Tie-breaks for keep or undo, in order: fewer `chunks_total`, then lower `plan_tokens_k`.

Size class ranges: S 8 to 20, M 20 to 60, L 60 to 150 units.

## Running this in Codex

- Start with `$appbuilder <one-line idea>`.
- The loop is meant to run unattended, so start Codex in a mode that does not
  stop for approvals on every command, such as `--full-auto` or a `codex exec`
  run. Check the flags of your installed version.
- Subagents are on by default in current Codex releases. Phrase the fan-out
  literally: "spawn one agent per chunk, in parallel, wait for all, each returns
  one line." Subagents inherit the parent's sandbox and approval settings.
- If `max_parallel` is larger than the session's concurrent-thread cap in
  `~/.codex/config.toml`, the cap wins; lower `max_parallel` to match.
