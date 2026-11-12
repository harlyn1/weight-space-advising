#!/bin/bash
# The full test: every lever on the same held-out sets, same prompts, same seeds.
#
#   export ANTHROPIC_API_KEY=sk-ant-...          # Fable 5.1, Opus 5.5 editors; Sonnet 5.5 for the frontier arm and the SFT curation
#   export OPENAI_API_KEY=sk-...                 # GPT-6 Astra editor (the paper's editor); the astra arm is skipped without it
#   bash ~/qwen-0.8b/wsp/run_compare.sh
#
# Arms, in the default order (ARMS="..." to choose):
#   base             the model as is (the floor; every other arm is compared with it, paired by prompt and seed)
#   astra_published  + the paper's loop fix, as published
#   context          + contexts/advising.md as the system prompt (context injection)        ~1 h of Mac time
#   frontier         Sonnet 5.5 + the same context (the capacity ceiling)                   ~$3-5 of API
#   finetune         + a LoRA on answers written by Sonnet 5.5 with the context (arm D)       ~$5 of API + ~1 h of training
#   fable opus astra kimi  + the editor's 40-pass edit on tasks/q2_full.md (weight-space programming)  ~12-14 h each
#                    (kimi = Kimi K3 through Fireworks' OpenAI-compatible endpoint, FIREWORKS_API_KEY)
#   fable_context ...  the editor's edit + the context (do the levers stack)
# Options via env: BUDGET=40 EFFORT=max WSP_MAX_COST_USD=250 (spend cap per editor arm) EVAL_N=2 TASK=tasks/q2_full.md
# FRONTIER_MODEL=claude-sonnet-5-5 CURATE_MODEL=claude-sonnet-5-5 SERVER_BATCH=8 EVAL_BATCH=8 (32 on a GPU box; see gpu/).
# Restartable: an arm with work/eval_<arm>/summary.json is skipped.
set -e
exec < /dev/null          # never depend on the launching terminal: a closed window must not break later Python starts (macOS nohup keeps the tty as stdin)
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi      # keys from wsp/.env (see .env.example)
ARMS=${ARMS:-"base astra_published context frontier finetune fable opus astra kimi fable_context opus_context astra_context kimi_context astra80 astra_xhigh sonnet astra80_context astra_xhigh_context sonnet_context"}
# follow-up arms (queued 2026-11-09): astra80 = Astra resumed from its own final state with 40 more passes (budget 80);
# astra_xhigh = a fresh Astra run at reasoning effort xhigh (skipped if the API rejects it); sonnet = Claude Sonnet 5.5 as the editor, 40 passes
BUDGET=${BUDGET:-40}; EFFORT=${EFFORT:-max}; EVAL_N=${EVAL_N:-2}; TASK=${TASK:-tasks/q2_full.md}
FRONTIER_MODEL=${FRONTIER_MODEL:-claude-sonnet-5-5}; CURATE_MODEL=${CURATE_MODEL:-claude-sonnet-5-5}
SERVER_BATCH=${SERVER_BATCH:-8}; EVAL_BATCH=${EVAL_BATCH:-8}     # generation batch sizes: 8 on the Air, 32 on a 24 GB GPU
HANDOFF=${HANDOFF:-150}     # turns per conversation before the handoff restart (40 cost the editors a quarter of their turns in re-orientation)
export WSP_MAX_COST_USD=${WSP_MAX_COST_USD:-250}
mkdir -p work runs results finetune
PY=.venv/bin/python
if pgrep -f "server.py --budget" > /dev/null; then echo "a forward-pass server from an earlier run is still alive; stopping it"; fi
KEEP=$(command -v caffeinate > /dev/null && echo "caffeinate -i" || echo "")
CTX=contexts/advising.md

model_of() { case "$1" in fable|fable_clean) echo claude-fable-5-1;; opus) echo claude-opus-5-5;; astra|astra80|astra_xhigh) echo gpt-6-astra;; kimi) echo accounts/fireworks/models/kimi-k3;; sonnet) echo claude-sonnet-5-5;; *) echo "$1";; esac; }
budget_of() { case "$1" in astra80) echo 80;; *) echo "$BUDGET";; esac; }
effort_of() { case "$1" in astra_xhigh) echo xhigh;; *) echo "$EFFORT";; esac; }
seed_astra80() {   # Astra's finished 40-pass run becomes the first half of an 80-pass run: same edit, notes and transcript, counter at 40 of 80
  [ -d runs/full_astra80 ] && return 0
  [ -f runs/full_astra/final_answer.md ] || { echo "cannot seed astra80: runs/full_astra has no final answer"; return 1; }
  mkdir -p runs/full_astra80 && cp -R runs/full_astra/turns runs/full_astra80/ && cp runs/full_astra/tool_calls.jsonl runs/full_astra/arm.json runs/full_astra/message.txt runs/full_astra80/
  mkdir -p runs/full_astra80/edited_partial && cp runs/full_astra/edited_final/*.npy runs/full_astra80/edited_partial/
  { cat runs/full_astra/notes.md 2>/dev/null; printf "\n\n## Budget extension\nYour first 40 passes ended with the final answer below. The budget is now 80 passes, so 40 remain: use them to improve the edit (dose, placement, collateral), verifying on recorded inputs before spending passes. The held-out evaluation is unchanged.\n\n### Final answer after 40 passes\n"; cat runs/full_astra/final_answer.md; } > runs/full_astra80/notes_partial.md
  $PY -c "import json; s=json.load(open('runs/full_astra/state.json')); s['budget']=80; json.dump(s, open('runs/full_astra80/state_partial.json','w'), indent=1)"
  echo "seeded runs/full_astra80 from runs/full_astra (40 passes used of 80)"; }
stop_server() {   # by the script's own arguments, not the interpreter name: the venv's python shows as ".../Python" on macOS
  # an editor orphaned by an earlier run (its caffeinate/tee parent gone) would keep driving whatever server is up: stop it first
  if pgrep -f "editor_api.py --task" > /dev/null; then echo "   stopping an editor left over from an earlier run: $(pgrep -fl 'editor_api.py --task' | head -3 | tr '\n' ';')"
    pkill -f "editor_api.py --task" 2>/dev/null || true; sleep 3; pkill -9 -f "editor_api.py --task" 2>/dev/null || true; sleep 1; fi
  pkill -f "server.py --budget" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -f "server.py --budget" > /dev/null || break; sleep 2; done
  pgrep -f "server.py --budget" > /dev/null && { pkill -9 -f "server.py --budget" 2>/dev/null || true; sleep 2; }
  rm -f work/server.pid
}
start_server() { stop_server; if [ "$2" != keep ] && [ -f state.json ]; then mv state.json "work/state_before_$(date +%s).json"; fi
  nohup $KEEP $PY server.py --budget "$1" --batch "$SERVER_BATCH" > work/server.log 2>&1 &
  echo "server starting (pid $!), loading the model..."; sleep 45; tail -2 work/server.log
  grep -q "already running" work/server.log && { echo "ABORT: a second server was running"; exit 1; }; }
reset_edit() { rm -rf edited; mkdir -p edited; cp work/best_paper/*.npy edited/; }
use_edit() { rm -rf edited; mkdir -p edited; cp "runs/full_$1/edited_final/"*.npy edited/ 2>/dev/null || cp work/best_paper/*.npy edited/; }
done_arm() { [ -f "work/eval_$1/summary.json" ]; }

run_editor() {   # $1 = arm (fable|opus|astra)
  local ARM=$1 MODEL RUN; MODEL=$(model_of "$ARM"); RUN="full_$ARM"
  if [ -f "runs/$RUN/final_answer.md" ]; then echo "== editor arm $ARM already ran"; return 0; fi
  local B EF; B=$(budget_of "$ARM"); EF=$(effort_of "$ARM")
  case "$MODEL" in
    gpt*) [ -n "$OPENAI_API_KEY" ] || { echo "skipping $ARM (OPENAI_API_KEY not set)"; return 1; } ;;
    accounts/*) [ -n "$FIREWORKS_API_KEY" ] || { echo "skipping $ARM (FIREWORKS_API_KEY not set)"; return 1; } ;;
    claude*) [ -n "$ANTHROPIC_API_KEY" ] || { echo "skipping $ARM (ANTHROPIC_API_KEY not set)"; return 1; } ;;
  esac
  case "$MODEL" in gpt*|accounts/*) ls .venv/lib/python3*/site-packages 2>/dev/null | grep -qi "^openai-" || { echo "skipping $ARM: run  .venv/bin/pip install openai  first"; return 1; } ;; esac
  [ "$ARM" = astra80 ] && { seed_astra80 || return 1; }
  [ "$EF" = xhigh ] && export WSP_EFFORT_STRICT=1 || unset WSP_EFFORT_STRICT
  if ls "runs/$RUN"/turns/t*/run.json > /dev/null 2>&1; then
    # an interrupted run with real turns: restore its edits and notes, keep the pass counter, continue from the editor's notes
    echo "== editor arm $ARM: resuming an interrupted run ($(ls "runs/$RUN/turns" | wc -l | tr -d ' ') turns so far)"
    if [ -d "runs/$RUN/edited_partial" ]; then rm -rf edited; mkdir -p edited; cp "runs/$RUN/edited_partial/"*.npy edited/ 2>/dev/null || true
    else echo "   (no snapshot of this arm's edits: continuing with what is in edited/ now)"; fi
    [ -f "runs/$RUN/notes_partial.md" ] && cp "runs/$RUN/notes_partial.md" work/notes.md
    [ -f "runs/$RUN/state_partial.json" ] && cp "runs/$RUN/state_partial.json" state.json
    [ -f "work/task_$ARM.md" ] || sed "s/{BUDGET}/$B/g" "$TASK" > "work/task_$ARM.md"
    EXP=$($PY -c "import json; print(json.load(open('state.json'))['passes_used'])" 2>/dev/null || echo "?")
    echo "   restored from runs/$RUN: passes used $EXP, $(ls edited | wc -l | tr -d ' ') edited tensors, notes $( [ -f work/notes.md ] && wc -c < work/notes.md | tr -d ' ' || echo 0) bytes"
    start_server "$B" keep
    grep -q "passes used $EXP," work/server.log || { echo "ABORT: the server did not start from the restored state (expected passes used $EXP): $(grep ready work/server.log | cut -c1-80)"; stop_server; return 1; }
    $KEEP $PY editor_api.py --task "work/task_$ARM.md" --run "$RUN" --model "$MODEL" --effort "$EF" --handoff_every "$HANDOFF" --resume 2>&1 | tee -a "runs/$RUN.log"
  else
    echo "== editor arm $ARM: $MODEL ($EF) on $TASK, $B passes, cap \$$WSP_MAX_COST_USD"
    reset_edit; rm -f work/notes.md; sed "s/{BUDGET}/$B/g" "$TASK" > "work/task_$ARM.md"
    start_server "$B"
    $KEEP $PY editor_api.py --task "work/task_$ARM.md" --run "$RUN" --model "$MODEL" --effort "$EF" --handoff_every "$HANDOFF" 2>&1 | tee -a "runs/$RUN.log"
  fi
  stop_server
  if [ ! -f "runs/$RUN/final_answer.md" ]; then
    rm -rf "runs/$RUN/edited_partial"; mkdir -p "runs/$RUN/edited_partial"; cp edited/*.npy "runs/$RUN/edited_partial/" 2>/dev/null || true
    cp work/notes.md "runs/$RUN/notes_partial.md" 2>/dev/null || true; cp state.json "runs/$RUN/state_partial.json" 2>/dev/null || true
    echo "== editor arm $ARM did not finish (API or server interruption); its notes and edits are kept. Re-run run_compare.sh to resume it."
    return 1
  fi
  rm -rf "runs/$RUN/edited_final"; mkdir -p "runs/$RUN/edited_final"; cp edited/*.npy "runs/$RUN/edited_final/" 2>/dev/null || true
  cp work/notes.md "runs/$RUN/notes.md" 2>/dev/null || true; mv state.json "runs/$RUN/state.json" 2>/dev/null || true
}

evaluate() {   # $1 = arm, rest = eval_heldout.py flags
  local ARM=$1; shift
  stop_server
  echo "== evaluating arm $ARM ($*), n=$EVAL_N"
  $KEEP $PY eval_heldout.py --arm "$ARM" --n "$EVAL_N" --batch "$EVAL_BATCH" "$@" 2>&1 | tee -a "results/eval_$ARM.log"
}

stop_server          # no server from an earlier run or a smoke test may be alive when the arms start
for ARM in $ARMS; do
  if done_arm "$ARM"; then echo "== $ARM already evaluated"; continue; fi
  case "$ARM" in
    base)             evaluate base --weights original ;;
    astra_published)  reset_edit; evaluate astra_published --weights edited ;;
    context)          evaluate context --weights original --context $CTX ;;
    frontier)         [ -n "$ANTHROPIC_API_KEY" ] || { echo "skipping frontier (ANTHROPIC_API_KEY not set)"; continue; }
                      evaluate frontier --api "$FRONTIER_MODEL" --context $CTX ;;
    finetune)         [ -n "$ANTHROPIC_API_KEY" ] || { echo "skipping finetune (ANTHROPIC_API_KEY needed for curation)"; continue; }
                      $PY -c "import peft" 2>/dev/null || { echo "skipping finetune: run  .venv/bin/pip install peft  first"; continue; }
                      [ -f finetune/sft.jsonl ] || $KEEP $PY finetune/curate.py --model "$CURATE_MODEL" 2>&1 | tee -a results/curate.log
                      [ -f finetune/merged/config.json ] || { stop_server; $KEEP $PY finetune/train_lora.py 2>&1 | tee -a results/train.log; }
                      [ -f finetune/merged/config.json ] || { echo "== fine-tune training did not finish (see results/train.log); arm skipped"; continue; }
                      # the fine-tune is evaluated like the editing arms: its changed tensors as edits on the original model (no second model copy)
                      if [ ! -d runs/full_finetune/edited_final ]; then rm -rf edited; mkdir -p edited; $PY finetune/merged_to_edits.py 2>&1 | tee -a results/train.log; fi
                      [ -d runs/full_finetune/edited_final ] || { echo "== fine-tune conversion to edits failed (see results/train.log); arm skipped"; continue; }
                      use_edit finetune; evaluate finetune --weights edited ;;
    fable|fable_clean|opus|astra|kimi|astra80|astra_xhigh|sonnet) run_editor "$ARM" || continue; use_edit "$ARM"; evaluate "$ARM" --weights edited ;;
    *_context)        E=${ARM%_context}; [ -f "runs/full_$E/final_answer.md" ] || { echo "skipping $ARM (editor arm $E not run)"; continue; }
                      use_edit "$E"; evaluate "$ARM" --weights edited --context $CTX ;;
    *)                echo "unknown arm $ARM"; continue ;;
  esac
done

$PY compare_arms.py | tee results/full_test.md
$PY make_report.py && echo "report: results/full_test.html"
echo "done. per-arm details: work/eval_<arm>/summary.md ; editor transcripts: runs/full_<arm>/ ; fine-tune: finetune/"
