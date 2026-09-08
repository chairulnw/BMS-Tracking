#!/bin/bash
# Eval OSNet vs TransReID dengan config LAMA (kode app/ = e725842^, HEAD saat
# benchmark pipeline-comparison.md pertama jalan). Stash perubahan working-tree,
# checkout kode lama, run, lalu pulihkan.
set -e
cd "$(dirname "$0")/.."   # ai-service/
OLD=e725842^

echo ">> stash + checkout $OLD -- app/"
git stash push -m "eval_oldcfg" -- app/ || { echo "stash gagal / nothing to stash"; }
STASHED=$(git stash list | grep -c "eval_oldcfg" || true)
git checkout "$OLD" -- app/

restore() {
  echo ">> restore app/"
  git checkout HEAD -- app/
  if [ "$STASHED" -gt 0 ]; then git stash pop; fi
}
trap restore EXIT

echo ">> run eval (oldcfg)"
python benchmark/run_eval_current.py oldcfg
