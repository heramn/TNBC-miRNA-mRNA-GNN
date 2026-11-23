#!/usr/bin/env bash
# Build a Zenodo-ready reproducibility archive locally. It is NOT uploaded or published.
#   bash study/make_repro_package.sh    -> results/study_final/repro_package/tnbc_study_final_<sha>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --short HEAD)
OUT=results/study_final/repro_package
STAGE=$(mktemp -d)/tnbc_study_final_${SHA}
mkdir -p "$STAGE" "$OUT"
git archive --format=tar HEAD | tar -x -C "$STAGE"
for f in predictions_final.csv.gz gat_runs_summary.csv.gz selected_configs.csv; do
  cp "results/study_final/$f" "$STAGE/results/study_final/" 2>/dev/null || true
done
python -m pip freeze > "$STAGE/ENVIRONMENT_pip_freeze.txt" 2>/dev/null || true
cp study/zenodo_metadata.json "$STAGE/.zenodo.json"
(cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -C "$(dirname "$STAGE")" -czf "$OUT/tnbc_study_final_${SHA}.tar.gz" "$(basename "$STAGE")"
sha256sum "$OUT/tnbc_study_final_${SHA}.tar.gz" | tee "$OUT/tnbc_study_final_${SHA}.tar.gz.sha256"
