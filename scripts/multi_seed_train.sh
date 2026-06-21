#!/bin/bash
# Run 5 parallel fischer training seeds, pick the best afterward.
# Each takes ~45-60 min. With 5 in parallel, CPU will be shared so expect ~2-3 hours.
# Run from project root:
#   bash scripts/multi_seed_train.sh

PYTHON="/Users/ranjotsandhu/Documents/Project/venv/bin/python"
SCRIPT="/Users/ranjotsandhu/Documents/Project/scripts/train_fischer_session.py"
BASE_DIR="/Users/ranjotsandhu/Documents/Project/project_assets/outputs"
WAYPOINTS="2.3,-1.0;1.0,1.5;2.5,0.0"

echo "Launching 5 parallel training seeds..."
echo "Each run: 500K steps, SAC, muscle actuation, EAST direction"
echo "Estimated time: 2-3 hours (CPU shared across 5 processes)"
echo ""

for seed in 1 2 3 4 5; do
    DIR="${BASE_DIR}/multiseed_run_${seed}"
    echo "Starting seed ${seed} -> ${DIR}"
    $PYTHON $SCRIPT \
        --timesteps 500000 \
        --actuation-mode muscle \
        --goal-direction EAST \
        --waypoints "$WAYPOINTS" \
        --save-dir "$DIR" \
        > "${DIR}_stdout.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All 5 seeds launched. Monitor with:"
echo "  ps aux | grep train_fischer"
echo ""
echo "When done, run:"
echo "  $PYTHON scripts/pick_best_seed.py"
