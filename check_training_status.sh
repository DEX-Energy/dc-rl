#!/bin/bash
# Quick training status checker for SustainDC
# Usage: ./check_training_status.sh [experiment_name]

EXPERIMENT=${1:-"spot_gpu_final"}
RESULTS_DIR="results/sustaindc/ny/happo/$EXPERIMENT"

echo "========================================="
echo "SustainDC Training Status Checker"
echo "========================================="
echo ""

# Check if experiment exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Experiment '$EXPERIMENT' not found in $RESULTS_DIR"
    exit 1
fi

# Find the latest seed directory
SEED_DIR=$(ls -td $RESULTS_DIR/seed-* 2>/dev/null | head -1)

if [ -z "$SEED_DIR" ]; then
    echo "❌ No seed directories found"
    exit 1
fi

echo "📁 Experiment: $EXPERIMENT"
echo "📁 Directory: $SEED_DIR"
echo ""

# Check if training is running
RUNNING=$(ps aux | grep "train_sustaindc.*$EXPERIMENT" | grep -v grep | wc -l | tr -d ' ')
if [ "$RUNNING" -gt 0 ]; then
    echo "🟢 Status: RUNNING"
else
    echo "🔴 Status: NOT RUNNING (completed or stopped)"
fi
echo ""

# Check progress file
PROGRESS_FILE="$SEED_DIR/progress.txt"
if [ -f "$PROGRESS_FILE" ]; then
    LINES=$(wc -l < "$PROGRESS_FILE" | tr -d ' ')
    LAST_LINE=$(tail -1 "$PROGRESS_FILE")

    # Extract last timestep (first number in last line)
    LAST_STEP=$(echo "$LAST_LINE" | grep -oE '^[0-9]+' | head -1)

    if [ ! -z "$LAST_STEP" ]; then
        # Calculate progress
        TOTAL_STEPS=10000
        EPISODE_LENGTH=168

        TOTAL_EPISODES=$(echo "scale=1; $TOTAL_STEPS / $EPISODE_LENGTH" | bc)
        CURRENT_EPISODE=$(echo "scale=1; $LAST_STEP / $EPISODE_LENGTH" | bc)
        PROGRESS_PCT=$(echo "scale=1; ($LAST_STEP / $TOTAL_STEPS) * 100" | bc)
        REMAINING=$(echo "scale=1; $TOTAL_EPISODES - $CURRENT_EPISODE" | bc)

        echo "📊 Progress:"
        echo "   Last step: $LAST_STEP / $TOTAL_STEPS"
        echo "   Episode: $CURRENT_EPISODE / $TOTAL_EPISODES"
        echo "   Progress: ${PROGRESS_PCT}%"
        echo "   Remaining: $REMAINING episodes"
        echo ""

        # Show last 3 entries
        echo "📈 Last 3 episodes:"
        tail -5 "$PROGRESS_FILE" | grep -E '^[0-9]' | tail -3 | while read line; do
            step=$(echo "$line" | cut -d',' -f1)
            reward=$(echo "$line" | cut -d',' -f2)
            ep=$(echo "scale=0; $step / $EPISODE_LENGTH" | bc)
            echo "   Episode $ep (step $step): reward = $reward"
        done
    fi
else
    echo "❌ Progress file not found: $PROGRESS_FILE"
fi

echo ""
echo "========================================="
echo "Quick Commands:"
echo "========================================="
echo "Monitor live:     tail -f $PROGRESS_FILE"
echo "Check process:    ps aux | grep train_sustaindc"
echo "TensorBoard:      tensorboard --logdir=$RESULTS_DIR"
echo "========================================="
