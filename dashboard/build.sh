#!/usr/bin/env bash
# Rebuild the dashboard data. Called at the end of every lab/research/import/pipeline run
# and by a Coolify Scheduled Task as a safety net. Never fails the caller.
DATA_DIR="${DATA_DIR:-/data}"
exec 6>"$DATA_DIR/.dashboard.lock"
flock -w 120 6 || exit 0
python /app/dashboard/build_dashboard.py >>"$DATA_DIR/logs/dashboard.log" 2>&1 || \
  echo "$(date -u +%FT%TZ) dashboard build failed" >>"$DATA_DIR/logs/dashboard.log"
exit 0
