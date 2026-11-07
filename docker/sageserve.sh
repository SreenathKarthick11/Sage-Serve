# Loaded by every interactive shell in the container.

mlops() { deactivate 2>/dev/null || true; cd "$MLOPS_DIR" && source /opt/venvs/mlops/bin/activate; }
sim()   { deactivate 2>/dev/null || true; cd "$SIM_DIR"   && source /opt/venvs/sim/bin/activate; }
root()  { deactivate 2>/dev/null || true; cd "$REPO"; }

sageserve-help() {
cat <<'EOF'
SageServe dev container
  mlops | sim | root          switch directory + venv
  registry <args>             model registry CLI      (e.g. registry list)
  pipeline <args>             pipeline runner         (e.g. pipeline --source static)
  latest <source> <stage>     latest artifact path    (e.g. latest streaming forecasts)
  run-sim <source>            simulator on latest trace + forecasts
  inference                   inference entry point
  sageserve-help              show this again
EOF
}

sageserve-help