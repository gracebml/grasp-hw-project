#!/usr/bin/env bash
# Download the checkpoints (baseline/ and grat/, seed 123) into GRAT_CKPT_ROOT, then serve the demo on the Space's port.
set -euo pipefail
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["CKPT_REPO"], local_dir=os.environ["GRAT_CKPT_ROOT"])
EOF
cd /home/user/app/demo
# The Space shows the app inside an iframe; without these flags Streamlit refuses file uploads there.
exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0 \
  --server.enableXsrfProtection false --server.enableCORS false
