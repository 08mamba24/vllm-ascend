#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/collect_310p_mla_debug_info.sh [options]

Collects minimal debugging info for 310P MLA generation issues.

Options:
  --model MODEL_PATH_OR_ID   Optional, run tokenizer checks with this model.
  --prompt TEXT              Optional, run tokenizer checks with this prompt.
  --script FILE              Optional, extract prompt/sampling hints from a python script.
  --repo DIR                 Optional, force git info collection from this repo.
  --base-ref GIT_REF         Optional, compare local changes against this ref (default: origin/main).
  --out FILE                 Output file path (default: ./mla_310p_debug_YYYYmmdd_HHMMSS.txt)
  -h, --help                 Show this help.
USAGE
}

MODEL=""
PROMPT=""
OUT=""
SCRIPT=""
REPO=""
BASE_REF="origin/main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"; shift 2 ;;
    --prompt)
      PROMPT="${2:-}"; shift 2 ;;
    --out)
      OUT="${2:-}"; shift 2 ;;
    --script)
      SCRIPT="${2:-}"; shift 2 ;;
    --repo)
      REPO="${2:-}"; shift 2 ;;
    --base-ref)
      BASE_REF="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="./mla_310p_debug_$(date +%Y%m%d_%H%M%S).txt"
fi

resolve_repo_root() {
  local candidate=""
  if [[ -n "$REPO" && -d "$REPO" ]]; then
    candidate="$REPO"
  elif git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel
    return 0
  elif [[ -n "$SCRIPT" && -f "$SCRIPT" ]]; then
    candidate="$(cd "$(dirname "$SCRIPT")" && pwd)"
  elif [[ -f "tools/collect_310p_mla_debug_info.sh" ]]; then
    candidate="$(pwd)"
  fi

  if [[ -n "$candidate" ]] && git -C "$candidate" rev-parse --show-toplevel >/dev/null 2>&1; then
    git -C "$candidate" rev-parse --show-toplevel
  fi
}

REPO_ROOT="$(resolve_repo_root || true)"

{
  echo "=== 310P MLA Debug Info ==="
  echo "timestamp: $(date -Is)"
  echo "cwd: $(pwd)"
  echo

  echo "== Git =="
  if [[ -n "$REPO_ROOT" ]]; then
    echo "repo_root: $REPO_ROOT"
    git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null | awk '{print "repo_head: "$0}' || true
    git -C "$REPO_ROOT" branch --show-current 2>/dev/null | awk '{print "repo_branch: "$0}' || true
    git -C "$REPO_ROOT" status --short 2>/dev/null || true
    if git -C "$REPO_ROOT" rev-parse --verify "$BASE_REF" >/dev/null 2>&1; then
      echo
      echo "base_ref: $BASE_REF"
      echo "commits_ahead_of_base:"
      git -C "$REPO_ROOT" log --oneline "$BASE_REF"..HEAD 2>/dev/null || true
      echo "changed_files_vs_base:"
      git -C "$REPO_ROOT" diff --name-status "$BASE_REF"..HEAD 2>/dev/null || true
    else
      echo
      echo "base_ref: $BASE_REF (not found)"
      echo "hint: pass --base-ref <reachable_ref> (e.g. main.oc or origin/main)"
    fi
  else
    echo "repo_root: <not-detected>"
    echo "hint: pass --repo /path/to/vllm-ascend"
  fi
  echo

  echo "== Runtime =="
  python3 - <<'PY'
import importlib
import platform
import sys

print(f"python: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")
for name in ["torch", "torch_npu", "vllm", "vllm_ascend"]:
    try:
        m = importlib.import_module(name)
        print(f"{name}: {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{name}: <import failed> {type(e).__name__}: {e}")
PY
  echo

  echo "== vllm_ascend module resolution =="
  python3 - <<'PY'
try:
    import vllm_ascend
    print(f"module_file: {getattr(vllm_ascend, '__file__', '<unknown>')}")
    print(f"module_version: {getattr(vllm_ascend, '__version__', '<missing __version__>')}")
except Exception as e:
    print(f"module_import_error: {type(e).__name__}: {e}")
PY
  echo

  echo "== Python package details =="
  python3 - <<'PY'
import subprocess
import sys

for pkg in ["torch", "torch_npu", "vllm", "vllm-ascend", "transformers"]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", pkg], stderr=subprocess.STDOUT, text=True)
        first = [line for line in out.splitlines() if line.startswith(("Name:", "Version:", "Location:", "Editable project location:"))]
        print(f"[{pkg}]")
        for line in first:
            print(f"  {line}")
    except Exception:
        print(f"[{pkg}] <pip show failed>")
PY
  echo

  echo "== Env (VLLM/ASCEND/NPU/HCCL/CANN/PYTORCH_NPU) =="
  env | sort | grep -E '^(VLLM|ASCEND|HCCL|CANN|PYTORCH_NPU|NPU)_' || true
  echo

  echo "== Critical 310P MLA env toggles =="
  for k in \
    VLLM_USE_V1 \
    VLLM_PLUGINS \
    VLLM_WORKER_MULTIPROC_METHOD \
    VLLM_ASCEND_DEBUG_KV_CACHE \
    VLLM_ASCEND_310P_MLA_KV_CACHE \
    VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT \
    VLLM_ASCEND_310P_MLA_ATTN_BACKEND
  do
    printf "%s=%s\n" "$k" "${!k-<unset>}"
  done
  echo

  echo "== Quick sanity checks =="
  if [[ "${VLLM_ASCEND_310P_MLA_KV_CACHE-}" != "" && "${VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT-}" != "" ]]; then
    if [[ "${VLLM_ASCEND_310P_MLA_KV_CACHE}" != "${VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT}" ]]; then
      echo "WARN: VLLM_ASCEND_310P_MLA_KV_CACHE and VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT are both set with different values."
      echo "      Keep only VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT to avoid ambiguity."
    else
      echo "INFO: Both MLA KV cache env keys are set to the same value (${VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT})."
      echo "      Prefer keeping only VLLM_ASCEND_310P_MLA_KV_CACHE_FORMAT."
    fi
  fi

  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES-}" && -n "${ASCEND_VISIBLE_DEVICES-}" ]]; then
    if [[ "${ASCEND_RT_VISIBLE_DEVICES}" != "${ASCEND_VISIBLE_DEVICES}" ]]; then
      echo "WARN: ASCEND_RT_VISIBLE_DEVICES differs from ASCEND_VISIBLE_DEVICES."
      echo "      Ensure worker pinning matches your TP/world-size expectations."
    fi
  fi

  python3 - <<'PY'
try:
    import torch
    ver = getattr(torch, "__version__", "unknown")
    if "+cpu" in str(ver):
        print("WARN: torch version contains '+cpu'; verify this build is expected for torch_npu runtime.")
except Exception as e:
    print(f"INFO: skip torch sanity check ({type(e).__name__}: {e})")
PY
  echo

  echo "== Suggested fields to attach =="
  echo "1) startup command"
  echo "2) sampling params: temperature/top_p/top_k/repetition_penalty/max_tokens/stop/ignore_eos"
  echo "3) full prompt and chat template"
  echo "4) eos_token_id / stop_token_ids / special token table"
  echo "5) first 50 generated token ids"
  echo

  if [[ -n "$MODEL" && -n "$PROMPT" ]]; then
    echo "== Tokenizer check =="
    MODEL="$MODEL" PROMPT="$PROMPT" python3 - <<'PY'
import os

model = os.environ["MODEL"]
prompt = os.environ["PROMPT"]

try:
    from transformers import AutoTokenizer
except Exception as e:
    print(f"transformers import failed: {type(e).__name__}: {e}")
    raise SystemExit(0)

try:
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
except Exception as e:
    print(f"load tokenizer failed: {type(e).__name__}: {e}")
    raise SystemExit(0)

ids = tok.encode(prompt, add_special_tokens=False)
print(f"model: {model}")
print(f"prompt_chars: {len(prompt)}")
print(f"prompt_token_count: {len(ids)}")
print(f"prompt_tail_100: {prompt[-100:]!r}")
print(f"token_ids_tail_64: {ids[-64:]}")
print(f"eos_token: {tok.eos_token!r}")
print(f"eos_token_id: {tok.eos_token_id}")
print(f"all_special_tokens: {tok.all_special_tokens}")
print(f"all_special_ids: {tok.all_special_ids}")
PY
  fi

  if [[ -n "$SCRIPT" ]]; then
    echo
    echo "== Script hints (${SCRIPT}) =="
    if [[ ! -f "$SCRIPT" ]]; then
      echo "script_not_found: $SCRIPT"
    else
      SCRIPT_PATH="$SCRIPT" python3 - <<'PY'
import ast
import os
from pathlib import Path

path = Path(os.environ["SCRIPT_PATH"])
text = path.read_text(encoding="utf-8")
print(f"script: {path}")

keys = [
    "apply_chat_template",
    "SamplingParams",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_tokens",
    "stop",
    "ignore_eos",
]
for i, line in enumerate(text.splitlines(), start=1):
    if any(k in line for k in keys):
        print(f"L{i}: {line.strip()}")

if path.suffix.lower() == ".py":
    try:
        mod = ast.parse(text)
        for node in ast.walk(mod):
            if isinstance(node, ast.Call) and getattr(getattr(node.func, "id", None), "lower", lambda: "")() == "samplingparams":
                print("sampling_params_kwargs:")
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if isinstance(kw.value, ast.Constant):
                        print(f"  {kw.arg}={kw.value.value!r}")
                    else:
                        print(f"  {kw.arg}=<non-constant>")
    except Exception as e:
        print(f"ast_parse_error: {type(e).__name__}: {e}")
else:
    print("ast_parse_skipped: non-python script")
PY
    fi
  fi
} | tee "$OUT"

echo "Saved debug info to: $OUT"
