#!/usr/bin/env bash
# Remove everything this project put into a temporary GPU workspace:
# the repo folder (incl. .env with your token and outputs/), Hugging Face caches and the pip packages
# installed for it. Only user-level files are touched; system packages are left alone.
#
# Usage: bash scripts/cleanup_gpu_workspace.sh [-y] [--keep-repo]
#   -y           skip the confirmation prompt
#   --keep-repo  keep the repo folder (still deletes .env and the caches/packages)

set -u

PACKAGES="transformers peft datasets accelerate huggingface-hub tokenizers safetensors hf-xet httpx2 httpcore2 truststore multiprocess dill xxhash torch"

main() {
  local yes=0 keep_repo=0
  for arg in "$@"; do
    case "$arg" in
      -y) yes=1 ;;
      --keep-repo) keep_repo=1 ;;
      *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
  done

  local repo
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ ! -f "$repo/train.py" ] || [ ! -f "$repo/publish.py" ]; then
    echo "Refusing to run: $repo does not look like the text2sql-lora-finetune repo." >&2
    exit 1
  fi

  echo "This will delete:"
  [ "$keep_repo" -eq 0 ] && echo "  - $repo (including .env and outputs/)"
  [ "$keep_repo" -eq 1 ] && echo "  - $repo/.env"
  echo "  - ~/.cache/huggingface and /voc/work/.cache/huggingface"
  echo "  - pip packages: $PACKAGES"
  if [ "$yes" -eq 0 ]; then
    read -r -p "Continue? [y/N] " answer
    case "$answer" in y|Y|yes) ;; *) echo "Aborted."; exit 0 ;; esac
  fi

  rm -f "$repo/.env"
  rm -rf "$HOME/.cache/huggingface" /voc/work/.cache/huggingface

  echo "Uninstalling pip packages..."
  # shellcheck disable=SC2086
  pip uninstall -y $PACKAGES 2>&1 | grep -E "Successfully|WARNING: Skipping" || true

  unset HF_TOKEN HUGGINGFACE_API_KEY

  if [ "$keep_repo" -eq 0 ]; then
    cd "$HOME" && rm -rf "$repo"
  fi

  echo
  echo "Done. Remaining user-level pip packages:"
  pip list --user 2>/dev/null | head -n 15
  echo
  echo "Next: revoke the Hugging Face token at https://huggingface.co/settings/tokens if you no longer need it."
}

main "$@"
exit $?
