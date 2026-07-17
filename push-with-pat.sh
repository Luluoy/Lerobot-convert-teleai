#!/usr/bin/env bash
set -Eeuo pipefail

# Git invokes this script through GIT_ASKPASS. Keep the token out of argv,
# shell history, configured remotes, and credential helpers.
if [[ -n "${LEROBOT_GITHUB_PAT:-}" ]]; then
  case "${1:-}" in
    *Username*)
      printf '%s\n' "x-access-token"
      exit 0
      ;;
    *Password*)
      printf '%s\n' "$LEROBOT_GITHUB_PAT"
      exit 0
      ;;
  esac
fi

usage() {
  cat <<'EOF'
Usage: ./push-with-pat.sh [remote] [remote-branch]

Push the current checkout's HEAD once with a GitHub PAT entered through a
hidden terminal prompt. Defaults: remote=origin, remote-branch=current branch.

The script does not commit changes, store credentials, modify remotes, or set
an upstream branch. The worktree must be clean before it will prompt for a PAT.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 2 )); then
  usage >&2
  exit 2
fi

SCRIPT_PATH=$(realpath -- "$0")
PROJECT_DIR=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
REMOTE=${1:-origin}

if ! git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This script must remain inside a Git worktree." >&2
  exit 1
fi

CURRENT_BRANCH=$(git -C "$PROJECT_DIR" branch --show-current)
if [[ -z "$CURRENT_BRANCH" ]]; then
  echo "Detached HEAD is not supported. Check out the branch to push first." >&2
  exit 1
fi
BRANCH=${2:-$CURRENT_BRANCH}
if ! git check-ref-format --branch "$BRANCH" >/dev/null 2>&1; then
  echo "Invalid remote branch name: $BRANCH" >&2
  exit 1
fi

if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]]; then
  echo "The worktree has uncommitted changes. Commit them before pushing." >&2
  git -C "$PROJECT_DIR" status --short >&2
  exit 1
fi

if ! RAW_URL=$(git -C "$PROJECT_DIR" remote get-url --push "$REMOTE" 2>/dev/null); then
  echo "Git remote not found: $REMOTE" >&2
  exit 1
fi

case "$RAW_URL" in
  https://github.com/*)
    REPOSITORY=${RAW_URL#https://github.com/}
    ;;
  git@github.com:*)
    REPOSITORY=${RAW_URL#git@github.com:}
    ;;
  ssh://git@github.com/*)
    REPOSITORY=${RAW_URL#ssh://git@github.com/}
    ;;
  https://*@github.com/*)
    echo "The configured remote contains user information. Replace it with a credential-free GitHub URL first." >&2
    exit 1
    ;;
  *)
    echo "Only github.com remotes are supported: $RAW_URL" >&2
    exit 1
    ;;
esac

REPOSITORY=${REPOSITORY%/}
REPOSITORY=${REPOSITORY%.git}
if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "Could not determine GitHub owner/repository from: $RAW_URL" >&2
  exit 1
fi
PUSH_URL="https://github.com/$REPOSITORY.git"

if [[ ! -r /dev/tty || ! -w /dev/tty ]]; then
  echo "An interactive terminal is required to enter the PAT securely." >&2
  exit 1
fi

printf 'GitHub PAT (input hidden): ' >/dev/tty
if ! IFS= read -r -s LEROBOT_GITHUB_PAT </dev/tty; then
  printf '\n' >/dev/tty
  echo "Could not read the PAT from the terminal." >&2
  exit 1
fi
printf '\n' >/dev/tty
if [[ -z "$LEROBOT_GITHUB_PAT" ]]; then
  echo "PAT must not be empty." >&2
  exit 1
fi

cleanup() {
  unset LEROBOT_GITHUB_PAT GIT_ASKPASS GIT_ASKPASS_REQUIRE GIT_TERMINAL_PROMPT
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

export LEROBOT_GITHUB_PAT
export GIT_ASKPASS="$SCRIPT_PATH"
export GIT_ASKPASS_REQUIRE=force
export GIT_TERMINAL_PROMPT=0

echo "[1/3] Verifying GitHub destination: $REPOSITORY"
git -C "$PROJECT_DIR" -c credential.helper= ls-remote "$PUSH_URL" >/dev/null

echo "[2/3] Pushing HEAD to $REMOTE/$BRANCH"
git -C "$PROJECT_DIR" -c credential.helper= push "$PUSH_URL" "HEAD:refs/heads/$BRANCH"

echo "[3/3] Verifying remote branch SHA"
LOCAL_SHA=$(git -C "$PROJECT_DIR" rev-parse HEAD)
REMOTE_LINE=$(git -C "$PROJECT_DIR" -c credential.helper= ls-remote "$PUSH_URL" "refs/heads/$BRANCH")
read -r REMOTE_SHA _ <<<"$REMOTE_LINE"
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
  echo "Push verification failed: local=$LOCAL_SHA remote=${REMOTE_SHA:-missing}" >&2
  exit 1
fi

echo "Push verified: $REPOSITORY $BRANCH $LOCAL_SHA"
