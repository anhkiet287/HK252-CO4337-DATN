#!/bin/sh
set -eu

if [ -z "${REPO_ROOT:-}" ]; then
  echo "REPO_ROOT must be set before sourcing scripts/qa/common.sh" >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python}

qa_section() {
  printf '\n== %s ==\n' "$1"
}

qa_run() {
  printf '+ %s\n' "$*"
  "$@"
}

qa_assert_exists() {
  if [ ! -e "$1" ]; then
    echo "[FAIL] expected path does not exist: $1" >&2
    exit 1
  fi
}

qa_assert_file_contains() {
  file_path=$1
  expected=$2
  if ! grep -F "$expected" "$file_path" >/dev/null 2>&1; then
    echo "[FAIL] expected '$expected' in $file_path" >&2
    exit 1
  fi
}

qa_expect_fail_contains() {
  log_path=$1
  expected=$2
  shift 2
  if "$@" >"$log_path" 2>&1; then
    echo "[FAIL] command unexpectedly succeeded: $*" >&2
    cat "$log_path" >&2
    exit 1
  fi
  qa_assert_file_contains "$log_path" "$expected"
}
