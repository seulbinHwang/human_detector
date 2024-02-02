#!/bin/bash

# Initializes the development workspace.
# Please make sure that this script is idempotent, so that running this
# script multiple times is safe.

check_command() {
  local tool=$1
  if [ -x "$(command -v "${tool}")" ]; then
    return 0
  fi
  return 1
}

check_pip() {
  if ! check_command pip; then
    echo "Error: pip is not installed."
    echo ""
    return 1
  fi

  return 0
}

check_git_lfs() {
  # Check git lfs command is available.
  if ! check_command git-lfs; then
    echo "Error: Git LFS not installed. You need to install Git LFS to finish"
    echo "initialization."
    echo ""
    echo "For macOS, run:"
    echo "$ brew install git-lfs"
    echo ""
    echo "For Ubuntu, run:"
    echo "$ sudo apt-get install git-lfs"
    return 1
  fi
}

get_platform() {
  # Get platform name. e.g. linux_x86_64, osx_x86_64.
  platform="$(uname -s)_$(uname -m)"
  echo ${platform} | tr '[:upper:]' '[:lower:]'
}

#############################################################################
# Main script
#############################################################################

# Die on error.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
TOP_DIR=$(git rev-parse --show-toplevel)
GIT_DIR=$(git rev-parse --git-dir)
PLATFORM=$(get_platform)
readonly SCRIPT_DIR
readonly TOP_DIR
readonly GIT_DIR
readonly PLATFORM

echo "============================================================"
echo "This script initializes your repository."
echo
echo "It is safe to run this script multiple times, so run this "
echo "script frequently to make sure your environment is up-to-date."
echo "============================================================"
echo

check_pip
pip install -r "${SCRIPT_DIR}"/requirements.txt > /dev/null

check_git_lfs

# Make rebase the default policy pulling.
git config --local pull.rebase preserve

for src_hook_path in "${SCRIPT_DIR}"/templates/git/hooks/*; do
  hook_name=$(basename "${src_hook_path}")
  target_hook=${GIT_DIR}/hooks/${hook_name}
  if [[ -f ${target_hook} ]]; then
    if [[ ! -h ${target_hook} || \
        "$(readlink "${target_hook}")" != "${src_hook_path}" ]]; then
      echo >&2 "Warning: ${hook_name} hook already exists."
      echo >&2
      echo >&2 "Skip installing ${hook_name}."
      echo >&2 "For correct operation, please remove your existing hook ${target_hook} and rerun this script."
      echo >&2
    fi
  else
    echo "Installing ${hook_name} hook."
    ln -s "${src_hook_path}" "${target_hook}"
  fi
done

# Try to install formatter & linter configurations
for src_lintrc_path in "${SCRIPT_DIR}"/templates/linters/*; do
  lintrc_name=".$(basename "${src_lintrc_path}")"
  target_lintrc="${TOP_DIR}/${lintrc_name}"
  if [[ -f ${target_lintrc} ]]; then
    if [[ ! -h ${target_lintrc} || \
        "$(readlink "${target_lintrc}")" != "${src_lintrc_path}" ]]; then
      echo >&2 "Warning: ${lintrc_name} already exists."
      echo >&2
      echo >&2 "Skip installing ${lintrc_name}."
      echo >&2 "For correct operation, please remove your existing ${target_lintrc} and rerun this script."
      echo >&2
    fi
  else
    echo "Installing ${lintrc_name}."
    ln -s "${src_lintrc_path}" "${target_lintrc}"
  fi
done

# Check if git lfs hooks are installed.
if ! grep -q 'git lfs post-checkout' "${GIT_DIR}/hooks/post-checkout"; then
  echo "Installing git-lfs hooks... (git lfs install --local)"
  if ! git lfs install --local; then
    echo >&2 "Error: Install failed. Maybe you have custom git hooks installed?"
    echo >&2 ""
    echo >&2 "Please follow the descriptions above to resolve the issues, and then re-run"
    echo >&2 "this script to resume initialization."
    exit 1
  fi
fi

echo "Pulling Git LFS files..."
git lfs pull

# Configure platform specific locked requirements.txt
requirements_name="requirements_lock.txt"
src_requirements="${TOP_DIR}/requirements_${PLATFORM}.txt"
target_requirements="${TOP_DIR}/${requirements_name}"
if [[ ! -f ${src_requirements} ]]; then
  touch ${src_requirements}
fi

if [[ -f ${target_requirements} ]]; then
  if [[ ! -h ${target_requirements} || \
      "$(readlink ${target_requirements})" != "${src_requirements}" ]]; then
    echo >&2 "Warning: ${requirements_name} already exists."
    echo >&2
    echo >&2 "Skip installing ${requirements_name}."
    echo >&2 "For correct operation, please remove your existing ${target_requirements} and rerun this script."
    echo >&2
  fi
else
  echo "Installing ${requirements_name}."
  ln -s ${src_requirements} ${target_requirements}
fi

echo "Init done."
echo
