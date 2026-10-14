#!/usr/bin/env bash
# Cut a release: bump the version, commit, tag and push.
#
# Pushing the `v<version>` tag triggers .github/workflows/release.yml, which
# builds the distributions and creates the GitHub release.
#
# Usage: scripts/release.sh [--test] [--yes] <version>
#   --test   run the integration tests before committing (CI runs them anyway)
#   --yes    do not ask for confirmation before pushing

set -euo pipefail

usage() {
    echo "Usage: $0 [--test] [--yes] <version>    e.g. $0 2.0.1" >&2
    exit 1
}

run_tests=false
assume_yes=false
version=""

for arg in "$@"; do
    case "$arg" in
        --test) run_tests=true ;;
        --yes | -y) assume_yes=true ;;
        -h | --help) usage ;;
        -*) echo "Unknown option: $arg" >&2; usage ;;
        *) [[ -z "$version" ]] || usage; version="${arg#v}" ;;
    esac
done

[[ -n "$version" ]] || usage
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-]?(a|b|rc|dev|post)[0-9]*)?$ ]] \
    || { echo "Not a valid version: $version" >&2; exit 1; }

tag="v$version"
cd "$(git rev-parse --show-toplevel)"

# --- Preconditions -----------------------------------------------------------

branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" == "main" ]] || { echo "Releases are cut from main (on: $branch)" >&2; exit 1; }

[[ -z "$(git status --porcelain)" ]] || { echo "Working tree is not clean" >&2; exit 1; }

git fetch --quiet --tags origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] \
    || { echo "main is not in sync with origin/main" >&2; exit 1; }

if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    echo "Tag $tag already exists" >&2
    exit 1
fi

current="$(sed -nE 's/^__version__ = "([^"]+)"/\1/p' src/corr_vars/__init__.py)"
echo "Releasing $current -> $version"

# --- Bump --------------------------------------------------------------------

today="$(date +%Y-%m-%d)"
perl -pi -e "s/^__version__ = \".*\"/__version__ = \"$version\"/" src/corr_vars/__init__.py
perl -pi -e "s/^version: .*/version: $version/; s/^date-released: .*/date-released: '$today'/" CITATION.cff

git --no-pager diff --stat

if $run_tests; then
    uv run --group test pytest -n auto
fi

# --- Commit, tag, push -------------------------------------------------------

if ! $assume_yes; then
    read -r -p "Commit, tag $tag and push to origin? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        git checkout -- src/corr_vars/__init__.py CITATION.cff
        echo "Aborted; version bump reverted."
        exit 1
    fi
fi

git commit --quiet -am "Release $tag"
git tag -a "$tag" -m "$tag"
git push --quiet origin main
git push --quiet origin "$tag"

echo "Pushed $tag. Follow the release workflow with: gh run watch"
