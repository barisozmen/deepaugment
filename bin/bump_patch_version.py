#!/usr/bin/env python3
"""
Automatically bump patch version in pyproject.toml if user didn't change it.

Single Source of Truth: Version lives ONLY in pyproject.toml.

Philosophy:
- Do one thing well (Unix)
- Fail fast with clear messages
- Optimize for programmer happiness
- Beautiful, minimal code (Norvig style)
- Smart: Sync with remote tags to avoid conflicts
"""

import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    """Run command, return output. Return None on error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def parse_version(version_str):
    """Parse version string to tuple (major, minor, patch)."""
    match = re.match(r'v?(\d+)\.(\d+)\.(\d+)', version_str)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def version_to_str(major, minor, patch):
    """Convert version tuple to string."""
    return f"{major}.{minor}.{patch}"


def get_remote_tags():
    """Fetch remote tags and return highest version. None if no remote/network issues."""
    # Fetch tags from remote (quietly)
    run("git fetch --tags -q 2>/dev/null")

    # Get all tags (local + fetched remote)
    tags_output = run("git tag -l 'v*'")
    if not tags_output:
        return None

    # Parse all version tags
    versions = []
    for tag in tags_output.split('\n'):
        v = parse_version(tag)
        if v:
            versions.append(v)

    # Return highest version
    return max(versions) if versions else None


def get_current_version():
    """Extract version from pyproject.toml - the SSOT."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("✗ pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    content = pyproject.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)

    if not match:
        print("✗ No version found in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    version_str = match.group(1)
    version_tuple = parse_version(version_str)

    if not version_tuple:
        print(f"✗ Invalid version format: {version_str}", file=sys.stderr)
        sys.exit(1)

    return version_tuple


def version_changed_in_staged():
    """Check if version line in pyproject.toml is staged for commit."""
    diff = run("git diff --cached pyproject.toml")
    if not diff:
        return False

    # Look for version = "x.y.z" changes in staged diff
    return bool(re.search(r'^\+version\s*=\s*["\']', diff, re.MULTILINE))


def bump_patch(version):
    """Bump patch version: (2, 0, 5) -> (2, 0, 6)"""
    major, minor, patch = version
    return (major, minor, patch + 1)


def sync_with_remote(local_version):
    """
    Sync with remote tags. If remote has higher version, start from there.
    Returns the version to use as baseline.
    """
    remote_version = get_remote_tags()

    # No remote tags or fetch failed - use local
    if not remote_version:
        return local_version

    # Remote is ahead - sync to remote version
    if remote_version > local_version:
        print(f"→ Remote version ({version_to_str(*remote_version)}) is ahead, syncing...")
        return remote_version

    # Local is ahead or equal - use local
    return local_version


def update_version_in_file(new_version):
    """Update version in pyproject.toml."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    version_str = version_to_str(*new_version)

    # Replace version line (match any existing version)
    new_content = re.sub(
        r'^version\s*=\s*["\'][^"\']+["\']',
        f'version = "{version_str}"',
        content,
        count=1,
        flags=re.MULTILINE
    )

    if new_content == content:
        print(f"✗ Could not update version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    pyproject.write_text(new_content)
    return version_str


def main():
    """Main flow: Sync with remote, check user changes, bump if needed."""

    # If user already updated version, do nothing
    if version_changed_in_staged():
        print("→ Version manually updated in pyproject.toml, skipping auto-bump")
        return

    # Get current local version (SSOT)
    local_version = get_current_version()

    # Sync with remote (handles fresh pulls, clones)
    baseline_version = sync_with_remote(local_version)

    # Bump patch from baseline
    new_version = bump_patch(baseline_version)

    # Update pyproject.toml
    new_version_str = update_version_in_file(new_version)

    # Stage the change
    run("git add pyproject.toml")

    # Create annotated tag
    old_version_str = version_to_str(*baseline_version)
    tag_msg = f"Auto-bump patch: {old_version_str} -> {new_version_str}"
    run(f'git tag -a v{new_version_str} -m "{tag_msg}"')

    # Pretty output
    if baseline_version != local_version:
        print(f"✓ Synced from remote and bumped: {old_version_str} -> {new_version_str}")
    else:
        print(f"✓ Bumped version: {old_version_str} -> {new_version_str}")


if __name__ == "__main__":
    main()
