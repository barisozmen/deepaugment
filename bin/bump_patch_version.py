#!/usr/bin/env python3
"""
Automatically bump patch version in pyproject.toml if user didn't change it.

Single Source of Truth: Version lives ONLY in pyproject.toml.

Philosophy:
- Do one thing well (Unix)
- Fail fast with clear messages
- Optimize for programmer happiness
- Beautiful, minimal code (Norvig style)
"""

import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    """Run command, return output. Fail fast on error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


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

    return match.group(1)


def version_changed_in_staged_pyproject_toml():
    """Check if version line in pyproject.toml is staged for commit."""
    diff = run("git diff --cached pyproject.toml")
    if not diff:
        return False

    # Look for version = "x.y.z" changes in staged diff
    return bool(re.search(r'^\+version\s*=\s*["\']', diff, re.MULTILINE))


def bump_patch(version):
    """Bump patch version: 2.0.5 -> 2.0.6"""
    parts = version.split(".")
    if len(parts) != 3:
        print(f"✗ Invalid version format: {version}", file=sys.stderr)
        sys.exit(1)

    major, minor, patch = parts
    try:
        new_patch = int(patch) + 1
    except ValueError:
        print(f"✗ Invalid patch number: {patch}", file=sys.stderr)
        sys.exit(1)

    return f"{major}.{minor}.{new_patch}"


def update_version_in_pyproject_toml(old_version, new_version):
    """Update version in pyproject.toml."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()

    # Replace version line
    new_content = re.sub(
        rf'^version\s*=\s*["\']({re.escape(old_version)})["\']',
        f'version = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE
    )

    if new_content == content:
        print(f"✗ Could not update version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    pyproject.write_text(new_content)


def main():
    """Main flow: Check if user changed version, if not, bump it."""

    # If user already updated version, do nothing
    if version_changed_in_staged_pyproject_toml():
        print("-> Version manually updated in pyproject.toml, skipping auto-bump")
        return

    # Get current version (SSOT)
    old_version = get_current_version()

    # Bump patch
    new_version = bump_patch(old_version)

    update_version_in_pyproject_toml(old_version, new_version)

    # Stage the change
    run("git add pyproject.toml")

    # Create annotated tag
    tag_msg = f"Auto-bumped patch: {old_version} -> {new_version}"
    run(f'git tag -a v{new_version} -m "{tag_msg}"')

    # Pretty output
    print(f"✓ Bumped version: {old_version} -> {new_version}")


if __name__ == "__main__":
    main()
