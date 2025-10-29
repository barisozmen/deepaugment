#!/usr/bin/env python3
"""
Push git tag when version changes in pyproject.toml.

Post-commit hook: Detects version changes and creates/pushes tags.

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
    """Run command, return output. Return None on error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def pyproject_changed_in_last_commit():
    """Check if pyproject.toml was modified in the last commit."""
    files = run("git diff HEAD~1 HEAD --name-only")
    return files and "pyproject.toml" in files.split('\n')


def version_changed_in_last_commit():
    """Check if version field changed in pyproject.toml in last commit."""
    diff = run("git diff HEAD~1 HEAD -- pyproject.toml")
    if not diff:
        return False
    return bool(re.search(r'^[+-]version\s*=\s*', diff, re.MULTILINE))


def get_current_version():
    """Extract version from pyproject.toml."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        return None

    content = pyproject.read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def create_and_push_tag(version):
    """Create annotated tag and push to remote."""
    tag = f"v{version}"

    # Create annotated tag
    result = run(f'git tag -a "{tag}" 2>&1')
    if result is None or "already exists" in result:
        print(f"→ Tag {tag} already exists, skipping")
        return False

    # Push tag to remote
    push_result = run("git push --tags 2>&1")
    if push_result is None or "error" in push_result.lower():
        print(f"→ Could not push tags (no remote or network issue)")
        return False

    print(f"✓ Tagged and pushed {tag}")
    return True


def main():
    """Main flow: Check if version changed, create and push tag."""

    # Check if pyproject.toml changed
    if not pyproject_changed_in_last_commit():
        sys.exit(0)

    # Check if version field specifically changed
    if not version_changed_in_last_commit():
        sys.exit(0)

    # Get current version
    version = get_current_version()
    if not version:
        print("✗ Could not extract version from pyproject.toml", file=sys.stderr)
        sys.exit(1)

    # Create and push tag
    create_and_push_tag(version)


if __name__ == "__main__":
    main()
