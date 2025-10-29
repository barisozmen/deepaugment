#!/usr/bin/env python3
"""
Publish package to PyPI. One command, zero friction.

Philosophy (Norvig):
- Minimal, elegant, correct
- Fail fast with clear errors
- Read version from single source of truth
- Composable Unix tools
"""

import re
import subprocess
import sys
from pathlib import Path


def run(cmd, check=True):
    """Run command. Return output or exit on error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"✗ Command failed: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def version():
    """Extract version from pyproject.toml."""
    content = Path("pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else None


def check_git_clean():
    """Ensure no uncommitted changes."""
    status = run("git status --porcelain", check=False)
    if status:
        print("✗ Uncommitted changes. Commit first:", file=sys.stderr)
        print(status, file=sys.stderr)
        sys.exit(1)


def clean():
    """Remove build artifacts."""
    print("→ Cleaning build artifacts...")
    run("rm -rf dist/ build/ *.egg-info", check=False)
    run("find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true", check=False)


def build():
    """Build wheel and sdist."""
    print("→ Building package...")
    run("uv build")


def tag_and_push(ver):
    """Create and push git tag."""
    tag = f"v{ver}"
    print(f"→ Creating tag {tag}...")
    run(f'git tag -a {tag} -m "Release {tag}"', check=False)  # ok if exists
    run(f"git push origin {tag}", check=False)


def publish():
    """Publish to PyPI."""
    print("→ Publishing to PyPI...")
    run("uv publish")


def main():
    """Orchestrate: check → clean → build → tag → publish."""
    print("📦 Publishing DeepAugment to PyPI\n")

    ver = version()
    if not ver:
        print("✗ Cannot read version from pyproject.toml", file=sys.stderr)
        sys.exit(1)

    print(f"Version: {ver}\n")

    check_git_clean()
    clean()
    build()
    tag_and_push(ver)
    publish()

    print(f"\n✓ Published deepaugment {ver}")
    print(f"→ https://pypi.org/project/deepaugment/{ver}/")


if __name__ == "__main__":
    main()
