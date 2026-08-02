#!/usr/bin/env python3
"""Fail on files or metadata that should not enter the public Git history."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
REQUIRED = {
    ".github/workflows/quality.yml",
    ".gitattributes",
    ".gitignore",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "checkpoints/manifest.json",
    "docs/GITHUB_RELEASE_CHECKLIST.md",
    "docs/REPRODUCTION.md",
    "docs/REPOSITORY_MAP.md",
    "requirements.txt",
}
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "checkpoints",
    "configs",
    "data",
    "docs",
    "external",
    "outputs",
    "projects",
    "requirements.txt",
    "scripts",
    "tests",
}
PLACEHOLDER_DIR_FILES = {
    "checkpoints/README.md",
    "checkpoints/manifest.json",
    "data/README.md",
    "external/README.md",
    "outputs/README.md",
}
PRIVATE_SUFFIXES = {".ckpt", ".onnx", ".pem", ".pt", ".pth", ".safetensors"}
AGENT_FILES = {".claude.json", ".mcp.json", "AGENTS.md", "CLAUDE.md"}
AGENT_DIRECTORIES = {".agents", ".claude", ".codex"}
LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9_$])/(?:data|home|Users)/[A-Za-z0-9._-]+/")
WINDOWS_USER_PATH = re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+\\")
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def public_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(item for item in result.stdout.decode().split("\0") if item)


def main() -> int:
    files = public_files()
    file_set = set(files)
    errors: list[str] = []
    warnings: list[str] = []

    for required in sorted(REQUIRED - file_set):
        errors.append(f"required public file is missing: {required}")

    for relative in files:
        path = ROOT / relative
        parts = Path(relative).parts
        top_level = relative.split("/", 1)[0]
        if top_level not in ALLOWED_TOP_LEVEL:
            errors.append(f"unexpected top-level public path: {relative}")
        if path.name in AGENT_FILES or any(
            part in AGENT_DIRECTORIES for part in parts
        ):
            errors.append(f"agent-local file is public: {relative}")
        if relative.startswith(("legacy/", "technical_report/")):
            errors.append(f"separately managed path is public: {relative}")
        if relative.startswith(("data/", "external/", "outputs/")):
            if relative not in PLACEHOLDER_DIR_FILES:
                errors.append(f"local artifact is public: {relative}")
        if relative.startswith("checkpoints/") and relative not in PLACEHOLDER_DIR_FILES:
            errors.append(f"checkpoint artifact is public: {relative}")
        if path.is_symlink():
            errors.append(f"symbolic link is not allowed: {relative}")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() in PRIVATE_SUFFIXES:
            errors.append(f"model or credential artifact is public: {relative}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"unexpected binary public file: {relative}")
            continue
        if LOCAL_PATH.search(text) or WINDOWS_USER_PATH.search(text):
            errors.append(f"machine-specific absolute path found: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} pattern found: {relative}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                errors.append(f"invalid JSON in {relative}: {error}")
        if path.suffix.lower() == ".md":
            for target in MARKDOWN_LINK.findall(text):
                target = target.split("#", 1)[0].strip().strip("<>")
                if not target or "://" in target or target.startswith(("#", "mailto:")):
                    continue
                resolved_target = (path.parent / target).resolve()
                if not resolved_target.exists():
                    errors.append(f"broken Markdown link in {relative}: {target}")
                    continue
                try:
                    public_target = resolved_target.relative_to(ROOT).as_posix()
                except ValueError:
                    errors.append(
                        f"Markdown link escapes repository in {relative}: {target}"
                    )
                    continue
                if resolved_target.is_file() and public_target not in file_set:
                    errors.append(
                        f"Markdown link targets a local-only file in {relative}: "
                        f"{target}"
                    )
                if resolved_target.is_dir() and not any(
                    item.startswith(f"{public_target}/") for item in file_set
                ):
                    errors.append(
                        f"Markdown link targets a local-only directory in {relative}: "
                        f"{target}"
                    )

    manifest_path = ROOT / "checkpoints/manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in manifest.get("artifacts", []):
            digest = artifact.get("sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                errors.append(f"invalid checkpoint hash: {artifact.get('id', '<unknown>')}")
            if artifact.get("role") == "final-inference" and not artifact.get("url"):
                warnings.append("final checkpoint URL has not been published")

    if "LICENSE" not in file_set:
        warnings.append("repository license has not been selected")

    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(f"public repository check passed: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
