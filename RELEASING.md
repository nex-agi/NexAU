# Release Process

This document describes how to create a new release of the NexAU framework.

## Overview

The project uses automated GitHub Actions workflows for continuous deployment (CD). When you push a version tag, the workflow will:

1. Run all tests and linting
2. Build the package
3. Create a GitHub release with changelog

## Prerequisites

## Release Steps

### 1. Prepare the Release

```bash
# Ensure you're on the main branch and up to date
git checkout main
git pull origin main

# Run tests locally to verify everything works
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format --check .
```

### 2. Create and Push a Version Tag

```bash
# Create a new version tag (e.g., v0.2.0)
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push the tag to GitHub
git push origin v0.2.0
```

### 3. Monitor the Release

1. Go to the [Actions tab](https://github.com//nexau/actions)
2. Watch the "CD - Release and Publish" workflow
3. The workflow will:
   - Run tests
   - Build the package
   - Create a GitHub release

### 4. Verify the Release

- Check [GitHub Releases](https://github.com//nexau/releases) for the release notes
- Test installation: `pip install nexau==0.2.0`

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (v1.0.0): Incompatible API changes
- **MINOR** version (v0.2.0): New functionality, backwards compatible
- **PATCH** version (v0.1.1): Backwards compatible bug fixes

## Hotfix Releases

For urgent fixes to a released version:

```bash
# Create a hotfix branch from the tag
git checkout -b hotfix/v0.1.1 v0.1.0

# Make your fixes and commit
git add .
git commit -m "Fix critical bug"

# Create and push the hotfix tag
git tag -a v0.1.1 -m "Release version 0.1.1 - Hotfix"
git push origin v0.1.1
```

## Rollback

If you need to remove a release:

1. Delete the GitHub release from the [Releases page](https://github.com//nexau/releases)
2. Delete the tag:
   ```bash
   git tag -d v0.2.0
   git push origin :refs/tags/v0.2.0
   ```

## Troubleshooting

### Tests Fail in Workflow

- Run tests locally first: `uv run pytest`
- Check the specific test failure in the workflow logs
- Fix the issue and create a new tag

### Version Mismatch

The workflow automatically updates `pyproject.toml` with the version from the tag. The version in the file is only used for local development.

## Continuous Integration

All pushes and PRs to `main` and `develop` branches trigger the CI workflow:
- Linting with ruff
- Format checking with ruff  
- Full test suite with pytest
- Coverage reporting

Make sure CI passes before creating a release tag.

