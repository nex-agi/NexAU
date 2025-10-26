#!/bin/bash
# Release helper script for Northau framework
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if version is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 0.2.0"
    exit 1
fi

VERSION=$1
TAG="v${VERSION}"

echo -e "${YELLOW}Preparing release ${TAG}${NC}"

# Check if on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: You must be on the main branch to create a release${NC}"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check if working directory is clean
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: Working directory is not clean${NC}"
    echo "Please commit or stash your changes first"
    exit 1
fi

# Pull latest changes
echo -e "${YELLOW}Pulling latest changes...${NC}"
git pull origin main

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag ${TAG} already exists${NC}"
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
if command -v uv &> /dev/null; then
    uv run pytest
else
    pytest
fi

# Run linting
echo -e "${YELLOW}Running linter...${NC}"
if command -v uv &> /dev/null; then
    uv run ruff check .
    uv run ruff format --check .
else
    ruff check .
    ruff format --check .
fi

# Prompt for confirmation
echo -e "${YELLOW}Ready to create release ${TAG}${NC}"
echo "This will:"
echo "  1. Create and push tag ${TAG}"
echo "  2. Trigger GitHub Actions workflow"
echo "  3. Create GitHub release"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Release cancelled${NC}"
    exit 1
fi

# Create and push tag
echo -e "${YELLOW}Creating tag ${TAG}...${NC}"
git tag -a "$TAG" -m "Release version ${VERSION}"

echo -e "${YELLOW}Pushing tag to GitHub...${NC}"
git push origin "$TAG"

echo -e "${GREEN}✓ Release ${TAG} initiated!${NC}"
echo ""
echo "Monitor the release progress at:"
echo "https://github.com/china-qijizhifeng/northau/actions"
echo ""
echo "After the workflow completes, the release will be available at:"
echo "- PyPI: https://pypi.org/project/northau/${VERSION}/"
echo "- GitHub: https://github.com/china-qijizhifeng/northau/releases/tag/${TAG}"

