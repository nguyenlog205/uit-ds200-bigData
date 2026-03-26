# Contributing Documentation
This document outlines the workflow and best practices to ensure smooth collaboration.

## 1. Getting Started

1. **Fork the repository** (if you don't have write access) or clone directly.
2. **Create a branch** from `main` with a descriptive name:
```bash
git checkout -b <your-branch>
```
Use prefixes: `feature/`, `fix/`, `docs/`, `refactor/`.

## 2. Keeping Your Branch Up‑to‑Date
Merge conflicts are minimized by frequently pulling the latest changes from main into your branch:
```bash
git fetch origin
git checkout main
git pull origin main
git checkout <your-branch>
git merge main
```

## 3. Git push
```bash
git add <path-to-file>
git commit -m '<type of change>: <Description>
git push 
```
 - **Type of change**: `feat`, `fix`, `docs`, `pytest-cicd`, `refactor`