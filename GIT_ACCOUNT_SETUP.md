# Git Account Setup

This project should be published with the intended GitHub account only. The current machine has a global Git identity configured, so use local repository settings before committing or pushing.

## Current Risk

From this folder, Git currently resolves the repository root as:

```bash
/home/sara/projects
```

That parent repository contains many sibling projects, so this project should be isolated before publishing.

## Safe Setup Used Here

The normal `.git` path in this folder is read-only in the current workspace, so this project uses a separate local Git directory:

```bash
cd /home/sara/projects/ResearchGraph/multihop-rag
git --git-dir=.git-local --work-tree=. status
```

The configured local identity is:

```bash
git --git-dir=.git-local --work-tree=. config user.name
git --git-dir=.git-local --work-tree=. config user.email
```

Commit and push with:

```bash
git --git-dir=.git-local --work-tree=. add .
git --git-dir=.git-local --work-tree=. commit -m "Initial ResearchGraph-RAG implementation"
git --git-dir=.git-local --work-tree=. push -u origin main
```

## Notes

- Do not rely on the global Git identity for this repository.
- Do not push until the remote points to the alternate GitHub account.
- If using HTTPS instead of SSH, make sure the token belongs to the alternate account.
