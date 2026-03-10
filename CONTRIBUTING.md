# How to Contribute to the Polar Pandas

This document outlines the branching strategy, naming conventions, and workflow for all developers on the team. Following these rules is mandatory to keep our project clean and prevent conflicts.

## 1. Branching Workflow

We use a "GitFlow-Lite" model with two main branches: `main` and `develop`.

* `main`: This branch is **sacred**. It represents the official, stable, production-ready version. No one ever pushes or merges directly to `main`.
* `develop`: This is our main development branch. All new features are merged into `develop`.

**Your Workflow for Every Task:**
1.  Get the latest `develop` branch:
    ```bash
    git checkout develop
    git pull origin develop
    ```
2.  Create your new feature branch from `develop`:
    ```bash
    git checkout -b feature/my-new-task
    ```
3.  Do your work and commit your changes.
4.  When finished, push your feature branch:
    ```bash
    git push -u origin feature/my-new-task
    ```
5.  Go to GitHub and create a **Pull Request (PR)**.
6.  Set the **Target Branch** to **`develop`**.

## 2. Naming Conventions

### Branch Names
Use the format `type/short-description`.
* `feature/string-literal-tokenization`
* `fix/lexer-infinite-loop`
* `docs/update-contributing`

### Commit Messages
We use **Conventional Commits**. This is the standard format: `type(scope): subject`.

* **type:** `feat` (new feature), `fix` (bug fix), `docs` (documentation), `refactor`, `style`.
* **scope (optional):** The part of the app (e.g., `api`, `ui`, `auth`).

**Examples:**
* `feat(lexer): add support for multi-line string literals`
* `fix(parser): resolve shift-reduce conflict in expression grammar`
* `docs: update branching workflow in CONTRIBUTING.md`