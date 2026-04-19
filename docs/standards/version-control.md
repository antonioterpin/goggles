# Version control discipline

## Commit strategy

- **Commit code often**, in small, logically coherent units.
- Do **not** commit unfinished or broken code.
- **Always follow the `.gitmessage` template** for commit messages.
- Configure git once with:
  ```bash
  git config commit.template .gitmessage
  ```
- Goggles uses [Conventional Commits](https://www.conventionalcommits.org/).
  The `commit-lint` GitHub Action rejects non-conforming titles.

## Branch workflow

- Feature branches start from `main` (this repo does not currently use a
  `dev` branch -- see `CONTRIBUTING.md` for the canonical description).
- Open a PR back to `main`; tooling workflows (code-style, tests) must
  be green before merge.
- Squash on merge to keep history linear.

## Remote repository

- **Do not push to remote on behalf of the user.** PRs are opened only
  when explicitly requested, and never force-pushed without explicit
  authorization.
