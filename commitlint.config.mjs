// Commitlint config for goggles.
//
// Inherits the standard `@commitlint/config-conventional` rules but
// skips the merge commits GitHub creates when a PR with multiple
// commits is merged (`Merge pull request ... from ...`) and the
// merge commits `git merge` itself produces. Without this, every
// merge-commit-style PR into main fails the commit-lint workflow.
//
// The body/footer line-length limits are disabled: footers carry
// unwrappable trailers (long `Co-authored-by` emails, tool-injected
// `Agent-Logs-Url:` URLs) that legitimately exceed 100 characters,
// and the workflow re-lints merged history on every downstream push,
// so a single long trailer would otherwise poison future merges.

// Squash-merge subjects already in main/dev history that predate
// conventional-title enforcement for bot PRs (#229, #230). History
// cannot be rewritten, and the workflow re-lints history on every
// push (force-pushes lint deep history), so these exact first lines
// are permanently exempted. Exact match only: new commits stay fully
// linted. When squash-merging a PR, the PR title becomes the commit
// subject -- keep it conventional so this list never grows.
const LEGACY_SQUASH_SUBJECTS = [
    'Pin Pillow to patched version to unblock pip-audit workflow (#229)',
    'Resolve pip-audit CI failure by upgrading Pillow to a patched' +
        ' version (#230)',
];

export default {
    extends: ['@commitlint/config-conventional'],
    ignores: [
        (message) =>
            /^Merge (pull request|branch|remote-tracking branch) /.test(
                message,
            ),
        (message) =>
            LEGACY_SQUASH_SUBJECTS.includes(message.split('\n')[0]),
    ],
    rules: {
        'body-max-line-length': [0],
        'footer-max-line-length': [0],
    },
};
