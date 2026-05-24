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
export default {
    extends: ['@commitlint/config-conventional'],
    ignores: [
        (message) =>
            /^Merge (pull request|branch|remote-tracking branch) /.test(
                message,
            ),
    ],
    rules: {
        'body-max-line-length': [0],
        'footer-max-line-length': [0],
    },
};
