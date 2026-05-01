// Commitlint config for goggles.
//
// Inherits the standard `@commitlint/config-conventional` rules but
// skips the merge commits GitHub creates when a PR with multiple
// commits is merged (`Merge pull request ... from ...`) and the
// merge commits `git merge` itself produces. Without this, every
// merge-commit-style PR into main fails the commit-lint workflow.
export default {
    extends: ['@commitlint/config-conventional'],
    ignores: [
        (message) =>
            /^Merge (pull request|branch|remote-tracking branch) /.test(
                message,
            ),
    ],
};
