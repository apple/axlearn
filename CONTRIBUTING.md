# Contributing to AXLearn

## Coding Style

We follow [the Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and [ML API Styles](docs/ml_api_style.md) unless otherwise stated.

Respect code consistency in the repo, including but not limited to naming conventions and code structure.

If you have not already, we recommend [setting up `pre-commit`](docs/01-start.md#optional-additional-setup-for-developers), which runs some of the linters/formatters prior to each commit. The same checks will be required to pass in CI, so this will help make the development process smoother.

### Type Annotations

Functions and methods must be annotated with types. We apply [pytype](https://google.github.io/pytype/user_guide.html) for type checking.

### Spurious Lint Failures

`pylint` and `pytype` are not perfect, and sometimes report false positives/negatives. If you believe a check is failing spuriously and plan to disable a lint check for the corresponding line(s) of code, please tag the original author for review.

## Testing

All new logic should have corresponding unittests. Design the API so that dependencies can be
injected for testing.

If changing existing functionality, tests should cover both the existing and new codepaths, to avoid losing coverage.

If applying a bugfix, tests should explicitly exercise the codepath(s) triggering the bug, to avoid regressions.

Try to [parameterize](https://github.com/abseil/abseil-py/blob/0ff1e24e9486900a895af805e58f4e468ec5edf7/absl/testing/parameterized.py#L17) tests where appropriate to avoid copy/pasting testing logic.

## Code Review Protocol

### Author

Before embarking on a major PR, send out a sketch PR (including the high level design notes in the PR description) to solicit feedback first.

Prefer small PRs that can be quickly reviewed and merged.

When selecting multiple reviewers, use "Assignees" to indicate that approvals from specific
reviewers are required before merging.

The PR authors are expected to reply to each comment they have addressed, e.g., with "Done".
However, they should refrain from resolving the comments -- instead, let the reviewers do so.

When addressing a comment, pay attention to other places where the same comment may apply, so that
the reviewer does not have to repeat the comment.

When a PR is ready for another round of review, click [**re-request review**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) to notify the reviewer.

Although not strictly enforced, a general etiquette is to wait for all reviewers who have left comments to approve the PR prior to merging, even if the PR has already received an approval.

In some cases, the comments may be "nits" and appropriate for addressing in follow-up PRs. We encourage authors to give a heads up to reviewers prior to merging a PR with unresolved comments by leaving PR comments (e.g. "Will address in a follow-up") as well as [leaving a TODO](#leaving-todos) where applicable.

### Reviewer

People on the team should feel free to add themselves as reviewers and/or to assignees.

Consider prioritizing reviews over writing one's own code.
This will make the entire team more productive.

Code review does not end with merge of the PR.
Reviewers should feel free to add comments after the merge, which can be addressed in follow-up PRs.

## Attributions

Code that refers to (or is adapted from) other sources must explicitly reference the original source by providing a [link in the docstring](https://github.com/apple/axlearn/blob/669f0cae6249e165caa1a94cf64b12e77bf4cfdf/axlearn/common/attention.py#L360-L365) of the corresponding function, class, etc.

Additionally, include the copyright notice of the original source in the [header of the same file](https://github.com/apple/axlearn/blob/669f0cae6249e165caa1a94cf64b12e77bf4cfdf/axlearn/common/attention.py#L5-L7), and (if not already present) in the [ACKNOWLEDGEMENTS.md](https://github.com/apple/axlearn/blob/main/ACKNOWLEDGEMENTS.md) file.

## Leaving TODOs

In some cases it's useful to track future work ("TODOs").
TODOs should have the format `TODO(username1,username2)` indicating the contributor(s) responsible for addressing the TODO.
Please use your actual Github username as opposed to an alias to avoid ambiguity.

For larger work items, consider creating a Github issue to track progress.

## Dependencies

### Updating PyProject

All dependencies should be tracked in `pyproject.toml`. Dependencies are grouped into different sections, the main ones being:
```
dependencies: Essential dependencies for AXLearn to function.
apple-silicon: Essential dependencies for AXLearn to function on Apple Silicon Macs.
dev: Dependencies for testing and development.
...
```

See the [Getting Started](docs/01-start.md) for details on installing dependencies.

When adding a new dependency, think about whether it can exist as an optional dependency, and which section it falls into.
If a dependency does not belong to an existing section, or if it introduces a large number of transitive dependencies, consider including it in a new section.
If an optional dependency is also required for testing and development, consider also adding it to `dev`.
