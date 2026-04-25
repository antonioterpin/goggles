# Retrospectives

Postmortem-style write-ups of bugs that taught us something non-obvious
about Goggles. Each entry should answer four questions:

1. **What went wrong** — observable symptom, ideally a metric.
2. **Root cause** — what was actually happening underneath.
3. **Fix** — what we changed and where the regression test lives.
4. **Lesson** — what to look for next time, or what assumption we
   were making that turned out to be wrong.

Link the retrospective from the regression test docstring so the
historical context is one click away from the test that guards it.

## Index

| Date | Title | PR | Issue |
|---|---|---|---|
| 2026-04-20 | [Shutdown drained `BYE` before its own send queue](2026-04-shutdown-bye-flush.md) | [#143](https://github.com/antonioterpin/goggles/pull/143) | [#158](https://github.com/antonioterpin/goggles/issues/158) |
