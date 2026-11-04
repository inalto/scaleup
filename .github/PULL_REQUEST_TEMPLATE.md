## What

<!-- One or two sentences: what changes and why. Link the issue if there is one. -->

## How to test

<!-- Commands, images or settings a reviewer can use to see the change working. -->

## Checklist

- [ ] `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
      `cargo test --workspace` pass locally
- [ ] New behaviour has a test; GPU-only tests are `#[ignore]`d
- [ ] `CHANGELOG.md` (`Unreleased`) updated for user-visible changes
- [ ] Docs updated (`README.md`, `docs/USER_GUIDE.md`, crate READMEs) where behaviour changed
- [ ] New models come with a license file in `models/licenses/` and a manifest entry
