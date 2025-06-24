# Git Workflow Rules

## Feature Development Workflow

For each new project, capability, or feature, use git worktree and create a dedicated feature branch:

```bash
git worktree add ../project-feat-NAME_OF_FEATURE -b feat/NAME_OF_FEATURE
```

Branch naming convention: `feat/NAME_OF_FEATURE` where NAME_OF_FEATURE is descriptive and uses kebab-case.

## Commit Frequency and Messages

Create a git commit after every phase or task completion using semantic commit messages.

Follow conventional commit format: `type(scope): description`

### Commit Types:
- `feat`: new feature or capability
- `fix`: bug fix
- `docs`: documentation changes
- `style`: formatting, whitespace, etc.
- `refactor`: code restructuring without behavior change
- `test`: adding or modifying tests
- `chore`: maintenance, build process, or tooling changes
- `perf`: performance improvements
- `ci`: continuous integration changes

### Examples:
```
feat(auth): implement OAuth2 login flow
fix(api): resolve timeout issues in user service
docs(readme): add installation instructions
refactor(utils): extract common validation logic
test(user): add unit tests for user registration
```

## Worktree Management

Switch to feature worktree for development:
```bash
cd ../project-feat-NAME_OF_FEATURE
```

Keep main worktree clean and switch back when needed:
```bash
cd ../project
```

Remove worktree after feature completion:
```bash
git worktree remove ../project-feat-NAME_OF_FEATURE
```

## Merge Strategy

Merge feature branches back to main using:
- Pull requests for code review
- Squash commits if multiple small commits exist
- Preserve meaningful commit history for significant changes