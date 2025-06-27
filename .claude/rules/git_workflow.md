# Git Workflow Rules

## MANDATORY: Git Worktree for All Development

**NEVER work directly on the main branch.** You MUST create a git worktree for ALL development activities including:

- New features or capabilities
- Bug fixes (except critical hotfixes)
- Documentation changes
- Refactoring tasks
- Configuration updates
- Rule changes or additions
- Any multi-commit development work

### Creating Feature Worktrees

For each new task, ALWAYS create a dedicated worktree:

```bash
git worktree add ../project-feat-NAME_OF_FEATURE -b feat/NAME_OF_FEATURE
```

Branch naming conventions:
- `feat/NAME_OF_FEATURE` - New features or capabilities
- `fix/NAME_OF_BUG` - Bug fixes
- `docs/NAME_OF_CHANGE` - Documentation updates
- `refactor/NAME_OF_CHANGE` - Code refactoring
- `chore/NAME_OF_TASK` - Maintenance tasks

Use descriptive names with kebab-case (e.g., `feat/changeset-implementation`, `fix/memory-leak-processor`).

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

## Mandatory Worktree Workflow

### 1. Start Every Task with Worktree Creation
```bash
# ALWAYS start with this step - NO EXCEPTIONS
git worktree add ../bg_remove-rs-feat-NAME_OF_FEATURE -b feat/NAME_OF_FEATURE
cd ../bg_remove-rs-feat-NAME_OF_FEATURE
```

### 2. Complete Development in Feature Worktree
- Make ALL commits in the feature worktree
- Update changelog files in the feature branch
- Run tests and validation in the feature branch
- NEVER switch back to main until feature is complete

### 3. Integration Back to Main
```bash
# Switch back to main worktree
cd ../bg_remove-rs

# Merge feature branch
git merge feat/NAME_OF_FEATURE

# Clean up worktree
git worktree remove ../bg_remove-rs-feat-NAME_OF_FEATURE
git branch -d feat/NAME_OF_FEATURE  # Optional: delete merged branch
```

## Worktree Management Rules

### Directory Naming Convention
Use consistent naming: `../bg_remove-rs-feat-NAME_OF_FEATURE`

Examples:
- `../bg_remove-rs-feat-changeset-implementation`
- `../bg_remove-rs-fix-memory-leak`
- `../bg_remove-rs-docs-api-documentation`
- `../bg_remove-rs-refactor-processor-architecture`

### When Worktrees Are Required
- ✅ **ALWAYS**: New features, bug fixes, documentation, refactoring
- ✅ **ALWAYS**: Multi-commit tasks or experimental work
- ✅ **ALWAYS**: Adding new rules or configuration changes
- ❌ **EXCEPTION**: Single typo fixes or emergency hotfixes (use main with immediate commit)

### Worktree Lifecycle
1. **Create**: `git worktree add` with appropriate branch name
2. **Develop**: Work exclusively in the feature worktree
3. **Validate**: Run all tests and checks in feature worktree
4. **Integrate**: Merge to main and remove worktree
5. **Cleanup**: Remove worktree directory and optionally delete branch

## Merge Strategy

### Preferred Merge Methods
- **Small features**: Direct merge to preserve history
- **Large features**: Squash commits for cleaner history
- **Bug fixes**: Direct merge with descriptive commit message
- **Documentation**: Direct merge unless extensive changes

### Pre-Merge Checklist
- ✅ All tests pass in feature worktree
- ✅ Code follows project standards (cargo fmt, cargo check)
- ✅ Changelog updated appropriately
- ✅ No conflicts with main branch
- ✅ Feature is complete and functional

## Enforcement

### Rule Violations
Working directly on main branch is a **serious workflow violation** that:
- Risks main branch stability
- Bypasses feature isolation
- Prevents proper code review
- Violates project development standards

### Recovery from Main Branch Work
If you accidentally work on main:
1. **Stop immediately** - make no more commits
2. **Create feature branch**: `git checkout -b feat/NAME_OF_FEATURE`
3. **Create worktree**: `git worktree add ../bg_remove-rs-feat-NAME_OF_FEATURE feat/NAME_OF_FEATURE`
4. **Continue work in worktree**
5. **Reset main**: `git checkout main && git reset --hard HEAD~N` (where N is commits to undo)
6. **Complete feature in worktree and merge properly**

## Integration with Other Rules

### Changelog Updates
- Update changelog files **within the feature worktree**
- Include changelog changes **in the same feature branch**
- Follow changelog update rules while working in feature branch

### Commit Guidelines
- Use conventional commit format **within feature worktrees**
- Create logical commit boundaries **during feature development**
- Maintain clean commit history **before merging to main**

### Testing Requirements
- Run `cargo check`, `cargo fmt`, `cargo test` **within feature worktree**
- Complete all validation **before merging to main**
- Ensure feature works independently **in isolated environment**

This enforces proper feature isolation and maintains main branch stability through disciplined worktree usage.