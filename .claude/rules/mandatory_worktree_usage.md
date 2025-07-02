# Mandatory Git Worktree Usage

## CRITICAL RULE: Never Work on Main Branch

You MUST create a git worktree for ALL development activities. Working directly on the main branch is **strictly prohibited** and considered a serious workflow violation.

## When Worktrees Are Required

### ✅ ALWAYS Required
- **New features or capabilities**
- **Bug fixes** (except emergency single-commit hotfixes)
- **Documentation changes** 
- **Refactoring tasks**
- **Configuration updates**
- **Rule additions or modifications**
- **Multi-commit development work**
- **Experimental or exploratory work**
- **Changelog updates**
- **Dependency updates**

### ❌ RARE Exceptions (Use with Extreme Caution)
- **Single typo fixes** with immediate commit
- **Emergency hotfixes** requiring immediate deployment
- **Critical security patches** that cannot wait for worktree setup

## Mandatory Workflow Steps

### 1. Start Every Task with Worktree Creation
```bash
# ALWAYS the first command - NO EXCEPTIONS
git worktree add worktree/feat-DESCRIPTIVE_NAME -b feat/DESCRIPTIVE_NAME
cd worktree/feat-DESCRIPTIVE_NAME
```

## CRITICAL: Directory Persistence in Worktrees

### After creating a worktree, you MUST:
1. **Immediately `cd` into the worktree directory** and NEVER leave it during development
2. **NEVER execute commands from outside the worktree** during feature development  
3. **Verify current directory** with `pwd` before each significant command sequence
4. **Use relative paths** within the worktree, not absolute paths to main
5. **Stay in worktree** until feature is complete and ready for merge

### Enforcement Commands
```bash
# MANDATORY after worktree creation
cd worktree/feat-DESCRIPTIVE_NAME
pwd && git branch --show-current  # Verify location and branch before proceeding
```

**Expected Output:**
- Directory: `/path/to/repo/worktree/feat-DESCRIPTIVE_NAME`  
- Branch: `feat/DESCRIPTIVE_NAME`

### Violation Prevention
- **Before ANY git/cargo/file command**: Run `pwd` to verify you're in the worktree
- **If not in worktree directory**: STOP immediately and `cd` back to correct location
- **Use `git status`** to verify you're on the feature branch
- **NEVER use absolute paths** to main repository files - use relative paths within worktree

### Command Pattern for Worktree Operations
```bash
# 1. Verify location (MANDATORY before command sequences)
pwd && git branch --show-current

# 2. If not in worktree, navigate there immediately
cd worktree/feat-DESCRIPTIVE_NAME

# 3. Verify again after navigation
pwd && git branch --show-current

# 4. Then execute your intended commands
git status
cargo check
# ... other development commands
```

### Directory Violations
**Serious Violation**: Executing development commands from main directory when a worktree exists because it:
- **Bypasses feature isolation** - changes affect main instead of feature branch
- **Creates confusion** about which branch changes are applied to
- **Violates worktree safety** - defeats the purpose of isolated development
- **Risks main branch contamination** - accidentally commits to wrong branch
- **Breaks development workflow** - mixes feature and main development

### 2. Complete ALL Work in Feature Worktree
- Make all commits in the feature worktree
- Update changelog files within feature branch
- Run all tests and validation in feature environment
- Develop, test, and validate in complete isolation

### 3. Merge Back to Main When Complete
```bash
# Switch back to main worktree (project root)
cd ../..

# Validate main branch is clean
git status

# Merge feature branch
git merge feat/DESCRIPTIVE_NAME

# Clean up worktree
git worktree remove worktree/feat-DESCRIPTIVE_NAME
```

## Branch Naming Standards

Use descriptive names with appropriate prefixes:

### Feature Branches
- `feat/changeset-implementation`
- `feat/model-optimization`
- `feat/cli-batch-processing`

### Bug Fix Branches
- `fix/memory-leak-processor`
- `fix/webp-transparency-corruption`
- `fix/aspect-ratio-distortion`

### Documentation Branches
- `docs/api-documentation`
- `docs/deployment-guide`
- `docs/performance-tuning`

### Refactoring Branches
- `refactor/processor-architecture`
- `refactor/error-handling`
- `refactor/module-organization`

### Maintenance Branches
- `chore/dependency-updates`
- `chore/ci-improvements`
- `chore/cleanup-unused-code`

## Directory Naming Convention

Always use the worktree subdirectory with descriptive names:

```
worktree/feat-FEATURE_NAME
worktree/fix-BUG_NAME
worktree/docs-DOC_NAME
worktree/refactor-REFACTOR_NAME
worktree/chore-TASK_NAME
```

## Benefits of Mandatory Worktree Usage

### Safety and Isolation
- **Main branch protection**: Main remains stable and deployable
- **Feature isolation**: Each feature develops independently
- **Conflict prevention**: Avoid merge conflicts and integration issues
- **Rollback safety**: Easy to abandon or restart development

### Quality and Review
- **Complete feature review**: Review entire feature as a unit
- **Testing isolation**: Test features in clean environment
- **Integration validation**: Verify feature works with current main
- **Code quality**: Maintain high standards through isolation

### Project Management
- **Clear feature boundaries**: Each worktree represents a discrete task
- **Progress tracking**: Easy to see what's in development
- **Parallel development**: Multiple features can develop simultaneously
- **Clean history**: Logical feature boundaries in git history

## Enforcement and Violations

### Workflow Violations
Working directly on main branch constitutes a **serious violation** because it:
- **Risks main branch stability**
- **Bypasses feature isolation safeguards**
- **Prevents proper code review**
- **Violates established development standards**
- **Creates integration risks**
- **Reduces code quality**

### Recovery from Violations
If you accidentally start work on main:

```bash
# 1. STOP immediately - make no more commits to main
# 2. Create feature branch from current state
git checkout -b feat/RECOVERY_FEATURE_NAME

# 3. Create worktree for the feature branch
git worktree add worktree/feat-RECOVERY_FEATURE_NAME feat/RECOVERY_FEATURE_NAME
cd worktree/feat-RECOVERY_FEATURE_NAME

# 4. Reset main to clean state
cd ../..
git reset --hard HEAD~N  # N = number of commits to undo

# 5. Continue development in feature worktree
cd worktree/feat-RECOVERY_FEATURE_NAME
# Continue your work here
```

## Integration with Project Rules

### Changelog Updates
- Update changelog files **within feature worktrees**
- Include changelog modifications **in feature branch commits**
- Follow changelog rules **while working in isolation**

### Testing Requirements
- Run `cargo check`, `cargo fmt`, `cargo test` **in feature worktree**
- Validate feature **before merging to main**
- Ensure all quality checks **pass in isolation**

### Commit Guidelines
- Use conventional commit format **within feature branches**
- Create logical commit boundaries **during feature development**
- Maintain clean history **throughout feature development**

## Worktree Management Commands

### Useful Commands
```bash
# List all worktrees
git worktree list

# Remove a worktree (when done)
git worktree remove worktree/feat-FEATURE_NAME

# Remove worktree and delete branch
git worktree remove worktree/feat-FEATURE_NAME
git branch -d feat/FEATURE_NAME

# Prune deleted worktrees
git worktree prune
```

### Worktree Status Check
```bash
# Check worktree status before merging
cd worktree/feat-FEATURE_NAME
git status
cargo check
cargo test

# Verify no uncommitted changes
git diff --exit-code
```

## Summary

**NEVER work directly on main.** Always create worktrees. This is not optional—it's a fundamental requirement for maintaining code quality, feature isolation, and project stability in the bg_remove-rs repository.