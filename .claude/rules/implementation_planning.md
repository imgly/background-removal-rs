# Implementation Planning and Worktree Creation

When starting any new plan or feature implementation, follow these steps:

## 1. MANDATORY: Create Feature Worktree
Before ANY implementation work begins, you MUST create a git worktree:

```bash
git worktree add ../bg_remove-rs-feat-FEATURE_NAME -b feat/FEATURE_NAME
cd ../bg_remove-rs-feat-FEATURE_NAME
```

**NEVER work directly on main branch.** All development must happen in isolated worktrees.

## 2. Create Implementation Plan
Write a detailed plan in `docs/implementation_plan.md` (within the feature worktree) that includes:
- Feature description and goals
- Step-by-step implementation tasks
- Potential risks or impacts on existing functionality
- Questions that need clarification
- Explicit list of any functionality that will be modified or removed
- Planned worktree workflow and merge strategy

## 3. Clarification Requirements
Ask the user about:
- Any ambiguous requirements
- Potential side effects on existing features
- Performance implications
- API changes or breaking changes
- Integration points with existing code

## 4. Track Progress
Maintain the implementation plan throughout development **within the feature worktree**:
- Mark tasks as ✅ completed, 🔄 in progress, or ❌ blocked
- Update with any discoveries or changes during implementation
- Document any deviations from the original plan
- Update changelog files within the feature worktree

## 5. Preserve Existing Functionality
NEVER remove or change existing functionality unless:
- It is explicitly stated in the implementation plan
- The user has confirmed the change is intended
- The change is documented with rationale

## 6. Questions to Always Ask
Before implementation:
- "Is there any existing functionality that should be preserved?"
- "Are there any edge cases I should consider?"
- "What should happen if [specific scenario]?"
- "Should this integrate with or replace existing features?"

## 7. Worktree Completion Workflow
When feature is complete:

```bash
# Ensure all tests pass in feature worktree
cargo test
cargo check
cargo fmt

# Switch back to main
cd ../bg_remove-rs

# Merge feature branch
git merge feat/FEATURE_NAME

# Clean up
git worktree remove ../bg_remove-rs-feat-FEATURE_NAME
```

## 8. Enforcement
**Serious Violation**: Working directly on main branch bypasses:
- Feature isolation and safety
- Proper code review processes
- Project development standards
- Risk management protocols

This ensures thoughtful, well-documented development that preserves system integrity through proper worktree isolation.