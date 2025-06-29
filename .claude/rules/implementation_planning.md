# Implementation Planning and Worktree Creation

When starting any new plan or feature implementation, follow these steps:

## 1. MANDATORY: Create Feature Worktree
Before ANY implementation work begins, you MUST create a git worktree:

```bash
git worktree add worktree/feat-FEATURE_NAME -b feat/FEATURE_NAME
cd worktree/feat-FEATURE_NAME
```

**NEVER work directly on main branch.** All development must happen in isolated worktrees.

## 2. MANDATORY: Create Implementation Plan IMMEDIATELY After Worktree
**CRITICAL**: Immediately after creating the worktree, you MUST create the implementation plan before writing any code.

Write a detailed plan in `docs/feat/TIMESTAMP_FEATURE_NAME/implementation_plan.md` (within the feature worktree) that includes:
- Feature description and goals
- Step-by-step implementation tasks
- Potential risks or impacts on existing functionality
- Questions that need clarification
- Explicit list of any functionality that will be modified or removed
- Planned worktree workflow and merge strategy

**ENFORCEMENT**: Creating implementation plan is NOT optional - it must be done immediately after worktree creation and before any code changes.

## 3. Clarification Requirements
Ask the user about:
- Any ambiguous requirements
- Potential side effects on existing features
- Performance implications
- API changes or breaking changes
- Integration points with existing code

## 4. Track Progress and Update Implementation Plan
**MANDATORY**: Always update the implementation plan during development **within the feature worktree**:
- Mark tasks as ✅ completed, 🔄 in progress, or ❌ blocked as you complete them
- Update with any discoveries or changes during implementation
- Document any deviations from the original plan
- Add new phases or tasks that emerge during development
- Update success criteria with actual results and validation outcomes
- Include final results section when project is complete
- Update changelog files within the feature worktree

**CRITICAL**: The implementation plan must be a living document that accurately reflects:
- Current progress status
- Actual implementation approach taken
- Issues encountered and how they were resolved
- Final validation results and success metrics
- Lessons learned and project outcomes

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
cd ../..

# Merge feature branch
git merge feat/FEATURE_NAME

# Clean up
git worktree remove worktree/feat-FEATURE_NAME
```

## 8. Enforcement
**Serious Violation**: Working directly on main branch bypasses:
- Feature isolation and safety
- Proper code review processes
- Project development standards
- Risk management protocols

This ensures thoughtful, well-documented development that preserves system integrity through proper worktree isolation.