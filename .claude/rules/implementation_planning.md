# Implementation Planning and Branching

When starting any new plan or feature implementation, follow these steps:

## 1. Create a Feature Branch
Before any implementation work begins:
```bash
git checkout -b feat/FEATURE_NAME
```
Or use git worktree for larger features:
```bash
git worktree add ../project-feat-FEATURE_NAME -b feat/FEATURE_NAME
```

## 2. Create Implementation Plan
Write a detailed plan in `docs/implementation_plan.md` that includes:
- Feature description and goals
- Step-by-step implementation tasks
- Potential risks or impacts on existing functionality
- Questions that need clarification
- Explicit list of any functionality that will be modified or removed

## 3. Clarification Requirements
Ask the user about:
- Any ambiguous requirements
- Potential side effects on existing features
- Performance implications
- API changes or breaking changes
- Integration points with existing code

## 4. Track Progress
Maintain the implementation plan throughout development:
- Mark tasks as ✅ completed, 🔄 in progress, or ❌ blocked
- Update with any discoveries or changes during implementation
- Document any deviations from the original plan

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

This ensures thoughtful, well-documented development that preserves system integrity.