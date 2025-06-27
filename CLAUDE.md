Your name is 4K17B.


# General
Each started session shall have an unique session SESSION_UUID that can be used as ENV variable 

Look for API Keys in .env files, with this particular priority
- .env.local
- .env.development
- .env


# Workflow
Whenever you switch directories, remember to switch back to the root before the next step.
Whenever an external process is started in the background, remember it's PID and stop it whenever it is not needed anymore.


# Rules
@.claude/rules/repository_structure_reference.md
@.claude/rules/task_management.md
@.claude/rules/implementation_planning.md
@.claude/rules/process_execution.md
@.claude/rules/kiss_principle.md
@.claude/rules/yagni_principle.md
@.claude/rules/code_quality.md

## Coding
@.claude/rules/solid_principles.md
@.claude/rules/commit_guidelines.md
@.claude/rules/documentation_protocols.md
@.claude/rules/changelog_update.md
@.claude/rules/git_workflow.md
@.claude/rules/cargo_check_after_changes.md
@.claude/rules/cargo_fmt_after_changes.md
@.claude/rules/cargo_test_after_completion.md

## User Interaction
@.claude/rules/conversational_interaction.md

## Documentation
@.claude/rules/maintain_llms_txt.md

## Tool use
@.claude/rules/thinking_and_tool_use.md
@.claude/rules/prompt_for_tool_triggering-websearch.md
@.claude/rules/parallel_tool_calling.md

### When to use
@.claude/rules/tool_use_playwright_mcp.md
