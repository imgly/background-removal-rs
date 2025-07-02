# Maintain LLMs.txt File

## CRITICAL RULE: Keep llms.txt Updated

The `llms.txt` file MUST be maintained and kept synchronized with the actual repository structure. This file serves as the **canonical reference** for understanding the codebase and MUST reflect the current state at all times.

## When llms.txt Updates Are Required

### ✅ MANDATORY Updates For:
- **Adding new files** to the repository
- **Removing files** from the repository  
- **Moving/renaming files** to different locations
- **Changing file purposes** or functionality significantly
- **Adding new directories** or restructuring
- **Modifying build scripts** or configuration files
- **Adding new crates** to the workspace
- **Updating documentation** file purposes

### 🔄 Update Workflow

#### 1. Before Making Structural Changes
```bash
# Check current llms.txt state
cat llms.txt | grep -i "filename_pattern"

# Understand existing organization patterns
# Plan changes within established structure
```

#### 2. During Development
- **Make structural changes** to files/directories
- **Note what needs updating** in llms.txt
- **Test that changes work** as expected

#### 3. Immediately After Changes
- **Update llms.txt** with new/changed/removed entries
- **Follow existing format** and style conventions  
- **Verify all paths are correct** and descriptions accurate
- **Maintain alphabetical order** within sections

#### 4. Before Committing
```bash
# Verify llms.txt is updated
git diff llms.txt

# Check for any missing files
find . -name "*.rs" -o -name "*.md" -o -name "*.toml" | sort | grep -v target/ | grep -v .git/
```

## llms.txt Format Requirements

### File Entry Format
```
/path/to/file : Concise one-line description of purpose and functionality
```

### Organizational Structure
1. **Core Configuration** - Cargo.toml, build configs, IDE settings
2. **Documentation** - README, CHANGELOG, guides
3. **Core Library** - Main library crate files
4. **CLI Application** - Command-line interface files  
5. **Backend Implementations** - ONNX, Tract backend files
6. **Testing Infrastructure** - E2E, integration, unit tests
7. **Build/CI/CD** - Scripts, workflows, automation
8. **Model Configuration** - AI model configs and parameters
9. **Claude AI Configuration** - Rules, commands, expert configs

### Description Guidelines

#### ✅ Good Descriptions
- **Concise and specific**: "Neural network inference backend trait and implementations"
- **Functional focus**: "ICC color profile handling for preserving image color accuracy"  
- **Clear purpose**: "Main CLI entry point with argument parsing and processing logic"
- **Context-aware**: "Cross-compilation script for multiple target architectures"

#### ❌ Poor Descriptions  
- **Too vague**: "Some utilities" or "Helper functions"
- **Implementation details**: "Uses HashMap with custom hasher for performance"
- **Obvious statements**: "Rust source file" or "Contains code"
- **Outdated information**: References to removed features or old architecture

## Quality Standards

### Accuracy Requirements
- **Descriptions MUST match** actual file content and purpose
- **Paths MUST be correct** and up-to-date
- **No missing files** that are significant to understanding the codebase
- **No orphaned entries** for files that no longer exist

### Consistency Standards
- **Follow established patterns** for similar file types
- **Use consistent terminology** across descriptions
- **Maintain section organization** as files are added/moved
- **Keep formatting uniform** with existing entries

### Completeness Criteria
- **All source files** (.rs) documented
- **All configuration files** (.toml, .json, .yml) included
- **All documentation files** (.md) listed
- **All build/script files** (.sh, build.rs) covered
- **All significant directories** represented

## Maintenance Automation

### Regular Verification
```bash
# Check for files not in llms.txt
comm -23 <(find . -type f \( -name "*.rs" -o -name "*.md" -o -name "*.toml" \) | grep -v target/ | sort) <(grep "^/" llms.txt | cut -d: -f1 | sort)

# Check for dead entries in llms.txt  
while read -r path; do [[ -f "${path#/}" ]] || echo "Missing: $path"; done < <(grep "^/" llms.txt | cut -d: -f1)
```

### Integration with Development Tools
- **Pre-commit hooks** should verify llms.txt completeness
- **CI/CD pipelines** should validate llms.txt accuracy
- **Code review process** should check llms.txt updates
- **Documentation builds** should reference llms.txt

## Error Prevention

### Common Mistakes to Avoid
- **Forgetting to update** llms.txt after file changes
- **Adding entries** for temporary or build-generated files
- **Using incorrect paths** (relative vs absolute)
- **Copying descriptions** without customizing for actual content
- **Leaving stale entries** after file removal

### Validation Checklist
Before committing changes that affect file structure:
- [ ] llms.txt includes all new files with accurate descriptions
- [ ] Removed/moved files have entries updated or deleted
- [ ] All paths are correct and accessible
- [ ] Descriptions accurately reflect current file purposes  
- [ ] Formatting follows established conventions
- [ ] Section organization is maintained appropriately

## Integration with Other Rules

### Works With Repository Structure Reference
- llms.txt serves as the **source of truth** referenced by other rules
- Changes to structure **MUST update both** files and llms.txt
- Documentation references **SHOULD point to** llms.txt for structure

### Supports Development Workflow
- **Mandatory worktree usage** includes llms.txt updates in feature branches
- **Implementation planning** should reference llms.txt for context
- **Code quality standards** include keeping documentation current

### Enables AI Assistant Effectiveness
- **Updated llms.txt** provides current context for Claude and other AI tools
- **Accurate descriptions** help AI understand codebase organization
- **Complete coverage** ensures AI has full repository awareness

## Summary

The `llms.txt` file is **not optional documentation**—it's a **critical infrastructure component** that enables effective navigation, understanding, and maintenance of the repository.

**Key Responsibilities:**
- **Update immediately** when making structural changes
- **Maintain accuracy** of all file descriptions  
- **Follow format conventions** consistently
- **Verify completeness** regularly
- **Integrate updates** into development workflow

**Failure to maintain llms.txt undermines:**
- Repository navigation efficiency
- New developer onboarding
- AI assistant effectiveness  
- Code review quality
- Documentation reliability