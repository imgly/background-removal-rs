# Repository Structure Reference

## Primary Source of Truth: llms.txt

The **authoritative reference** for the repository structure is the `llms.txt` file located in the project root. This file serves as the **canonical source of truth** for understanding the codebase organization and file purposes.

## When to Consult llms.txt

### ✅ ALWAYS Reference llms.txt For:
- **Understanding project structure** and file organization
- **Locating specific functionality** across the codebase
- **Understanding file relationships** and dependencies
- **Finding relevant files** for specific tasks or features
- **Onboarding** new developers or AI assistants
- **Documentation** of what each file contains
- **Architecture understanding** of the multi-crate workspace

### 📋 llms.txt Content Structure

The `llms.txt` file contains:
- **One-line summaries** of each file's purpose
- **Organized sections** by functionality (core, CLI, backends, etc.)
- **File path mappings** to their functional descriptions
- **Documentation files** and their content scope
- **Build and configuration files** and their roles

## Benefits of Centralized Structure Reference

### 🎯 Single Source of Truth
- **Eliminates confusion** about file purposes
- **Prevents outdated** scattered documentation
- **Ensures consistency** across documentation
- **Simplifies navigation** for new contributors

### 🔄 Maintenance Efficiency
- **One file to update** when structure changes
- **Automatic synchronization** with actual codebase
- **Easy to keep current** with project evolution
- **Clear ownership** of structure documentation

### 🚀 Developer Experience
- **Quick orientation** for new team members
- **Efficient file discovery** during development
- **Context understanding** before diving into code
- **Reduced cognitive load** when navigating large codebase

## Integration with Development Workflow

### Before Making Structural Changes
1. **Check current structure** in llms.txt
2. **Understand file relationships** and dependencies
3. **Plan changes** with full context
4. **Update llms.txt** after structural modifications

### When Adding New Files
1. **Determine appropriate location** using llms.txt as reference
2. **Follow existing patterns** and organization principles
3. **Add new files** to llms.txt with descriptive summaries
4. **Maintain consistency** with existing documentation style

### For Code Review and Quality
- **Verify structural decisions** align with documented organization
- **Check for missing files** in llms.txt documentation
- **Ensure new additions** follow established patterns
- **Validate file purposes** match their actual implementation

## Compliance and Enforcement

### Mandatory Requirements
- **ALWAYS consult** llms.txt when navigating unfamiliar parts of the codebase
- **NEVER assume** file purposes without checking documentation
- **IMMEDIATELY update** llms.txt when adding/removing/moving files
- **VALIDATE** that file descriptions match actual content

### Quality Standards
- **Accurate descriptions** that reflect actual file content
- **Consistent formatting** following established patterns
- **Complete coverage** of all significant files in the repository
- **Regular reviews** to ensure continued accuracy

## Summary

The `llms.txt` file is the **definitive guide** to understanding this repository's structure. It provides comprehensive, up-to-date information about every file's purpose and role in the project. Always reference it first when exploring the codebase or making structural decisions.

**Key Principle:** When in doubt about repository structure, check `llms.txt` first.