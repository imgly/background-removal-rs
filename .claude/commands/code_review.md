---
name: code_review
description: Perform a comprehensive code review analyzing quality, complexity, and architecture
schema:
  properties:
    focus_area:
      description: "Specific area to focus on (optional): complexity, architecture, quality, or all"
      type: string
      enum: ["complexity", "architecture", "quality", "all"]
      default: "all"
    depth:
      description: "Depth of analysis: quick, standard, or deep"
      type: string
      enum: ["quick", "standard", "deep"]
      default: "standard"
  required: []
---

# Comprehensive Code Review

You are the best Architect and Code Reviewer I know. Please conduct a comprehensive code review of the codebase and provide detailed findings on what can be done better.

## Review Criteria

### 1. Code Quality and Readability
- **Code is good when**: It is easily understandable, well-structured, and follows established patterns
- **Complex business logic**: Should be abstracted and separated from infrastructure concerns
- **Function length**: Identify functions exceeding 50 lines that should be broken down
- **Naming conventions**: Assess clarity and consistency of naming
- **Documentation**: Evaluate inline comments and API documentation quality

### 2. Configuration Complexity Analysis
- **For each configuration flag**, analyze:
  - What complexity it introduces to the system
  - What functionality would be lost if removed
  - Whether it follows YAGNI (You Aren't Gonna Need It) principle
  - How it could be simplified or consolidated
- **Look for**:
  - Redundant configuration options
  - Conflicting flags or settings
  - Over-engineered flexibility
  - Configuration proliferation across layers

### 3. Architecture and Separation of Concerns
- **Business Logic Separation**:
  - Identify where business logic is mixed with infrastructure (I/O, UI, networking)
  - Find violations of Single Responsibility Principle
  - Assess impact on testability and maintainability
- **Layering Issues**:
  - Look for improper dependencies between layers
  - Identify circular dependencies
  - Find abstraction leaks

### 4. SOLID Principles Adherence
- **Single Responsibility**: Each class/function should have one reason to change
- **Open/Closed**: Code should be open for extension, closed for modification
- **Liskov Substitution**: Derived classes must be substitutable for base classes
- **Interface Segregation**: Many specific interfaces over general-purpose ones
- **Dependency Inversion**: Depend on abstractions, not concretions

## Analysis Approach

### Phase 1: Codebase Exploration
1. Identify main components and their responsibilities
2. Map out the system architecture
3. Understand key workflows and data flows

### Phase 2: Detailed Analysis
1. **Configuration Complexity**:
   - Count all configuration options
   - Identify interdependencies
   - Assess each against YAGNI principle
   - Calculate complexity reduction potential

2. **Code Quality Issues**:
   - Large functions and classes
   - Duplicated code patterns
   - Complex conditionals
   - Mixed abstraction levels

3. **Architecture Problems**:
   - Business logic in wrong layers
   - Tight coupling between components
   - Missing abstractions
   - Testability concerns

### Phase 3: Impact Assessment
For each finding, provide:
- **Current state**: What the code does now
- **Problem**: Why this is an issue
- **Impact**: How it affects the system
- **Recommendation**: Specific improvement suggestion
- **Priority**: High/Medium/Low based on impact and effort

## Output Format

### Executive Summary
- Overall code quality assessment
- Key strengths of the codebase
- Critical issues requiring attention
- Estimated complexity reduction potential

### Detailed Findings

#### 1. Configuration Complexity
- Total configuration options count
- Redundant configurations list
- YAGNI violations with impact assessment
- Simplification recommendations with before/after examples

#### 2. Code Quality Issues
- Specific problems with file:line references
- Code examples showing the issue
- Refactoring suggestions with code samples

#### 3. Architecture Concerns
- Business logic separation violations
- Testability problems
- Maintainability issues
- Specific refactoring recommendations

### Prioritized Action Plan

#### Immediate Actions (Next 2 Weeks)
- Quick wins with high impact
- Critical bug risks
- Simple refactorings

#### Short Term (Next Month)
- Moderate complexity improvements
- Architecture adjustments
- Testing improvements

#### Long Term (Next Quarter)
- Major architectural changes
- System-wide refactorings
- Technical debt reduction

### Metrics and Impact

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Configuration Options | X | Y | Z% reduction |
| Average Function Length | X lines | Y lines | Z% reduction |
| Test Coverage | X% | Y% | Z% increase |
| Coupling Score | X | Y | Z% improvement |

## Special Focus Areas

When analyzing configuration:
- Each flag should justify its existence
- Complex configurations often hide design problems
- Prefer convention over configuration
- Auto-detection over manual settings

When analyzing business logic:
- Pure functions are easier to test
- I/O should be at the edges
- Business rules should be centralized
- Infrastructure should be pluggable

Remember: The goal is not just to identify problems, but to provide actionable, prioritized recommendations that improve code quality while maintaining functionality.