# Playwright MCP Usage Rules

Use Playwright MCP tools when:
- User requests browser automation or web testing tasks
- Need to interact with web pages programmatically (clicking, typing, form submission)
- Taking screenshots or generating PDFs of web pages
- Testing web applications or validating UI behavior
- Scraping dynamic content that requires JavaScript execution
- Monitoring network requests or console logs during page interactions
- Generating automated test scripts for web applications

Do NOT use Playwright MCP for:
- Simple content fetching (use WebFetch instead)
- Static HTML parsing (use WebFetch with appropriate prompts)
- Non-browser tasks or general automation
- Tasks that can be accomplished with simpler tools

Always use `mcp__playwright__browser_snapshot` to understand page structure before performing actions.

Prefer batch operations when possible to minimize browser overhead.