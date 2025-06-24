# Browser MCP Usage Rules

Use Browser MCP tools when:
- Need lightweight browser interactions for basic web tasks
- Quick page navigation and simple content retrieval
- Taking screenshots for visual inspection
- Basic form interactions and clicking elements
- Getting console logs for debugging web pages
- Simple accessibility testing via snapshots
- Quick browser automation without complex workflow requirements

Do NOT use Browser MCP for:
- Complex multi-step browser workflows (use Playwright MCP instead)
- Advanced testing scenarios requiring detailed control
- File uploads or complex drag-and-drop operations
- Multi-tab management or advanced browser features
- PDF generation or advanced screenshot features
- Static content that doesn't require browser rendering (use WebFetch)

Always use `mcp__browser__browser_snapshot` to understand page structure before performing actions.

Browser MCP is ideal for simple, quick browser tasks where Playwright would be overkill.