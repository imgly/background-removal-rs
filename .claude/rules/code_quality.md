# Code Quality Standards

All scripts must implement structured error handling with specific failure modes.

Every function must include a concise, purpose-driven docstring.

Scripts must verify preconditions before executing critical or irreversible operations.

Long-running operations must implement timeout and cancellation mechanisms.

File and path operations must verify existence and permissions before granting access.