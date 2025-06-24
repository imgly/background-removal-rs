# Process Execution Requirements

Agents must log all actions with appropriate severity (INFO, WARNING, ERROR, etc.).

Any failed task must include a clear, human-readable error report.

Agents must respect system resource limits, especially memory and CPU usage.

Long-running tasks must expose progress indicators or checkpoints.

Retry logic must include exponential backoff and failure limits.