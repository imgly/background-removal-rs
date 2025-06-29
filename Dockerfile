FROM ubuntu:24.04

# Install dependencies including Node.js
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    ca-certificates \
    tmux \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Claude Code globally using npm
RUN npm install -g @anthropic-ai/claude-code

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Check if Claude Code is installed' >> /entrypoint.sh && \
    echo 'if command -v claude &> /dev/null; then' >> /entrypoint.sh && \
    echo '    echo "Claude Code is installed and ready to use."' >> /entrypoint.sh && \
    echo '    echo "You can run: claude --help"' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '    echo "Error: Claude Code not found in PATH"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    echo '' >> /entrypoint.sh && \
    echo '# Start bash shell' >> /entrypoint.sh && \
    echo 'exec /bin/bash --login' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]