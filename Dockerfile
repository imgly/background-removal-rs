FROM ubuntu:24.04

# Install dependencies including Node.js and git-lfs plus development tools
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    ca-certificates \
    tmux \
    nodejs \
    npm \
    git-lfs \
    git \
    vim \
    nano \
    wget \
    unzip \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    jq \
    tree \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally using npm (as root)
RUN npm install -g @anthropic-ai/claude-code

# Create claude user (regular user without privileges)
RUN useradd -m -s /bin/bash claude

# Switch to claude user
USER claude

# Install rustup and cargo for claude user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/claude/.cargo/bin:${PATH}"

# Install latest stable Rust toolchain
RUN /home/claude/.cargo/bin/rustup update stable
RUN /home/claude/.cargo/bin/rustup default stable

# Create app and .claude directories in claude's home
RUN mkdir -p /home/claude/app /home/claude/.claude

# Set app directory as working directory
WORKDIR /home/claude/app

# Create entrypoint script in claude's home directory
RUN echo '#!/bin/bash' > /home/claude/entrypoint.sh && \
    echo '' >> /home/claude/entrypoint.sh && \
    echo '# Ensure we are in the app directory' >> /home/claude/entrypoint.sh && \
    echo 'cd /home/claude/app' >> /home/claude/entrypoint.sh && \
    echo '' >> /home/claude/entrypoint.sh && \
    echo '# Check if Claude Code is installed' >> /home/claude/entrypoint.sh && \
    echo 'if command -v claude &> /dev/null; then' >> /home/claude/entrypoint.sh && \
    echo '    echo "Starting Claude Code with dangerously-skip-permissions..."' >> /home/claude/entrypoint.sh && \
    echo '    echo "Working directory: $(pwd)"' >> /home/claude/entrypoint.sh && \
    echo '    echo "Claude config directory: /home/claude/.claude"' >> /home/claude/entrypoint.sh && \
    echo '    exec claude --dangerously-skip-permissions' >> /home/claude/entrypoint.sh && \
    echo 'else' >> /home/claude/entrypoint.sh && \
    echo '    echo "Error: Claude Code not found in PATH"' >> /home/claude/entrypoint.sh && \
    echo '    echo "Falling back to bash shell..."' >> /home/claude/entrypoint.sh && \
    echo '    exec /bin/bash --login' >> /home/claude/entrypoint.sh && \
    echo 'fi' >> /home/claude/entrypoint.sh && \
    chmod +x /home/claude/entrypoint.sh

ENTRYPOINT ["/home/claude/entrypoint.sh"]