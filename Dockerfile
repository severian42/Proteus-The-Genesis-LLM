FROM python:3.10-slim
ARG TARGETPLATFORM
ARG BUILDPLATFORM
RUN echo "I am running on $BUILDPLATFORM, building for $TARGETPLATFORM"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-dev \
    xvfb \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual display for headless rendering
ENV DISPLAY=:99
RUN mkdir -p /tmp/.X11-unix

# Create non-root user
RUN useradd -m -s /bin/bash genesis

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup script executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set permissions
RUN chown -R genesis:genesis /app

# Switch to non-root user
USER genesis

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GENESIS_BACKEND=cpu
ENV GENESIS_LOGGING_LEVEL=debug
ENV PORT=4242
EXPOSE 4242

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:4242/health || exit 1

# Run the application
CMD ["/app/start.sh"] 