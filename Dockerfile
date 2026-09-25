# syntax=docker/dockerfile:1.7

# Start from R image
FROM ghcr.io/rocker-org/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

# Copy uv from image
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# Prevent dialog boxes
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary Linux libraries
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        curl \
        git \
        libcurl4-openssl-dev \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure uv and renv, limit numerical libraries to one thread
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12.12 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/venv \
    RENV_PATHS_LIBRARY=/opt/renv/library \
    PATH=/opt/venv/bin:$PATH \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Make folder for lockfiles
WORKDIR /tmp/dependencies

# Install R packages
COPY renv.lock ./
COPY renv/activate.R renv/activate.R
RUN --mount=type=cache,target=/opt/renv/cache \
    mkdir -p /opt/renv/library \
    && RENV_PATHS_CACHE=/opt/renv/cache \
        RENV_CONFIG_CACHE_SYMLINKS=FALSE \
        RENV_CONFIG_INSTALL_JOBS=1 \
        R -s -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Install Python packages
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    uv sync --locked --no-install-project --compile-bytecode

# Register a kernel that does not depend on Python environment discovery
RUN /opt/venv/bin/python -m ipykernel install \
    --prefix=/usr/local \
    --name=esa-analysis \
    --display-name="Python (ESA analysis)"

# Create the user and the workspace directory
ARG ANALYSIS_UID=1000
ARG ANALYSIS_GID=1000

RUN if ! getent group "${ANALYSIS_GID}" >/dev/null; then \
        groupadd --gid "${ANALYSIS_GID}" analysis; \
    fi \
    && useradd \
        --create-home \
        --no-log-init \
        --gid "${ANALYSIS_GID}" \
        --shell /bin/bash \
        --uid "${ANALYSIS_UID}" \
        analysis \
    && install --directory \
        --owner="${ANALYSIS_UID}" \
        --group="${ANALYSIS_GID}" \
        /workspace

USER analysis
ENV HOME=/home/analysis

WORKDIR /workspace

# Set bash as the default command
CMD ["bash"]
