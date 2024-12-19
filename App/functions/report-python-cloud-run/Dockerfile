# Copyright 2020 Google, LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use the official uv python image
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y libgdal-dev libgl1

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Accept EULA and install Microsoft fonts (including Arial)
RUN apt-get update && \
    echo "deb http://deb.debian.org/debian bookworm contrib non-free" > /etc/apt/sources.list.d/contrib.list && \
    apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y ttf-mscorefonts-installer

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Clone the coclicodata and global-coastal-atlas repositories STAC catalogs
RUN mkdir -p /app/data/catalogs
RUN cd /app/data/catalogs && \
    git clone -n --depth=1 --filter=tree:0 https://github.com/openearth/coclicodata.git && \
    cd coclicodata && \
    git sparse-checkout set --no-cone /current && \
    git checkout && \
    cd /app/data/catalogs && \
    git clone -b subsidence_etienne -n --depth=1 --filter=tree:0 https://github.com/openearth/global-coastal-atlas.git && \
    cd global-coastal-atlas && \
    git sparse-checkout set --no-cone /STAC/data/current && \
    git checkout && \
    cd /app

ENV STAC_ROOT_DEFAULT="./data/catalogs/global-coastal-atlas/STAC/data/current/catalog.json"
ENV STAC_COCLICO="./data/catalogs/coclicodata/current/catalog.json"

CMD uv run --with gunicorn gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 main:app
