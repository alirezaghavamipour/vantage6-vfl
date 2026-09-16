FROM python:3.12-slim
LABEL org.opencontainers.image.description="vantage6-vfl: privacy-preserving vertical FL algorithms (PSI, training) via Rep3 MPC"
WORKDIR /app
RUN pip install --no-cache-dir vantage6-algorithm-tools
COPY vfl_pkg/ ./vfl_pkg/
COPY setup.py .
RUN pip install --no-cache-dir -e .
ENV PYTHONUNBUFFERED=1
ARG PKG_NAME="vfl_pkg"
ENV PKG_NAME=${PKG_NAME}
CMD python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
