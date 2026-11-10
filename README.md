# SageServe MLOps Platform

**SageServe** is an end-to-end MLOps platform for forecast-aware, disaggregated, multi-region autoscaling and GPU resource allocation in Large Language Model (LLM) serving.

Instead of reactive autoscaling, SageServe anticipates incoming token demand across geographical regions and models using time-series forecasting (ARIMA, HMM) and proactively solves global Mixed-Integer Linear Programs (MILP) to minimize SLA violations (TTFT/TBT) and operational GPU costs. The platform provides full MLOps lifecycle governance—from automated data quality validation (Great Expectations) and cryptographic data lineage tracking to model registry management, drift detection, and interactive simulation.

---

## Quick Start & Setup

Your commands are completely correct! The entire environment is containerized with Docker.

### Prerequisites

You will need to install  [Docker](https://docs.docker.com/get-docker/)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/SreenathKarthick11/Sage-Serve.git
cd Sage-Serve

# 2. Build the Docker image
docker compose build

# 3. Launch the Web UI in the background
docker compose up -d webui
```

### Walk through of Setup

<video src="docs/Video/Build_and_setup.mp4" autoplay loop muted playsinline width="100%"></video>

Once running, open your browser and navigate to: **[http://localhost:8501](http://localhost:8501)**

---

### Additional Commands

**Interactive Dev Shell**:
  ```bash
  docker compose run --rm sageserve bash
  ```
**View Web UI Logs**:
  ```bash
  docker compose logs -f webui
  ```
**Stop Services**:
  ```bash
  docker compose down
  ```

---

## Documentation & Details

For in-depth details on system architecture, algorithms, and experiments:

- **Presentation Slides**: [SageServe MLOps Platform (PDF)](docs/Presentation/SageServe%20MLOps%20Platform.pdf)
- **Technical Architecture Reference**: [SageServe MLOps Platform Detail](docs/Summary_Docs/SageServe_MLOps_Platform_detail.md)
- **GPU Allocation Deep Dive**: [How SageServe does GPU Allocation](docs/Summary_Docs/How%20does%20Sage%20Serve%20does%20%20GPU%20allocation.md)
- **Original Research Paper**: [SageServe Highlighted Paper](docs/Papers/sage_serve_highlighted.pdf)

---

