# Deployment Strategy

This AI Customer Support Platform is deployed via Docker to **Render**.

## Architectural Decisions for Deployment

While many applications opt for serverless deployments (such as AWS App Runner, AWS Lambda, or Vercel), this platform utilizes a Docker-native persistent service on Render. This was a deliberate engineering decision based on the specific constraints of our machine learning pipeline and available infrastructure tiers:

1. **Long-Running Process Limits**: Serverless functions often have strict execution timeouts (e.g., 10 seconds for Vercel). Loading Heavy transformers (`distilbert`) and OCR engines (`easyocr`) requires more time, particularly on cold starts.
2. **Package Size Constraints**: AI dependencies (`torch`, `transformers`, `easyocr`, `opencv-python-headless`) result in an environment size well over 2GB. Serverless platforms often enforce bundle limits (typically 250MB to 500MB) which completely precludes packaging real PyTorch models.
3. **Lazy Loading into Memory**: By running a dedicated Docker container, we can lazily load models into persistent RAM. The first request that includes an image loads the YOLOv8 and easyOCR models, and subsequent requests benefit from zero-load inference.
4. **Persistent SQLite Database**: The system utilizes SQLite to store ticket history, session state, and analytics. Render web services support disk mounting for persistent databases without the overhead of spinning up an external PostgreSQL/RDS instance for this portfolio demonstration.

## CI/CD Pipeline

The project uses **GitHub Actions** for Continuous Integration and Continuous Deployment.

- **CI**: On every push or pull request, the `ci.yml` workflow installs dependencies, runs `flake8` for strict Python linting, trains the `distilbert` LoRA classifier, and runs all 184 `pytest` tests with coverage reporting.
- **CD**: Upon a successful merge to `main`, Render's Auto-Deploy webhook automatically pulls the latest commit, builds the Docker image, and rolls it out with zero downtime.
