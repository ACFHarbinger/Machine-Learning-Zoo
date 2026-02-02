"""Visual Training Dashboard for Machine Learning Zoo."""

from pathlib import Path
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="ML Zoo Training Dashboard",
    description="Real-time monitoring and management of ML training runs.",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

# Base directory for MLflow runs
MLRUNS_DIR = Path("mlruns")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Render a simple dashboard UI.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Zoo Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f4f4f9; }
            .container { max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .run-list { margin-bottom: 20px; }
            .run-item { padding: 10px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; }
            .chart-container { width: 100%; height: 400px; }
            .status { font-weight: bold; }
            .status.active { color: green; }
            .status.finished { color: blue; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Machine Learning Zoo - Dashboard</h1>
            <div class="run-list" id="run-list">
                <h3>Loading runs...</h3>
            </div>
            <div class="chart-container">
                <canvas id="metricsChart"></canvas>
            </div>
        </div>

        <script>
            async function fetchRuns() {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                const runList = document.getElementById('run-list');
                runList.innerHTML = '<h3>Recent Training Runs</h3>';
                
                runs.forEach(run => {
                    const div = document.createElement('div');
                    div.className = 'run-item';
                    div.innerHTML = `
                        <span>Run: ${run.name} (${run.id})</span>
                        <span class="status ${run.status.toLowerCase()}">${run.status}</span>
                    `;
                    runList.appendChild(div);
                });
            }

            // Simple mock metrics for demonstration
            const ctx = document.getElementById('metricsChart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [1, 2, 3, 4, 5],
                    datasets: [{
                        label: 'Training Loss',
                        data: [0.9, 0.7, 0.5, 0.4, 0.3],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });

            fetchRuns();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_template)


@app.get("/api/runs")
async def list_runs():
    """
    List runs from MLflow directory.
    """
    runs = []
    if not MLRUNS_DIR.exists():
        return []

    # Simple directory traversal to find runs (simplified version of MLflow API)
    for exp_dir in MLRUNS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name != ".trash":
            for run_dir in exp_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "meta.yaml").exists():
                    # For now, just return ids and names
                    runs.append(
                        {
                            "id": run_dir.name,
                            "name": f"Experiment {exp_dir.name}",
                            "status": "Finished",  # Simplified
                        }
                    )

    return runs[:10]  # Return last 10 runs


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
