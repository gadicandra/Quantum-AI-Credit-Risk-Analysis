"""
FastAPI Backend untuk Quantum Finance System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import uvicorn
from datetime import datetime
import uuid

# Import quantum modules
import sys
sys.path.append('../')
from src.integrations.ionq_azure import IonQAzureBackend
from src.integrations.qcentroid_api import QCentroidAPI
from src.llm.recommendation_engine import LLMRecommender
from src.data.ingestion import load_derivatives_dataset
from src.algorithms.qae import QuantumAmplitudeEstimation

# Initialize FastAPI
app = FastAPI(
    title="Quantum Finance API",
    description="API untuk analisis derivatif menggunakan quantum computing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize backends
ionq_backend = IonQAzureBackend()
qcentroid_backend = QCentroidAPI()
llm_recommender = LLMRecommender()

# In-memory job storage (use Redis/PostgreSQL in production)
jobs_db: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class DerivativeInput(BaseModel):
    option_type: str
    style: str
    spot_price: float
    strike: float
    maturity: float
    volatility: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0


class JobRequest(BaseModel):
    derivatives: List[DerivativeInput]
    target_backend: str = "ionq.simulator"  # ionq.simulator, ionq.qpu, qcentroid
    algorithm: str = "qae"  # qae, iqae, qmc, classical_mc
    shots: int = 100


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Quantum Finance API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/backends")
async def list_backends():
    """List available quantum backends"""
    ionq_targets = ionq_backend.list_available_targets()
    return {
        "ionq": ionq_targets,
        "qcentroid": {
            "name": "QCentroid",
            "status": "available",
            "algorithms": ["qae", "qmc", "vqa"]
        }
    }


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload CSV dataset of derivatives
    """
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate columns
        required_columns = ['option_type', 'spot_price', 'strike', 'maturity', 'volatility']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert to derivatives list
        derivatives = df.to_dict('records')
        
        return {
            "success": True,
            "message": f"Uploaded {len(derivatives)} derivatives",
            "data": derivatives
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend")
async def get_recommendations(derivatives: List[DerivativeInput]):
    """
    Get LLM recommendations for best algorithm
    """
    try:
        # Convert to dataset format
        dataset = {
            'derivatives': [d.dict() for d in derivatives],
            'n_contracts': len(derivatives),
            'option_types': list(set(d.option_type for d in derivatives))
        }
        
        # Get recommendations from LLM
        recommendations = llm_recommender.recommend(dataset)
        
        return {
            "success": True,
            "recommendations": recommendations['recommendations'],
            "features": recommendations['features']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-job")
async def submit_job(request: JobRequest, background_tasks: BackgroundTasks):
    """
    Submit quantum job to IonQ or QCentroid
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Store job metadata
        jobs_db[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "backend": request.target_backend,
            "algorithm": request.algorithm,
            "n_contracts": len(request.derivatives),
            "results": None,
            "error": None
        }
        
        # Submit job in background
        background_tasks.add_task(
            process_quantum_job,
            job_id,
            request.derivatives,
            request.target_backend,
            request.algorithm,
            request.shots
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status and results of a job
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs_db[job_id]


@app.get("/jobs")
async def list_jobs():
    """
    List all jobs
    """
    return {
        "jobs": list(jobs_db.values()),
        "total": len(jobs_db)
    }


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_db[job_id]
    return {"success": True, "message": "Job deleted"}


# Background task to process quantum job
async def process_quantum_job(
    job_id: str,
    derivatives: List[DerivativeInput],
    backend: str,
    algorithm: str,
    shots: int
):
    """
    Process quantum job asynchronously
    """
    try:
        jobs_db[job_id]['status'] = 'running'
        
        results = []
        
        for deriv in derivatives:
            deriv_dict = deriv.dict()
            
            if backend.startswith('ionq'):
                # Run on IonQ
                if algorithm == 'qae':
                    result = ionq_backend.run_qae(deriv_dict, target=backend, shots=shots)
                else:
                    raise ValueError(f"Algorithm {algorithm} not supported on IonQ yet")
            
            elif backend == 'qcentroid':
                # Run on QCentroid
                result = qcentroid_backend.submit_quantum_finance_job(algorithm, deriv_dict)
            
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            results.append({
                'derivative': deriv_dict,
                'result': result
            })
        
        # Update job status
        jobs_db[job_id]['status'] = 'completed'
        jobs_db[job_id]['completed_at'] = datetime.now().isoformat()
        jobs_db[job_id]['results'] = results
    
    except Exception as e:
        jobs_db[job_id]['status'] = 'failed'
        jobs_db[job_id]['error'] = str(e)
        jobs_db[job_id]['completed_at'] = datetime.now().isoformat()


# Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
