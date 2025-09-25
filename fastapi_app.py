from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
from inference import DrugAnalyzer
from load_model import load_model_components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Drug Repurposing API",
    description="GNN + BioBERT Drug Repurposing for Alzheimer's Disease",
    version="1.0.0"
)

# Request models
class DrugAnalysisRequest(BaseModel):
    drug_ids: List[str]

# Global analyzer
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Initialize analyzer on startup"""
    global analyzer
    try:
        logger.info("Loading model components...")
        model_components = load_model_components()
        
        if model_components:
            analyzer = DrugAnalyzer(model_components)
            logger.info("Drug analyzer initialized successfully")
        else:
            logger.error("Failed to load model components")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Drug Repurposing API is running",
        "analyzer_ready": analyzer is not None
    }

@app.post("/analyze")
async def analyze_drugs(request: DrugAnalysisRequest):
    """Main drug analysis endpoint"""
    if analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )

    if not request.drug_ids:
        raise HTTPException(
            status_code=400, 
            detail="No drug IDs provided"
        )

    # Limit to 5 drugs
    drug_ids = request.drug_ids[:5]
    
    logger.info(f"Analyzing drugs: {drug_ids}")
    
    results = []
    for drug_id in drug_ids:
        try:
            result = analyzer.analyze_drug(drug_id)
            results.append(result)
        except Exception as e:
            logger.error(f"Error analyzing drug {drug_id}: {e}")
            results.append({
                "drug_id": drug_id,
                "error": f"Analysis failed: {str(e)}",
                "status": "failed"
            })

    return {
        "results": results,
        "status": "success",
        "total_analyzed": len(results)
    }

@app.get("/drug/{drug_id}")
async def analyze_single_drug(drug_id: str):
    """Analyze a single drug"""
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        result = analyzer.analyze_drug(drug_id)
        return result
    except Exception as e:
        logger.error(f"Error analyzing drug {drug_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/status")
async def get_status():
    """Get API status"""
    status_info = {
        "api_status": "running",
        "model_loaded": analyzer is not None,
        "supported_operations": ["analyze", "single_drug_analysis"],
        "max_drugs_per_request": 5
    }
    
    if analyzer:
        status_info.update({
            "entities_count": len(analyzer.entity_to_id) if hasattr(analyzer, 'entity_to_id') else 'unknown',
            "model_ready": True
        })
    
    return status_info

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)