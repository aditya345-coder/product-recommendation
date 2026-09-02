import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import RecommendRequest, RecommendResponse, HealthResponse, ProductResult
from data_loader import load_catalog, filter_active_products
from recommender import OccasionRecommender


# Initialize recommender
recommender = OccasionRecommender()

# Determine paths
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
CATALOG_PATH = PROJECT_DIR / "products.json"
FRONTEND_DIR = PROJECT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and catalog on startup."""
    # Load ML model
    recommender.load_model()

    # Load and filter catalog
    if CATALOG_PATH.exists():
        all_products = load_catalog(str(CATALOG_PATH))
        active_products = filter_active_products(all_products)
        print(f"Loaded {len(active_products)} active products from {len(all_products)} total.")
        recommender.build_index(active_products)
    else:
        print(f"Warning: Catalog not found at {CATALOG_PATH}")

    yield


app = FastAPI(
    title="Occasion-Based Product Recommendation API",
    description="Semantic product recommendations based on occasion queries",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    info = recommender.get_health_info()
    return HealthResponse(
        status="ok",
        catalog_count=info["catalog_count"],
        model_loaded=info["model_loaded"],
    )


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get product recommendations for an occasion."""
    try:
        results = recommender.recommend(
            occasion=request.occasion,
            top_k=request.top_k,
        )

        product_results = [
            ProductResult(
                product_id=p["product_id"],
                title=p["title"],
                category=p.get("vendor_category_desc", ""),
                description_snippet=p.get("description", "")[:200],
                image_url=p.get("image_url", ""),
                brand=p.get("brand", ""),
                price=p.get("price", ""),
                score=p["score"],
            )
            for p in results
        ]

        return RecommendResponse(
            occasion=request.occasion,
            total_results=len(product_results),
            results=product_results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
