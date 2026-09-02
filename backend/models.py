from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendRequest(BaseModel):
    """Request model for product recommendations."""
    occasion: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The occasion to get recommendations for",
        examples=["Diwali"]
    )
    top_k: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Number of top results to return",
        examples=[12]
    )


class ProductResult(BaseModel):
    """Single product recommendation result."""
    product_id: str = Field(description="Unique product identifier")
    title: str = Field(description="Product title")
    category: str = Field(default="", description="Product category description")
    description_snippet: str = Field(default="", description="Product description")
    image_url: str = Field(default="", description="Product image URL")
    brand: str = Field(default="", description="Product brand")
    price: str = Field(default="", description="Product price")
    score: float = Field(description="Similarity score (0-1)")


class RecommendResponse(BaseModel):
    """Response model for product recommendations."""
    occasion: str = Field(description="The queried occasion")
    total_results: int = Field(description="Number of results returned")
    results: List[ProductResult] = Field(description="Ranked product recommendations")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    catalog_count: int = Field(description="Number of products in catalog")
    model_loaded: bool = Field(description="Whether the ML model is loaded")
