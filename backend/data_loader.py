import json
from pathlib import Path
from typing import List, Dict, Any


def extract_image(payload: Dict[str, Any]) -> str:
    """Extract image URL from product payload, supporting multiple formats."""
    # Support files array (assignment format)
    files = payload.get("files", [])
    if files and isinstance(files, list) and len(files) > 0:
        first_file = files[0]
        if isinstance(first_file, str):
            return first_file
        elif isinstance(first_file, dict):
            return first_file.get("url") or first_file.get("path") or ""

    # Support mainImage (actual data format)
    return payload.get("mainImage") or payload.get("image") or ""


def normalize_product(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize product fields to support both assignment and actual data formats."""
    return {
        "product_id": payload.get("product_id") or payload.get("productId") or payload.get("id"),
        "vendor_id": payload.get("vendor_id") or payload.get("vendorId") or payload.get("shopId"),
        "vendor_category": payload.get("vendor_category") or payload.get("categoryId") or "",
        "vendor_category_desc": payload.get("vendor_category_desc") or payload.get("category_desc") or "",
        "title": payload.get("title") or payload.get("name") or "",
        "description": payload.get("description") or "",
        "short_description": payload.get("shortDescription") or "",
        "image_url": extract_image(payload),
        "brand": payload.get("brand") or "",
        "price": payload.get("price") or "",
        "status": payload.get("status") or "ACTIVE",
        "slug": payload.get("slug") or "",
    }


def build_product_text(product: Dict[str, Any]) -> str:
    """Build semantic text from product fields for embedding."""
    parts = [
        product.get("title", ""),
        product.get("vendor_category_desc", ""),
        product.get("description", ""),
        product.get("short_description", ""),
        product.get("brand", ""),
    ]
    return " ".join(filter(None, parts))


def load_catalog(file_path: str) -> List[Dict[str, Any]]:
    """Load and normalize the product catalog from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Catalog must be a JSON array of products")

    # Normalize all products
    normalized = [normalize_product(p) for p in raw_data]

    # Add semantic text for embedding
    for product in normalized:
        product["semantic_text"] = build_product_text(product)

    return normalized


def filter_active_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only active products."""
    return [p for p in products if p.get("status") == "ACTIVE"]
