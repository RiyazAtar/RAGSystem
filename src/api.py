"""
FastAPI Application for Marketing AI System
Complete REST API with all endpoints
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_processor import DataProcessor
from discount_model import DiscountPredictor
from rag_system import RAGSystem
from monitoring import SystemMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Pydantic Models ====================

class ProductInput(BaseModel):
    """Input for discount prediction"""
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Category (e.g., 'Electronics|Headphones')")
    actual_price: float = Field(..., ge=0, description="Original price")
    rating: float = Field(default=4.0, ge=0, le=5)
    rating_count: int = Field(default=100, ge=0)
    about_product: str = Field(default="")
    review_content: str = Field(default="")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "boAt Rockerz 450 Bluetooth Headphones",
                "category": "Electronics|Headphones",
                "actual_price": 2990,
                "rating": 4.1,
                "rating_count": 45000
            }
        }


class PredictionResponse(BaseModel):
    """Discount prediction response"""
    predicted_discount: float
    confidence_lower: float
    confidence_upper: float
    category: str
    explanation: str
    recommendation: str
    top_factors: List[Dict[str, Any]]
    prediction_time: float


class QuestionInput(BaseModel):
    """Input for RAG Q&A"""
    question: str = Field(..., min_length=3)
    session_id: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = Field(default=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the best wireless earbuds under 2000 rupees?",
                "session_id": "user123"
            }
        }


class AnswerResponse(BaseModel):
    """RAG answer response"""
    answer: str
    confidence: str
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str
    conversation_turns: int
    retrieval_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    components: Dict[str, str]
    uptime_seconds: float


# ==================== Application State ====================

class AppState:
    def __init__(self):
        self.processor: Optional[DataProcessor] = None
        self.model: Optional[DiscountPredictor] = None
        self.rag: Optional[RAGSystem] = None
        self.monitor: Optional[SystemMonitor] = None
        self.df: Optional[pd.DataFrame] = None
        self.start_time = datetime.now()

state = AppState()


# ==================== Startup/Shutdown ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    logger.info("=" * 50)
    logger.info("Starting Marketing AI System...")
    logger.info("=" * 50)
    
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(base_dir, 'data', 'amazon_sales.csv')
        models_dir = os.path.join(base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize monitor with Prometheus enabled
        state.monitor = SystemMonitor({'enable_prometheus': True, 'prometheus_port': 8001})

        # Start Prometheus metrics server
        if state.monitor.metrics_collector.enable_prometheus:
            state.monitor.metrics_collector.start_prometheus_server()

        # Load/process data
        state.processor = DataProcessor()
        if os.path.exists(data_path):
            state.df = state.processor.load_and_clean(data_path)
            state.df = state.processor.engineer_features(state.df)
            state.df = state.processor.prepare_for_rag(state.df)
            logger.info(f"Loaded {len(state.df)} products")
        
        # Load/train model
        model_path = os.path.join(models_dir, 'discount_model.joblib')
        processor_path = os.path.join(models_dir, 'processor.joblib')
        
        state.model = DiscountPredictor(model_type="xgboost")
        
        if os.path.exists(model_path) and os.path.exists(processor_path):
            state.model.load(model_path)
            state.processor.load(processor_path)
            logger.info("Loaded existing model")
        elif state.df is not None:
            logger.info("Training model...")
            X, y = state.processor.prepare_features(state.df, fit=True)
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            state.model.train(X_train, y_train, X_val, y_val, state.processor.feature_columns)
            state.model.save(model_path)
            state.processor.save(processor_path)
            
            # Set reference for drift detection
            state.monitor.drift_detector.set_reference(y_val, X_val)
        
        # Initialize RAG
        rag_path = os.path.join(models_dir, 'rag_system')
        state.rag = RAGSystem({'top_k': 5, 'similarity_threshold': 0.0, 'use_llm': False})
        
        if os.path.exists(os.path.join(rag_path, 'vector_store.pkl')):
            state.rag.load(rag_path)
            logger.info("Loaded RAG system")
        else:
            state.rag.initialize()
            if state.df is not None:
                state.rag.index_products(state.df, 'rag_document')
                os.makedirs(rag_path, exist_ok=True)
                state.rag.save(rag_path)
        
        logger.info("=" * 50)
        logger.info("System ready!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    logger.info("Shutting down...")


# ==================== FastAPI App ====================

app = FastAPI(
    title="Marketing AI System",
    description="""
## 🚀 AI-Powered Marketing Intelligence Platform

### Features
- **🎯 Discount Prediction**: ML-based optimal discount recommendations
- **💬 Product Q&A**: RAG-powered conversational assistant
- **📊 Monitoring**: Drift detection, metrics, health checks
- **🔄 Auto-Retraining**: Continuous model improvement

### Endpoints
- `POST /predict_discount` - Predict optimal discount
- `POST /answer_question` - Ask product questions (with conversation history)
- `GET /health` - System health
- `GET /metrics` - Performance metrics
- `GET /conversation/{session_id}` - Get conversation history
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files for UI
ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui')
if os.path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


# ==================== Helper Functions ====================

def prepare_input(product: ProductInput) -> np.ndarray:
    """Prepare single product for prediction"""
    # Get valid values from training data if available
    valid_main_cats = set()
    valid_sub_cats = set()
    valid_brands = set()

    if state.df is not None:
        valid_main_cats = set(state.df['main_category'].unique())
        if 'sub_category' in state.df.columns:
            valid_sub_cats = set(state.df['sub_category'].unique())
        if 'brand' in state.df.columns:
            valid_brands = set(state.df['brand'].unique())

    parts = product.category.split('|')
    main_cat = parts[0] if len(parts) > 0 and parts[0] else "Electronics"
    sub_cat = parts[1] if len(parts) > 1 and parts[1] else "Accessories&Peripherals"
    brand = product.product_name.split()[0] if product.product_name else "boAt"

    # Validate against training data
    if valid_main_cats and main_cat not in valid_main_cats:
        main_cat = "Electronics" if "Electronics" in valid_main_cats else list(valid_main_cats)[0]

    if valid_sub_cats and sub_cat not in valid_sub_cats:
        sub_cat = "Accessories&Peripherals" if "Accessories&Peripherals" in valid_sub_cats else list(valid_sub_cats)[0]

    if valid_brands and brand not in valid_brands:
        brand = "boAt" if "boAt" in valid_brands else list(valid_brands)[0]
    
    # Price bucket
    if product.actual_price < 500:
        price_bucket = 'very_low'
    elif product.actual_price < 2000:
        price_bucket = 'low'
    elif product.actual_price < 10000:
        price_bucket = 'medium'
    elif product.actual_price < 50000:
        price_bucket = 'high'
    else:
        price_bucket = 'premium'
    
    # Rating bucket
    if product.rating < 2.5:
        rating_bucket = 'poor'
    elif product.rating < 3.5:
        rating_bucket = 'average'
    elif product.rating < 4.0:
        rating_bucket = 'good'
    elif product.rating < 4.5:
        rating_bucket = 'very_good'
    else:
        rating_bucket = 'excellent'
    
    features = {
        'main_category': main_cat,
        'sub_category': sub_cat,
        'brand': brand,
        'price_bucket': price_bucket,
        'rating_bucket': rating_bucket,
        'actual_price_clean': product.actual_price,
        'rating_clean': product.rating,
        'rating_count_log': np.log1p(product.rating_count),
        'about_product_length': len(product.about_product),
        'review_content_length': len(product.review_content),
        'price_log': np.log1p(product.actual_price),
        'category_popularity': 100,
        'brand_popularity': 50,
        'high_engagement': 1 if product.rating_count > 1000 else 0,
        'price_rating_interaction': np.log1p(product.actual_price) * product.rating,
        'engagement_score': product.rating * np.log1p(product.rating_count)
    }
    
    df = pd.DataFrame([features])
    X, _ = state.processor.prepare_features(df, fit=False)
    return X


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to docs or show simple landing"""
    return """
    <html>
        <head><title>Marketing AI System</title></head>
        <body>
            <h1>🚀 Marketing AI System</h1>
            <p>AI-powered marketing intelligence platform</p>
            <ul>
                <li><a href="/docs">📖 API Documentation</a></li>
                <li><a href="/health">❤️ Health Check</a></li>
                <li><a href="/ui">🖥️ Web Interface</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    health_data = state.monitor.get_health() if state.monitor else {}
    
    components = {
        "model": "healthy" if state.model and state.model.model else "unavailable",
        "rag": "healthy" if state.rag and state.rag.vector_store else "unavailable",
        "data": "healthy" if state.df is not None else "unavailable"
    }
    
    status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        components=components,
        uptime_seconds=(datetime.now() - state.start_time).total_seconds()
    )


@app.post("/predict_discount", response_model=PredictionResponse)
async def predict_discount(product: ProductInput, background_tasks: BackgroundTasks):
    """Predict optimal discount for a product"""
    if state.model is None or state.model.model is None:
        raise HTTPException(503, "Model not available")
    
    start = datetime.now()
    
    try:
        X = prepare_input(product)
        pred, lower, upper = state.model.predict_with_confidence(X)
        prediction = float(pred[0])
        
        explanation = state.model.explain_prediction(prediction)
        
        latency = (datetime.now() - start).total_seconds()
        
        # Record metrics
        if state.monitor:
            state.monitor.metrics_collector.record_prediction(prediction, latency)
            state.monitor.drift_detector.record(prediction)
        
        return PredictionResponse(
            predicted_discount=round(prediction, 2),
            confidence_lower=round(float(lower[0]), 2),
            confidence_upper=round(float(upper[0]), 2),
            category=explanation['category'],
            explanation=explanation['explanation'],
            recommendation=explanation['recommendation'],
            top_factors=explanation['top_factors'],
            prediction_time=latency
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        if state.monitor:
            try:
                state.monitor.metrics_collector.record_error("prediction_error", str(e))
            except:
                pass
        raise HTTPException(500, str(e))


@app.post("/answer_question", response_model=AnswerResponse)
async def answer_question(query: QuestionInput, background_tasks: BackgroundTasks):
    """Answer product questions with conversation history"""
    if state.rag is None or state.rag.vector_store is None:
        raise HTTPException(503, "RAG system not available")
    
    try:
        original_k = state.rag.top_k
        state.rag.top_k = query.top_k
        
        response = state.rag.answer(
            query.question,
            session_id=query.session_id,
            include_sources=query.include_sources
        )
        
        state.rag.top_k = original_k
        
        # Record metrics
        if state.monitor:
            state.monitor.metrics_collector.record_query(
                query.question, 
                response['retrieval_time'],
                response['confidence'],
                query.session_id
            )
        
        return AnswerResponse(
            answer=response['answer'],
            confidence=response['confidence'],
            sources=response.get('sources'),
            session_id=response['session_id'],
            conversation_turns=response['conversation_turns'],
            retrieval_time=response['retrieval_time']
        )
    
    except Exception as e:
        if state.monitor:
            state.monitor.metrics_collector.record_error("rag_error", str(e))
        raise HTTPException(500, str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Get conversation history for a session"""
    if state.rag is None:
        raise HTTPException(503, "RAG system not available")
    
    return state.rag.get_conversation(session_id)


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history"""
    if state.rag is None:
        raise HTTPException(503, "RAG system not available")
    
    state.rag.clear_conversation(session_id)
    return {"status": "success", "message": f"Cleared history for {session_id}"}


@app.get("/conversations")
async def list_conversations():
    """List all active sessions"""
    if state.rag is None:
        raise HTTPException(503, "RAG system not available")
    
    return {"sessions": state.rag.get_all_sessions()}


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    if state.monitor is None:
        raise HTTPException(503, "Monitoring not available")
    
    return state.monitor.get_full_report()


@app.get("/drift")
async def check_drift():
    """Check for model drift"""
    if state.monitor is None:
        raise HTTPException(503, "Monitoring not available")
    
    return state.monitor.drift_detector.check_drift()


@app.get("/products")
async def list_products(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: Optional[str] = None
):
    """List products"""
    if state.df is None:
        raise HTTPException(503, "Data not available")
    
    df = state.df
    if category:
        df = df[df['main_category'].str.contains(category, case=False, na=False)]
    
    products = df.iloc[offset:offset+limit][
        ['product_id', 'product_name', 'main_category', 'discounted_price', 
         'actual_price', 'rating_clean', 'discount_percentage', 'brand']
    ].to_dict(orient='records')
    
    return {"total": len(df), "products": products}


@app.get("/categories")
async def list_categories():
    """List categories"""
    if state.df is None:
        raise HTTPException(503, "Data not available")
    
    return {"categories": state.df['main_category'].value_counts().to_dict()}


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "products": len(state.df) if state.df is not None else 0,
        "categories": state.df['main_category'].nunique() if state.df is not None else 0,
        "brands": state.df['brand'].nunique() if state.df is not None else 0,
        "model_metrics": state.model.metrics if state.model else {},
        "rag_stats": state.rag.get_stats() if state.rag else {},
        "uptime_seconds": (datetime.now() - state.start_time).total_seconds()
    }


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    logger.error(f"Error: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
