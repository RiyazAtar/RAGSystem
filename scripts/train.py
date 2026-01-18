"""
Training Pipeline for Marketing AI System
Trains all models and prepares the system
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root and src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from data_processor import DataProcessor
from discount_model import DiscountPredictor
from rag_system import RAGSystem
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_discount_model(df, processor, models_dir):
    """Train discount prediction model"""
    print("\n" + "=" * 60)
    print("📊 TRAINING DISCOUNT PREDICTION MODEL")
    print("=" * 60)
    
    X, y = processor.prepare_features(df, fit=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {len(processor.feature_columns)}")
    
    config = {
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'enable_mlflow': False
    }
    
    model = DiscountPredictor(model_type="xgboost", config=config)
    metrics = model.train(X_train, y_train, X_val, y_val, processor.feature_columns)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_metrics = model.cross_validate(X, y, cv=5)
    
    # Save
    model.save(os.path.join(models_dir, 'discount_model.joblib'))
    processor.save(os.path.join(models_dir, 'processor.joblib'))
    
    print(model.get_summary())
    
    if model.feature_importance:
        print("\n📈 Top 10 Feature Importances:")
        for i, (feat, imp) in enumerate(list(model.feature_importance.items())[:10], 1):
            bar = "█" * int(imp * 50)
            print(f"  {i:2}. {feat:30} {imp:.4f} {bar}")
    
    return model, metrics, cv_metrics


def build_rag(df, models_dir):
    """Build RAG system"""
    print("\n" + "=" * 60)
    print("🔍 BUILDING RAG SYSTEM")
    print("=" * 60)
    
    rag = RAGSystem({'top_k': 5, 'similarity_threshold': 0.3, 'use_llm': False})
    rag.initialize()
    rag.index_products(df, 'rag_document')
    
    rag_dir = os.path.join(models_dir, 'rag_system')
    os.makedirs(rag_dir, exist_ok=True)
    rag.save(rag_dir)
    
    # Test queries
    print("\n💬 Testing RAG:")
    test_questions = [
        "What are some good USB cables?",
        "Recommend earbuds under 2000 rupees",
        "Which products have best ratings?"
    ]
    
    for q in test_questions:
        response = rag.answer(q)
        print(f"\n📌 Q: {q}")
        answer_preview = response['answer'][:150] + "..." if len(response['answer']) > 150 else response['answer']
        print(f"💬 A: {answer_preview}")
        print(f"📊 Confidence: {response['confidence']}")
    
    return rag


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("🚀 MARKETING AI SYSTEM - TRAINING PIPELINE")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().isoformat()}")

    # Use project root instead of script directory
    base_dir = project_root
    data_path = os.path.join(base_dir, 'data', 'amazon_sales.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return
    
    # Load and process data
    print(f"\n📂 Loading data from {data_path}")
    processor = DataProcessor()
    df = processor.load_and_clean(data_path)
    df = processor.engineer_features(df)
    df = processor.prepare_for_rag(df)
    
    print(f"✅ Loaded {len(df)} unique products")
    print(f"📊 Categories: {df['main_category'].nunique()}")
    print(f"🏷️ Brands: {df['brand'].nunique()}")
    
    print("\n📦 Category Distribution:")
    for cat, count in df['main_category'].value_counts().items():
        print(f"   • {cat}: {count}")
    
    print("\n💰 Price Statistics:")
    print(f"   • Min: ₹{df['actual_price_clean'].min():,.0f}")
    print(f"   • Max: ₹{df['actual_price_clean'].max():,.0f}")
    print(f"   • Mean: ₹{df['actual_price_clean'].mean():,.0f}")
    
    print("\n🏷️ Discount Statistics:")
    print(f"   • Min: {df['discount_percentage'].min():.0f}%")
    print(f"   • Max: {df['discount_percentage'].max():.0f}%")
    print(f"   • Mean: {df['discount_percentage'].mean():.1f}%")
    
    # Train model
    model, metrics, cv_metrics = train_discount_model(df, processor, models_dir)
    
    # Build RAG
    rag = build_rag(df, models_dir)
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total_products': len(df),
            'categories': df['main_category'].nunique(),
            'brands': df['brand'].nunique(),
            'price_range': [float(df['actual_price_clean'].min()), float(df['actual_price_clean'].max())],
            'discount_range': [float(df['discount_percentage'].min()), float(df['discount_percentage'].max())]
        },
        'model': {
            'train_rmse': metrics.get('train_rmse'),
            'val_rmse': metrics.get('val_rmse'),
            'train_r2': metrics.get('train_r2'),
            'val_r2': metrics.get('val_r2'),
            'cv_rmse': cv_metrics.get('cv_rmse_mean')
        },
        'rag': rag.get_stats()
    }
    
    with open(os.path.join(models_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📁 Models saved to: {models_dir}")
    print(f"📊 Discount Model RMSE: {metrics.get('val_rmse', 'N/A'):.2f}%")
    print(f"🔍 RAG Index: {rag.get_stats()['total_documents']} documents")
    print(f"⏰ Completed: {datetime.now().isoformat()}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    main()
