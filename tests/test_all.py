"""
Comprehensive Test Suite for Marketing AI System
Includes unit, integration, and load tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import time
import asyncio
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor
from discount_model import DiscountPredictor
from rag_system import RAGSystem, VectorStore, ConversationManager
from monitoring import MetricsCollector, DriftDetector, SystemMonitor


# ==================== Fixtures ====================

@pytest.fixture
def sample_data():
    """Generate sample data"""
    np.random.seed(42)
    n = 100
    
    categories = ['Electronics|Headphones', 'Computers|Cables', 'Home|Kitchen']
    brands = ['boAt', 'Samsung', 'Philips', 'Mi', 'Anker']
    
    return pd.DataFrame({
        'product_id': [f'B0{i:08d}' for i in range(n)],
        'product_name': [f'{np.random.choice(brands)} Product {i}' for i in range(n)],
        'category': [np.random.choice(categories) for _ in range(n)],
        'discounted_price': [f'₹{np.random.randint(100, 5000):,}' for _ in range(n)],
        'actual_price': [f'₹{np.random.randint(500, 10000):,}' for _ in range(n)],
        'discount_percentage': [f'{np.random.randint(10, 70)}%' for _ in range(n)],
        'rating': [str(round(np.random.uniform(3, 5), 1)) for _ in range(n)],
        'rating_count': [f'{np.random.randint(100, 50000):,}' for _ in range(n)],
        'about_product': ['Product description'] * n,
        'user_id': [f'U{i}' for i in range(n)],
        'user_name': ['User'] * n,
        'review_id': [f'R{i}' for i in range(n)],
        'review_title': ['Great'] * n,
        'review_content': ['Great product review'] * n,
        'img_link': ['http://example.com/img.jpg'] * n,
        'product_link': ['http://example.com/product'] * n
    })


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def trained_model(sample_data, processor):
    df = processor.load_and_clean(sample_data)
    df = processor.engineer_features(df)
    X, y = processor.prepare_features(df, fit=True)
    
    model = DiscountPredictor(model_type="gradient_boosting")
    model.train(X, y, feature_names=processor.feature_columns)
    
    return model, processor


# ==================== Unit Tests: Data Processor ====================

class TestDataProcessor:
    """Test data processing"""
    
    def test_clean_price(self, processor):
        assert processor.clean_price('₹1,999') == 1999.0
        assert processor.clean_price('₹500') == 500.0
        assert processor.clean_price(None) == 0.0
    
    def test_clean_percentage(self, processor):
        assert processor.clean_percentage('50%') == 50.0
        assert processor.clean_percentage('25%') == 25.0
        assert processor.clean_percentage(None) == 0.0
    
    def test_clean_rating(self, processor):
        assert processor.clean_rating('4.2') == 4.2
        assert processor.clean_rating(None) == 0.0
    
    def test_clean_rating_count(self, processor):
        assert processor.clean_rating_count('24,269') == 24269
        assert processor.clean_rating_count(None) == 0
    
    def test_extract_categories(self, processor):
        main, sub, subsub = processor.extract_categories('Electronics|Headphones|Wireless')
        assert main == 'Electronics'
        assert sub == 'Headphones'
        assert subsub == 'Wireless'
    
    def test_extract_brand(self, processor):
        assert processor.extract_brand('boAt Rockerz') == 'boAt'
        assert processor.extract_brand('Samsung Galaxy') == 'Samsung'
        assert processor.extract_brand(None) == 'Unknown'
    
    def test_load_and_clean(self, processor, sample_data):
        df = processor.load_and_clean(sample_data)
        
        assert 'actual_price_clean' in df.columns
        assert 'main_category' in df.columns
        assert 'brand' in df.columns
        assert len(df) > 0
    
    def test_engineer_features(self, processor, sample_data):
        df = processor.load_and_clean(sample_data)
        df = processor.engineer_features(df)
        
        assert 'price_log' in df.columns
        assert 'price_bucket' in df.columns
        assert 'rating_bucket' in df.columns
    
    def test_prepare_features(self, processor, sample_data):
        df = processor.load_and_clean(sample_data)
        df = processor.engineer_features(df)
        X, y = processor.prepare_features(df, fit=True)
        
        assert len(X) == len(y)
        assert X.shape[1] > 0
        assert y.min() >= 0


# ==================== Unit Tests: Discount Model ====================

class TestDiscountModel:
    """Test discount prediction model"""
    
    def test_model_creation(self):
        model = DiscountPredictor(model_type="gradient_boosting")
        assert model.model_type == "gradient_boosting"
        assert model.model is None
    
    def test_training(self, trained_model):
        model, processor = trained_model
        
        assert model.model is not None
        assert 'train_rmse' in model.metrics
        assert 'train_r2' in model.metrics
    
    def test_prediction(self, trained_model):
        model, processor = trained_model
        
        X_test = np.random.randn(5, len(processor.feature_columns))
        preds = model.predict(X_test)
        
        assert len(preds) == 5
        assert all(0 <= p <= 100 for p in preds)
    
    def test_prediction_with_confidence(self, trained_model):
        model, processor = trained_model
        
        X = np.random.randn(3, len(processor.feature_columns))
        pred, lower, upper = model.predict_with_confidence(X)
        
        assert len(pred) == 3
        assert all(lower[i] <= pred[i] <= upper[i] for i in range(3))
    
    def test_explanation(self, trained_model):
        model, _ = trained_model
        
        explanation = model.explain_prediction(35.0)
        
        assert 'predicted_discount' in explanation
        assert 'category' in explanation
        assert 'recommendation' in explanation
    
    def test_save_load(self, trained_model, tmp_path):
        model, _ = trained_model
        
        path = tmp_path / "model.joblib"
        model.save(str(path))
        
        new_model = DiscountPredictor()
        new_model.load(str(path))
        
        assert new_model.model_type == model.model_type


# ==================== Unit Tests: RAG System ====================

class TestRAGSystem:
    """Test RAG system"""
    
    def test_vector_store(self):
        store = VectorStore(dimension=4)
        
        docs = [{'id': '1', 'text': 'doc1'}, {'id': '2', 'text': 'doc2'}]
        embeddings = np.random.randn(2, 4).astype('float32')
        
        store.add_documents(docs, embeddings)
        
        assert len(store.documents) == 2
    
    def test_vector_search(self):
        store = VectorStore(dimension=4)
        
        docs = [{'id': '1'}, {'id': '2'}, {'id': '3'}]
        embeddings = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype='float32')
        
        store.add_documents(docs, embeddings)
        
        query = np.array([0.9, 0.1, 0, 0], dtype='float32')
        results = store.search(query, top_k=2)
        
        assert len(results) == 2
    
    def test_conversation_manager(self):
        cm = ConversationManager(max_history=5)
        
        cm.add_message("session1", "user", "Hello")
        cm.add_message("session1", "assistant", "Hi!")
        
        history = cm.get_history("session1")
        assert len(history) == 2
        assert history[0]['role'] == 'user'
    
    def test_conversation_context(self):
        cm = ConversationManager()
        
        cm.add_message("s1", "user", "Question 1")
        cm.add_message("s1", "assistant", "Answer 1")
        
        context = cm.get_context("s1")
        assert "Question 1" in context
    
    def test_rag_initialization(self):
        rag = RAGSystem({'use_llm': False})
        rag.initialize()
        
        assert rag.embedding_model is not None
        assert rag.vector_store is not None


# ==================== Unit Tests: Monitoring ====================

class TestMonitoring:
    """Test monitoring components"""
    
    def test_metrics_collector(self):
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.record_prediction(35.0, 0.05)
        collector.record_query("test", 0.1, "high", "session1")
        
        summary = collector.get_summary()
        assert summary['total_predictions'] == 1
        assert summary['total_queries'] == 1
    
    def test_drift_detector(self):
        detector = DriftDetector()
        
        preds = np.random.uniform(20, 60, 100)
        detector.set_reference(preds)
        
        assert detector.reference_stats is not None
        assert 'pred_mean' in detector.reference_stats
    
    def test_drift_detection(self):
        detector = DriftDetector()
        
        ref_preds = np.random.uniform(30, 50, 100)
        detector.set_reference(ref_preds)
        
        # Add similar predictions
        for p in np.random.uniform(30, 50, 150):
            detector.record(p)
        
        drift = detector.check_drift()
        assert 'drift_detected' in drift
    
    def test_system_monitor(self):
        monitor = SystemMonitor({'enable_prometheus': False})
        
        health = monitor.get_health()
        assert 'status' in health


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests"""
    
    def test_full_prediction_pipeline(self, sample_data):
        processor = DataProcessor()
        df = processor.load_and_clean(sample_data)
        df = processor.engineer_features(df)
        X, y = processor.prepare_features(df, fit=True)
        
        model = DiscountPredictor(model_type="gradient_boosting")
        model.train(X, y)
        
        preds = model.predict(X[:5])
        
        assert len(preds) == 5
        assert all(0 <= p <= 100 for p in preds)
    
    def test_full_rag_pipeline(self, sample_data, processor):
        df = processor.load_and_clean(sample_data)
        df = processor.prepare_for_rag(df)
        
        rag = RAGSystem({'use_llm': False})
        rag.initialize()
        rag.index_products(df, 'rag_document')
        
        response = rag.answer("Recommend a product")
        
        assert 'answer' in response
        assert 'confidence' in response
    
    def test_conversation_flow(self, sample_data, processor):
        df = processor.load_and_clean(sample_data)
        df = processor.prepare_for_rag(df)
        
        rag = RAGSystem({'use_llm': False})
        rag.initialize()
        rag.index_products(df, 'rag_document')
        
        # First message
        r1 = rag.answer("What products are available?", session_id="test")
        assert r1['conversation_turns'] == 1
        
        # Second message (should have context)
        r2 = rag.answer("Tell me more", session_id="test")
        assert r2['conversation_turns'] == 2
        
        # Check history
        conv = rag.get_conversation("test")
        assert len(conv['messages']) == 4


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance tests"""
    
    def test_prediction_latency(self, trained_model):
        model, processor = trained_model
        
        X = np.random.randn(100, len(processor.feature_columns))
        
        start = time.time()
        model.predict(X)
        latency = time.time() - start
        
        assert latency < 1.0  # Should be under 1 second
    
    def test_batch_scaling(self, trained_model):
        model, processor = trained_model
        
        latencies = []
        for batch_size in [10, 50, 100]:
            X = np.random.randn(batch_size, len(processor.feature_columns))
            
            start = time.time()
            model.predict(X)
            latencies.append(time.time() - start)
        
        # Should scale reasonably
        assert latencies[2] < latencies[0] * 20


# ==================== Safety Tests ====================

class TestSafety:
    """Safety and validation tests"""
    
    def test_prediction_bounds(self, trained_model):
        model, processor = trained_model
        
        # Extreme inputs
        X = np.random.randn(50, len(processor.feature_columns)) * 100
        preds = model.predict(X)
        
        assert all(0 <= p <= 100 for p in preds)
    
    def test_input_validation(self):
        processor = DataProcessor()
        
        assert processor.clean_price(None) == 0.0
        assert processor.clean_percentage(None) == 0.0
        assert processor.clean_rating(None) == 0.0
    
    def test_empty_conversation(self):
        cm = ConversationManager()
        
        history = cm.get_history("nonexistent")
        assert history == []
        
        context = cm.get_context("nonexistent")
        assert context == ""


# ==================== API Tests ====================

class TestAPI:
    """API endpoint tests (using TestClient)"""
    
    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from api import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI or dependencies not available")
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_predict_endpoint(self, client):
        response = client.post("/predict_discount", json={
            "product_name": "Test Product",
            "category": "Electronics|Test",
            "actual_price": 1000,
            "rating": 4.0,
            "rating_count": 100
        })
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_discount" in data
            assert 0 <= data["predicted_discount"] <= 100


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
