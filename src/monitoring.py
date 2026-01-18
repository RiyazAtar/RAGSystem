"""
Monitoring Module for Marketing AI System
Includes metrics collection, drift detection, and auto-retraining
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing Prometheus
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available")


class MetricsCollector:
    """Collect and track system metrics"""
    
    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 8001):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.metrics: Dict[str, List] = {
            'predictions': [],
            'latencies': [],
            'errors': [],
            'queries': []
        }
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.prom_predictions = Counter('discount_predictions_total', 'Total discount predictions')
        self.prom_queries = Counter('rag_queries_total', 'Total RAG queries')
        self.prom_errors = Counter('errors_total', 'Total errors', ['type'])
        self.prom_latency = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])
        self.prom_prediction_value = Gauge('last_prediction_value', 'Last prediction value')
        self.prom_drift_score = Gauge('drift_score', 'Current drift score')
    
    def start_prometheus_server(self):
        """Start Prometheus HTTP server"""
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")
    
    def record_prediction(self, prediction: float, latency: float, features: Dict = None):
        """Record a prediction"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'latency': latency,
            'features': features
        }
        self.metrics['predictions'].append(record)
        
        if self.enable_prometheus:
            self.prom_predictions.inc()
            self.prom_latency.labels(endpoint='predict').observe(latency)
            self.prom_prediction_value.set(prediction)
    
    def record_query(self, query: str, latency: float, confidence: str, session_id: str):
        """Record a RAG query"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'latency': latency,
            'confidence': confidence,
            'session_id': session_id
        }
        self.metrics['queries'].append(record)
        
        if self.enable_prometheus:
            self.prom_queries.inc()
            self.prom_latency.labels(endpoint='query').observe(latency)
    
    def record_error(self, error_type: str, message: str, context: Dict = None):
        """Record an error"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message,
            'context': context
        }
        self.metrics['errors'].append(record)
        
        if self.enable_prometheus:
            self.prom_errors.labels(type=error_type).inc()
    
    def get_latency_stats(self, endpoint: str = 'all') -> Dict[str, float]:
        """Get latency statistics"""
        if endpoint == 'predict':
            latencies = [p['latency'] for p in self.metrics['predictions'][-100:]]
        elif endpoint == 'query':
            latencies = [q['latency'] for q in self.metrics['queries'][-100:]]
        else:
            latencies = [p['latency'] for p in self.metrics['predictions'][-100:]]
            latencies += [q['latency'] for q in self.metrics['queries'][-100:]]
        
        if not latencies:
            return {'avg': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        return {
            'avg': float(np.mean(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'count': len(latencies)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'total_predictions': len(self.metrics['predictions']),
            'total_queries': len(self.metrics['queries']),
            'total_errors': len(self.metrics['errors']),
            'prediction_latency': self.get_latency_stats('predict'),
            'query_latency': self.get_latency_stats('query'),
            'recent_errors': self.metrics['errors'][-5:]
        }


class DriftDetector:
    """Detect data and model drift"""
    
    def __init__(self, window_size: int = 1000, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_stats: Optional[Dict] = None
        self.prediction_window: deque = deque(maxlen=window_size)
        self.feature_windows: Dict[str, deque] = {}
        self.drift_history: List[Dict] = []
    
    def set_reference(self, predictions: np.ndarray, features: np.ndarray = None):
        """Set reference distribution"""
        self.reference_stats = {
            'pred_mean': float(np.mean(predictions)),
            'pred_std': float(np.std(predictions)),
            'pred_min': float(np.min(predictions)),
            'pred_max': float(np.max(predictions)),
            'timestamp': datetime.now().isoformat()
        }
        
        if features is not None:
            self.reference_stats['feature_means'] = np.mean(features, axis=0).tolist()
            self.reference_stats['feature_stds'] = np.std(features, axis=0).tolist()
        
        logger.info("Reference distribution set")
    
    def record(self, prediction: float, features: Dict = None):
        """Record a prediction for drift monitoring"""
        self.prediction_window.append(prediction)
        
        if features:
            for key, value in features.items():
                if key not in self.feature_windows:
                    self.feature_windows[key] = deque(maxlen=self.window_size)
                self.feature_windows[key].append(value)
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for drift"""
        if self.reference_stats is None:
            return {'status': 'no_reference', 'drift_detected': False}
        
        if len(self.prediction_window) < 100:
            return {'status': 'insufficient_data', 'drift_detected': False, 'samples': len(self.prediction_window)}
        
        current_preds = np.array(list(self.prediction_window))
        
        # Calculate drift metrics
        current_mean = np.mean(current_preds)
        current_std = np.std(current_preds)
        
        ref_mean = self.reference_stats['pred_mean']
        ref_std = self.reference_stats['pred_std']
        
        mean_drift = abs(current_mean - ref_mean) / (ref_std + 1e-10)
        std_drift = abs(current_std - ref_std) / (ref_std + 1e-10)
        
        # PSI (Population Stability Index) approximation
        psi = self._calculate_psi(current_preds, ref_mean, ref_std)
        
        drift_detected = mean_drift > self.threshold or std_drift > self.threshold or psi > 0.2
        
        result = {
            'status': 'ok' if not drift_detected else 'drift_detected',
            'drift_detected': drift_detected,
            'mean_drift': round(mean_drift, 4),
            'std_drift': round(std_drift, 4),
            'psi': round(psi, 4),
            'current_mean': round(current_mean, 2),
            'reference_mean': round(ref_mean, 2),
            'samples_analyzed': len(current_preds),
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            self.drift_history.append(result)
            logger.warning(f"Drift detected: mean_drift={mean_drift:.4f}, psi={psi:.4f}")
        
        return result
    
    def _calculate_psi(self, current: np.ndarray, ref_mean: float, ref_std: float) -> float:
        """Calculate Population Stability Index"""
        # Simplified PSI calculation
        bins = np.linspace(ref_mean - 3*ref_std, ref_mean + 3*ref_std, 11)
        
        # Reference distribution (normal approximation)
        ref_dist = np.diff(np.concatenate([[0], 
            [0.001, 0.02, 0.14, 0.34, 0.68, 0.84, 0.98, 0.999, 1.0]]))
        ref_dist = np.clip(ref_dist, 0.001, 1)
        
        # Current distribution
        current_counts, _ = np.histogram(current, bins=bins)
        current_dist = current_counts / (len(current) + 1e-10)
        current_dist = np.clip(current_dist, 0.001, 1)
        
        # PSI
        psi = np.sum((current_dist - ref_dist) * np.log(current_dist / ref_dist))
        
        return max(0, psi)


class AutoRetrainer:
    """Automated model retraining based on drift and performance"""
    
    def __init__(self, drift_detector: DriftDetector, metrics_collector: MetricsCollector):
        self.drift_detector = drift_detector
        self.metrics_collector = metrics_collector
        self.retrain_threshold = 0.15
        self.min_samples = 500
        self.last_retrain: Optional[datetime] = None
        self.retrain_history: List[Dict] = []
    
    def should_retrain(self) -> Dict[str, Any]:
        """Determine if retraining is needed"""
        reasons = []
        should_retrain = False
        
        # Check drift
        drift_status = self.drift_detector.check_drift()
        if drift_status.get('drift_detected', False):
            reasons.append(f"Drift detected: mean_drift={drift_status.get('mean_drift', 0):.4f}")
            should_retrain = True
        
        # Check error rate
        total_preds = len(self.metrics_collector.metrics['predictions'])
        total_errors = len(self.metrics_collector.metrics['errors'])
        if total_preds > 0:
            error_rate = total_errors / total_preds
            if error_rate > 0.05:
                reasons.append(f"High error rate: {error_rate:.2%}")
                should_retrain = True
        
        # Check time since last retrain
        if self.last_retrain:
            days_since = (datetime.now() - self.last_retrain).days
            if days_since > 30:
                reasons.append(f"Time since last retrain: {days_since} days")
                should_retrain = True
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'drift_status': drift_status,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None
        }
    
    def record_retrain(self, metrics: Dict, trigger: str):
        """Record a retraining event"""
        self.last_retrain = datetime.now()
        self.retrain_history.append({
            'timestamp': self.last_retrain.isoformat(),
            'trigger': trigger,
            'metrics': metrics
        })
        logger.info(f"Model retrained. Trigger: {trigger}")


class SystemMonitor:
    """Combined system monitoring"""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        self.metrics_collector = MetricsCollector(
            enable_prometheus=config.get('enable_prometheus', True),
            prometheus_port=config.get('prometheus_port', 8001)
        )
        
        self.drift_detector = DriftDetector(
            window_size=config.get('drift_window_size', 1000),
            threshold=config.get('drift_threshold', 0.1)
        )
        
        self.auto_retrainer = AutoRetrainer(self.drift_detector, self.metrics_collector)
        
        self.start_time = datetime.now()
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        drift = self.drift_detector.check_drift()
        latency = self.metrics_collector.get_latency_stats()
        errors = len(self.metrics_collector.metrics['errors'])
        
        # Determine health status
        if drift.get('drift_detected') or errors > 100 or latency.get('p95', 0) > 5:
            status = 'unhealthy'
        elif errors > 50 or latency.get('p95', 0) > 2:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'drift_status': drift.get('status', 'unknown'),
            'error_count': errors,
            'latency_p95': latency.get('p95', 0),
            'total_predictions': len(self.metrics_collector.metrics['predictions']),
            'total_queries': len(self.metrics_collector.metrics['queries']),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get full monitoring report"""
        return {
            'health': self.get_health(),
            'metrics': self.metrics_collector.get_summary(),
            'drift': self.drift_detector.check_drift(),
            'retrain_status': self.auto_retrainer.should_retrain()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        report = self.get_full_report()
        report['exported_at'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
