"""
Discount Prediction Model with MLflow Tracking
Implements ensemble models with comprehensive monitoring
"""

import numpy as np
import pandas as pd
import logging
import os
import json
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from collections import deque

import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Try imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscountPredictor:
    """Discount prediction model with MLflow tracking and monitoring"""
    
    def __init__(self, model_type: str = "xgboost", config: Dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.feature_importance: Optional[Dict] = None
        self.metrics: Dict = {}
        self.training_history: List[Dict] = []
        
        # For drift detection
        self.reference_predictions: Optional[np.ndarray] = None
        self.prediction_history: deque = deque(maxlen=1000)
        
        # MLflow tracking
        self.mlflow_enabled = MLFLOW_AVAILABLE and self.config.get('enable_mlflow', True)
        self.experiment_name = self.config.get('experiment_name', 'discount_prediction')
        
    def _create_model(self):
        """Create the underlying ML model"""
        params = self.config.get('params', {})
        
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=params.get('random_state', 42),
                n_jobs=-1,
                objective='reg:squarederror'
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 5),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
        else:
            # Default to GradientBoosting
            self.model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42)
            )
        
        logger.info(f"Created {self.model_type} model")
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Train model with optional MLflow tracking"""
        logger.info(f"Training {self.model_type} model...")
        
        self._create_model()
        
        # Start MLflow run if enabled
        if self.mlflow_enabled:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config.get('params', {}))
            mlflow.log_param('model_type', self.model_type)
        
        try:
            start_time = datetime.now()
            
            # Train
            if self.model_type == "xgboost" and XGBOOST_AVAILABLE and X_val is not None:
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                self.model.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            train_preds = self.model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_preds, "train")
            
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_preds = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_preds, "val")
                self.reference_predictions = val_preds  # For drift detection
            
            # Feature importance
            if feature_names:
                self._extract_feature_importance(feature_names)
            
            self.metrics = {**train_metrics, **val_metrics, "training_time": training_time}
            
            # Log to MLflow
            if self.mlflow_enabled:
                mlflow.log_metrics(self.metrics)
                if self.feature_importance:
                    mlflow.log_dict(self.feature_importance, "feature_importance.json")
                mlflow.sklearn.log_model(self.model, "model")
            
            # Store history
            self.training_history.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics,
                "model_type": self.model_type
            })
            
            logger.info(f"Training completed in {training_time:.2f}s")
            logger.info(f"Train RMSE: {train_metrics['train_rmse']:.4f}")
            if val_metrics:
                logger.info(f"Val RMSE: {val_metrics['val_rmse']:.4f}")
            
            return self.metrics
            
        finally:
            if self.mlflow_enabled:
                mlflow.end_run()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate regression metrics"""
        y_pred = np.clip(y_pred, 0, 100)
        
        return {
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_r2": float(r2_score(y_true, y_pred))
        }
    
    def _extract_feature_importance(self, feature_names: List[str]):
        """Extract feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if len(feature_names) == len(importance):
                self.feature_importance = dict(zip(feature_names, importance.tolist()))
            else:
                self.feature_importance = {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
            self.feature_importance = dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with tracking"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions = np.clip(self.model.predict(X), 0, 100)
        
        # Track for drift detection
        for pred in predictions:
            self.prediction_history.append({
                'prediction': float(pred),
                'timestamp': datetime.now().isoformat()
            })
        
        return predictions
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
        predictions = self.predict(X)
        
        std_estimate = self.metrics.get('val_rmse', self.metrics.get('train_rmse', 10))
        lower = np.clip(predictions - 1.96 * std_estimate, 0, 100)
        upper = np.clip(predictions + 1.96 * std_estimate, 0, 100)
        
        return predictions, lower, upper
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for prediction drift"""
        if self.reference_predictions is None or len(self.prediction_history) < 100:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        recent_preds = np.array([p['prediction'] for p in list(self.prediction_history)[-100:]])
        
        ref_mean, ref_std = np.mean(self.reference_predictions), np.std(self.reference_predictions)
        current_mean, current_std = np.mean(recent_preds), np.std(recent_preds)
        
        mean_drift = abs(current_mean - ref_mean) / (ref_std + 1e-10)
        std_drift = abs(current_std - ref_std) / (ref_std + 1e-10)
        
        drift_detected = mean_drift > 0.5 or std_drift > 0.5
        
        return {
            'drift_detected': drift_detected,
            'mean_drift': float(mean_drift),
            'std_drift': float(std_drift),
            'reference_mean': float(ref_mean),
            'current_mean': float(current_mean),
            'samples_analyzed': len(recent_preds)
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        if self.model is None:
            self._create_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        
        return {
            "cv_rmse_mean": float(-scores.mean()),
            "cv_rmse_std": float(scores.std()),
            "cv_scores": list(-scores)
        }
    
    def explain_prediction(self, prediction: float) -> Dict[str, Any]:
        """Generate explanation for a prediction"""
        if prediction < 15:
            category = "minimal_discount"
            explanation = "Low discount recommended. Product has strong demand or premium positioning."
            recommendation = "Consider maintaining current price or minimal promotional discount."
        elif prediction < 35:
            category = "moderate_discount"
            explanation = "Standard promotional discount suitable."
            recommendation = "Apply 20-35% discount for effective promotion."
        elif prediction < 55:
            category = "significant_discount"
            explanation = "Higher discount indicated due to competitive pressure or inventory needs."
            recommendation = "Consider bundle offers or seasonal promotions."
        else:
            category = "deep_discount"
            explanation = "Deep discount suggested. Review pricing strategy."
            recommendation = "Evaluate clearance options or price repositioning."
        
        # Convert top factors from tuples to dictionaries
        top_factors = []
        if self.feature_importance:
            for name, value in list(self.feature_importance.items())[:5]:
                top_factors.append({'name': name, 'importance': float(value)})

        return {
            'predicted_discount': round(prediction, 2),
            'category': category,
            'explanation': explanation,
            'recommendation': recommendation,
            'top_factors': top_factors
        }
    
    def save(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics,
            'reference_predictions': self.reference_predictions,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        state = joblib.load(filepath)
        self.model = state['model']
        self.model_type = state['model_type']
        self.config = state['config']
        self.feature_importance = state['feature_importance']
        self.metrics = state['metrics']
        self.reference_predictions = state.get('reference_predictions')
        self.training_history = state.get('training_history', [])
        logger.info(f"Model loaded from {filepath}")
    
    def get_summary(self) -> str:
        """Get model summary"""
        return f"""
{'='*50}
DISCOUNT PREDICTION MODEL SUMMARY
{'='*50}
Model Type: {self.model_type}

Training Metrics:
  RMSE: {self.metrics.get('train_rmse', 'N/A'):.4f}
  MAE:  {self.metrics.get('train_mae', 'N/A'):.4f}
  R²:   {self.metrics.get('train_r2', 'N/A'):.4f}

Validation Metrics:
  RMSE: {self.metrics.get('val_rmse', 'N/A'):.4f}
  MAE:  {self.metrics.get('val_mae', 'N/A'):.4f}
  R²:   {self.metrics.get('val_r2', 'N/A'):.4f}

Training Time: {self.metrics.get('training_time', 'N/A'):.2f}s
{'='*50}
"""
