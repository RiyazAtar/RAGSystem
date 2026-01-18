"""
Data Processing Module for Marketing AI System
Handles data loading, cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
import re
import os
import logging
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Comprehensive data processing pipeline"""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.target_column: str = "discount_percentage"
        self.fitted: bool = False
        
        # Store statistics for monitoring
        self.data_stats: Dict[str, Any] = {}
        
    def clean_price(self, price_str) -> float:
        """Clean price string: '₹1,099' -> 1099.0"""
        if pd.isna(price_str):
            return 0.0
        cleaned = re.sub(r'[₹$,\s]', '', str(price_str))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def clean_percentage(self, pct_str) -> float:
        """Clean percentage string: '64%' -> 64.0"""
        if pd.isna(pct_str):
            return 0.0
        cleaned = re.sub(r'[%\s]', '', str(pct_str))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def clean_rating(self, rating_str) -> float:
        """Clean rating string: '4.2' -> 4.2"""
        if pd.isna(rating_str):
            return 0.0
        try:
            return float(str(rating_str).strip())
        except ValueError:
            return 0.0
    
    def clean_rating_count(self, count_str) -> int:
        """Clean rating count: '24,269' -> 24269"""
        if pd.isna(count_str):
            return 0
        cleaned = re.sub(r'[,\s]', '', str(count_str))
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    
    def extract_categories(self, category_str: str) -> Tuple[str, str, str]:
        """Extract category hierarchy"""
        if pd.isna(category_str):
            return "Unknown", "Unknown", "Unknown"
        
        parts = str(category_str).split('|')
        main_cat = parts[0].strip() if len(parts) > 0 else "Unknown"
        sub_cat = parts[1].strip() if len(parts) > 1 else "Unknown"
        sub_sub_cat = parts[2].strip() if len(parts) > 2 else "Unknown"
        
        return main_cat, sub_cat, sub_sub_cat
    
    def extract_brand(self, product_name: str) -> str:
        """Extract brand dynamically from product name"""
        if pd.isna(product_name):
            return "Unknown"
        
        name = str(product_name).strip()
        if not name:
            return "Unknown"
        
        words = name.split()
        if not words:
            return "Unknown"
        
        brand = words[0].replace('®', '').replace('™', '').replace('©', '').strip()
        
        # Handle multi-word brands
        if len(words) > 1:
            second_word = words[1].lower()
            if brand.lower() == 'amazon' and second_word == 'basics':
                brand = 'AmazonBasics'
            elif brand.lower() == 'fire' and second_word == 'boltt':
                brand = 'Fire-Boltt'
        
        # Normalize common variations
        normalizations = {
            'mi': 'MI', 'hp': 'HP', 'lg': 'LG', 'jbl': 'JBL',
            'ptron': 'pTron', 'boat': 'boAt', 'iqoo': 'iQOO'
        }
        if brand.lower() in normalizations:
            brand = normalizations[brand.lower()]
        
        return brand if brand else "Unknown"
    
    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Load and clean raw data"""
        logger.info(f"Loading data from {filepath}")
        
        if isinstance(filepath, pd.DataFrame):
            df = filepath.copy()
        else:
            df = pd.read_csv(filepath)
        
        logger.info(f"Raw data shape: {df.shape}")
        
        # Clean numeric columns
        df['actual_price_clean'] = df['actual_price'].apply(self.clean_price)
        df['discounted_price_clean'] = df['discounted_price'].apply(self.clean_price)
        df['discount_percentage'] = df['discount_percentage'].apply(self.clean_percentage)
        df['rating_clean'] = df['rating'].apply(self.clean_rating)
        df['rating_count_clean'] = df['rating_count'].apply(self.clean_rating_count)
        
        # Extract categories
        category_data = df['category'].apply(self.extract_categories)
        df['main_category'] = category_data.apply(lambda x: x[0])
        df['sub_category'] = category_data.apply(lambda x: x[1])
        df['sub_sub_category'] = category_data.apply(lambda x: x[2])
        
        # Extract brand
        df['brand'] = df['product_name'].apply(self.extract_brand)
        
        # Text length features
        df['about_product_length'] = df['about_product'].fillna('').str.len()
        df['review_content_length'] = df['review_content'].fillna('').str.len()
        df['product_name_length'] = df['product_name'].fillna('').str.len()
        
        # Remove duplicates by product_name (get 1337 unique products)
        df = df.drop_duplicates(subset=['product_name'], keep='first')
        df = df.dropna(subset=['product_id'])
        
        # Filter invalid data
        df = df[df['actual_price_clean'] > 0]
        df = df[df['discount_percentage'] >= 0]
        df = df[df['discount_percentage'] <= 100]
        
        logger.info(f"Cleaned data shape: {df.shape}")
        
        # Store statistics
        self.data_stats = {
            'total_products': len(df),
            'unique_brands': df['brand'].nunique(),
            'unique_categories': df['main_category'].nunique(),
            'price_mean': df['actual_price_clean'].mean(),
            'price_std': df['actual_price_clean'].std(),
            'discount_mean': df['discount_percentage'].mean(),
            'discount_std': df['discount_percentage'].std()
        }
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for modeling"""
        logger.info("Engineering features...")
        
        # Price features
        df['price_log'] = np.log1p(df['actual_price_clean'])
        df['price_bucket'] = pd.cut(
            df['actual_price_clean'],
            bins=[0, 500, 2000, 10000, 50000, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'premium']
        ).astype(str)
        
        # Rating features
        df['rating_bucket'] = pd.cut(
            df['rating_clean'],
            bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
            labels=['poor', 'average', 'good', 'very_good', 'excellent']
        ).astype(str)
        
        df['rating_count_log'] = np.log1p(df['rating_count_clean'])
        df['high_engagement'] = (df['rating_count_clean'] > df['rating_count_clean'].median()).astype(int)
        
        # Popularity features
        cat_counts = df['main_category'].value_counts()
        df['category_popularity'] = df['main_category'].map(cat_counts)
        
        brand_counts = df['brand'].value_counts()
        df['brand_popularity'] = df['brand'].map(brand_counts)
        
        # Interaction features
        df['price_rating_interaction'] = df['price_log'] * df['rating_clean']
        df['engagement_score'] = df['rating_clean'] * np.log1p(df['rating_count_clean'])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model"""
        
        self.categorical_columns = ['main_category', 'sub_category', 'brand', 'price_bucket', 'rating_bucket']
        self.numerical_columns = [
            'actual_price_clean', 'rating_clean', 'rating_count_log',
            'about_product_length', 'review_content_length', 'price_log',
            'category_popularity', 'brand_popularity', 'high_engagement',
            'price_rating_interaction', 'engagement_score'
        ]
        
        # Encode categorical variables
        encoded_features = []
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    # During training: drop rows with NaN in categorical columns
                    # This prevents "Unknown" from becoming a label
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].value_counts().index[0])
                    self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(df[col])
                else:
                    # During prediction: map unknown labels to most common label from training
                    known_labels = set(self.label_encoders[col].classes_)
                    most_common = self.label_encoders[col].classes_[0]  # First class (most common during training)
                    df[col] = df[col].fillna(most_common)
                    df[col] = df[col].apply(lambda x: x if x in known_labels else most_common)
                    encoded = self.label_encoders[col].transform(df[col])
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Get numerical features
        numerical_data = df[self.numerical_columns].fillna(0).values
        
        # Combine features
        if encoded_features:
            X = np.hstack(encoded_features + [numerical_data])
        else:
            X = numerical_data
        
        # Scale features
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            X = self.scaler.transform(X)

        self.feature_columns = self.categorical_columns + self.numerical_columns

        # Only extract y if target column exists (for training)
        if self.target_column in df.columns:
            y = df[self.target_column].values
        else:
            y = None

        return X, y
    
    def prepare_for_rag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare rich text documents for RAG indexing"""
        logger.info("Preparing data for RAG...")
        
        def create_document(row):
            return f"""
Product: {row['product_name']}
Product ID: {row['product_id']}
Brand: {row['brand']}
Category: {row['main_category']} > {row['sub_category']}
Original Price: {row['actual_price']}
Sale Price: {row['discounted_price']}
Discount: {row['discount_percentage']:.0f}% off
Rating: {row['rating_clean']}/5 ({row['rating_count_clean']:,} reviews)
Description: {str(row['about_product'])[:500] if pd.notna(row['about_product']) else 'N/A'}
Review: {str(row['review_content'])[:300] if pd.notna(row['review_content']) else 'N/A'}
            """.strip()
        
        df['rag_document'] = df.apply(create_document, axis=1)
        return df
    
    
    def save(self, filepath: str):
        """Save processor state"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'data_stats': self.data_stats,
            'fitted': self.fitted
        }
        joblib.dump(state, filepath)
        logger.info(f"Processor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load processor state"""
        state = joblib.load(filepath)
        self.label_encoders = state['label_encoders']
        self.scaler = state['scaler']
        self.feature_columns = state['feature_columns']
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        self.data_stats = state['data_stats']
        self.fitted = state['fitted']
        logger.info(f"Processor loaded from {filepath}")
