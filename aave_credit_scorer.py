# File: c:\Users\Lenovo Idepad Gaming\Desktop\aave_credit_scorer\aave_credit_scorer.py
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class AaveCreditScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.features = None
        
    def load_data(self, filepath):
        """Load data with comprehensive structure validation"""
        try:
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(raw_data, dict):
                if 'transactions' in raw_data:
                    df = pd.DataFrame(raw_data['transactions'])
                else:
                    df = pd.DataFrame([raw_data])  # Single transaction
            elif isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
            else:
                raise ValueError("Unsupported JSON structure")
                
            # Validate required columns
            required_cols = ['_id', 'timestamp', 'actionData']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")

            # Ensure actionData is present and is a dictionary or can be converted
            if 'actionData' in df.columns:
                # Attempt to convert stringified JSON in actionData to dicts
                df['actionData'] = df['actionData'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                # Check if 'type' and 'amount' keys exist within actionData dictionaries
                if not all(df['actionData'].apply(lambda x: isinstance(x, dict) and 'type' in x and 'amount' in x)): # type: ignore
                    raise KeyError("Missing 'type' or 'amount' within 'actionData' dictionaries or 'actionData' is not a dictionary.")
            else:
                raise KeyError("Missing required column: 'actionData'")
                
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(f"Available columns: {list(df.columns) if 'df' in locals() else 'N/A'}")
            return None
    
    def preprocess_data(self, df):
        """Robust data cleaning with flexible field handling"""
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Ensure _id is hashable (convert to string)
        df['_id'] = df['_id'].astype(str)
        
        # Handle reserve/asset information
        if 'reserve' in df.columns:
            if isinstance(df['reserve'].iloc[0], dict):
                df['asset'] = df['reserve'].apply(lambda x: x.get('symbol', 'UNKNOWN'))
            else:
                df['asset'] = df['reserve'].astype(str)
        elif 'asset' in df.columns:
            df['asset'] = df['asset'].astype(str)
        else:
            df['asset'] = 'UNKNOWN'
        
        # Clean amount
        df['actionData'] = df['actionData'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
        df['amount'] = df['actionData'].apply(lambda x: x.get('amount'))
        df['type'] = df['actionData'].apply(lambda x: x.get('type'))
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df[df['amount'] > 0]
        
        # Standardize transaction types
        valid_types = ['deposit', 'borrow', 'repay', 'redeemUnderlying', 'liquidationCall']
        df['type'] = df['type'].where(df['type'].isin(valid_types), 'other')
        
        return df
    
    def engineer_features(self, df):
        """Feature engineering with robust calculations"""
        if df.empty:
            raise ValueError("Empty DataFrame after preprocessing")
            
        grouped = df.groupby('_id')
        
        # Transaction counts
        features = pd.DataFrame({
            'total_tx': grouped.size(),
            'deposit_count': grouped['type'].apply(lambda x: (x == 'deposit').sum()),
            'borrow_count': grouped['type'].apply(lambda x: (x == 'borrow').sum()),
            'repay_count': grouped['type'].apply(lambda x: (x == 'repay').sum()),
            'redeem_count': grouped['type'].apply(lambda x: (x == 'redeemUnderlying').sum()),
            'liquidation_count': grouped['type'].apply(lambda x: (x == 'liquidationCall').sum())
        })
        
        # Time features
        features['days_active'] = grouped['timestamp'].apply(
            lambda x: (x.max() - x.min()).days if len(x) > 1 else 1
        )
        features['tx_frequency'] = features['total_tx'] / features['days_active']
        
        # Amount features with safe calculations
        for ttype in ['deposit', 'borrow', 'repay', 'redeemUnderlying']:
            mask = df['type'] == ttype
            if mask.any():
                txn_group = df[mask].groupby('_id')['amount']
                features[f'{ttype}_avg'] = txn_group.mean()
                features[f'{ttype}_std'] = txn_group.std()
                features[f'{ttype}_total'] = txn_group.sum()
            else:
                features[f'{ttype}_avg'] = 0
                features[f'{ttype}_std'] = 0
                features[f'{ttype}_total'] = 0
        
        # Risk ratios with safe division
        features['borrow_deposit_ratio'] = np.where(
            features['deposit_total'] > 0,
            features['borrow_total'] / features['deposit_total'],
            0
        )
        features['repay_borrow_ratio'] = np.where(
            features['borrow_total'] > 0,
            features['repay_total'] / features['borrow_total'],
            0
        )
        features['liquidation_ratio'] = np.where(
            features['borrow_count'] > 0,
            features['liquidation_count'] / features['borrow_count'],
            0
        )
        
        # Fill NA and infinite values
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        
        self.features = features
        return features
    
    def train_model(self, features):
        """Model training with robust scoring"""
        scaled = self.scaler.fit_transform(features)
        
        # Anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled)
        distances = kmeans.transform(scaled)
        
        # Scoring logic
        min_distances = distances.min(axis=1)
        scores = 1000 * (1 - (min_distances - min_distances.min()) / 
                        (min_distances.max() - min_distances.min() + 1e-6))
        
        # Penalize anomalies
        scores[anomaly_scores == -1] *= 0.7
        
        return np.clip(scores, 0, 1000).astype(int)
    
    def process_file(self, input_file, output_dir='results'):
        """Complete processing pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = self.load_data(input_file)
        if df is None:
            return None, None
            
        # Preprocess
        df = self.preprocess_data(df)
        
        # Feature engineering
        try:
            features = self.engineer_features(df)
        except Exception as e:
            print(f"Feature engineering failed: {str(e)}")
            return None, None
        
        # Generate scores
        scores = self.train_model(features)
        
        # Save results
        results = pd.DataFrame({
            'wallet': features.index,
            'credit_score': scores
        })
        results.to_csv(f'{output_dir}/scores.csv', index=False)
        
        # Generate analysis
        analysis = self._generate_analysis(features, scores, output_dir)
        return results, analysis
    
    def _generate_analysis(self, features, scores, output_dir):
        """Create comprehensive analysis"""
        analysis = {
            'score_stats': {
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'score_distribution': pd.cut(scores, bins=range(0, 1100, 100))
                              .value_counts().sort_index().to_dict(),
            'feature_correlations': features.corr().to_dict()
        }
        
        # Create visualizations
        plt.figure(figsize=(12, 6))
        pd.Series(scores).plot(kind='hist', bins=20)
        plt.title('Credit Score Distribution')
        plt.savefig(f'{output_dir}/score_dist.png')
        plt.close()
        
        return analysis

if __name__ == '__main__':
    scorer = AaveCreditScorer()
    
    # Example usage
    input_path = 'user-wallet-transactions.json'  # Update with your path
    results, analysis = scorer.process_file(input_path)
    
    if results is not None:
        print(f"Successfully processed {len(results)} wallets")
        print(f"Average score: {analysis['score_stats']['mean']:.1f}")
    else:
        print("Processing failed - check error messages")
