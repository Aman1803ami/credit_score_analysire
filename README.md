# Aave V2 Wallet Credit Scoring

## Overview
This solution assigns credit scores (0-1000) to Ethereum wallets based on their transaction history with Aave V2 protocol. Higher scores indicate more reliable and responsible usage patterns.

## Methodology

### Feature Engineering
1. **Transaction Counts**: Deposit, borrow, repay, redeem, liquidation counts
2. **Time Patterns**: Activity duration, transaction frequency, regularity
3. **Amount Statistics**: Average, total, and volatility of transaction amounts
4. **Risk Ratios**: Borrow/deposit ratio, repay/borrow ratio, liquidation frequency

### Modeling Approach
1. **Hybrid Model**:
   - Isolation Forest for anomaly detection
   - K-means clustering for behavior pattern grouping
2. **Scoring Logic**:
   - Scores based on distance to nearest cluster center
   - Anomalous wallets penalized by 30%
   - Normalized to 0-1000 range

### Score Interpretation
- **900-1000**: Exemplary users (consistent deposits, timely repayments)
- **700-900**: Responsible users (moderate borrowing, good repayment)
- **500-700**: Average users (some risk factors present)
- **300-500**: Risky users (high borrowing, late repayments)
- **0-300**: High-risk (frequent liquidations, exploit-like patterns)

## Architecture
1. **Input**: Raw transaction JSON (local file or Google Drive URL)
2. **Processing**:
   - Data cleaning and feature engineering
   - Hybrid model training and scoring
3. **Output**:
   - CSV of wallet scores
   - Analysis report with visualizations
   - Score distribution and feature correlations

## Usage
```python
from aave_scorer import AaveCreditScorer

scorer = AaveCreditScorer()
# Use local file
results = scorer.process_file('data/user_transactions.json')
# Or download from Google Drive
results = scorer.process_file('https://drive.google.com/...')
