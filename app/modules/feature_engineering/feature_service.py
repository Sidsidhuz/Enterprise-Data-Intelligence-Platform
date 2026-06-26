from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class FeatureService:
    @staticmethod
    def preprocess_dataframe_pandas(
        df: pd.DataFrame, target_col: str, dtypes: Dict[str, str]
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Decomposes datetime columns into numeric components (year, month, day)
        and returns the modified DataFrame and updated dtypes dictionary.
        Also drops columns that are not useful for modeling (e.g. IDs).
        """
        df_out = df.copy()
        new_dtypes = dtypes.copy()

        # Identify datetime columns and extract features
        for col in df.columns:
            if col == target_col:
                continue

            col_type = dtypes.get(col, "categorical")

            # Simple heuristic to drop IDs (high cardinality categorical containing 'id' or 'key' in name)
            if col_type == "categorical" and any(x in col.lower() for x in ["id", "key", "uuid", "index"]):
                if df_out[col].nunique() == len(df_out):
                    df_out = df_out.drop(columns=[col])
                    new_dtypes.pop(col, None)
                    continue

            if col_type == "datetime":
                try:
                    dt_series = pd.to_datetime(df_out[col])
                    df_out[f"{col}_year"] = dt_series.dt.year
                    df_out[f"{col}_month"] = dt_series.dt.month
                    df_out[f"{col}_day"] = dt_series.dt.day
                    df_out[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                    
                    # Drop original datetime column
                    df_out = df_out.drop(columns=[col])
                    new_dtypes.pop(col, None)

                    # Register new columns
                    new_dtypes[f"{col}_year"] = "numeric"
                    new_dtypes[f"{col}_month"] = "numeric"
                    new_dtypes[f"{col}_day"] = "numeric"
                    new_dtypes[f"{col}_dayofweek"] = "numeric"
                except Exception:
                    # Fallback to categorical if parsing fails
                    new_dtypes[col] = "categorical"

        return df_out, new_dtypes

    @staticmethod
    def get_preprocessing_pipeline(
        feature_cols: List[str], dtypes: Dict[str, str], df: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Builds a scikit-learn ColumnTransformer for preprocessing numeric
        and categorical features.
        """
        numeric_features = []
        categorical_features = []

        for col in feature_cols:
            col_type = dtypes.get(col, "categorical")
            if col_type == "numeric" or col_type == "boolean":
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        # Preprocessing for numeric data: impute (backup) and scale
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Preprocessing for categorical data: one-hot for low cardinality, ordinal for high
        # We will separate categoricals into low vs high cardinality
        low_card_features = []
        high_card_features = []
        
        for col in categorical_features:
            cardinality = df[col].nunique()
            if cardinality <= 10:
                low_card_features.append(col)
            else:
                high_card_features.append(col)

        transformers = []

        if numeric_features:
            transformers.append(("num", numeric_transformer, numeric_features))
            
        if low_card_features:
            low_card_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat_low", low_card_transformer, low_card_features))

        if high_card_features:
            high_card_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            transformers.append(("cat_high", high_card_transformer, high_card_features))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor
