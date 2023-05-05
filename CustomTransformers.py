import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

class ColumnDropperTransformer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        columns_to_drop = ['TransactionId', 'BatchId', 'CurrencyCode', 'CountryCode', 'TransactionStartTime']
        return X.drop(columns_to_drop, axis=1)
    
class IdFormaterTransforer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        id_columns = [col for col in X.columns if 'Id' in col]
        for col in id_columns:
            X[col] = X[col].apply(lambda x: x.split('_')[1])
        # On remplace le type des colonnes par int64
        X[id_columns] = X[id_columns].astype('int64')
        return X
    
class DateFormaterTransformer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['Month'] = X['TransactionStartTime'].dt.month
        # X['Day'] = X['TransactionStartTime'].dt.day
        X['DayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
        X['Hour'] = X['TransactionStartTime'].dt.hour
        # X['Minute'] = X['TransactionStartTime'].dt.minute
        # X['Second'] = X['TransactionStartTime'].dt.
        return X
    
class OneHotEncoderTransformer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        categorical_columns = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        columns_to_encode = [col for col in categorical_columns if X[col].nunique() < 10]

        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[columns_to_encode]))
        # On remets les index
        OH_cols.index = X.index
        # On supprime les colonnes catégorielles
        X = X.drop(columns_to_encode, axis=1)
        # On ajoute les colonnes encodées
        X = pd.concat([X, OH_cols], axis=1)
        # On transforme le nom des colonnes en string
        X.columns = X.columns.map(str)
        return X
    
class OversamplingTransformer (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if 'FraudResult' in X.columns:
            X_fraud = X[X['FraudResult'] == 1]
            X_not_fraud = X[X['FraudResult'] == 0]
            X_fraud_over = resample(X_fraud, replace=True, n_samples=X_not_fraud.shape[0], random_state=42)
            X = pd.concat([X_not_fraud, X_fraud_over])
        return X



    