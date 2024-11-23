from imblearn.under_sampling import (
    ClusterCentroids, 
    NearMiss, 
    TomekLinks, 
    CondensedNearestNeighbour
)
from collections import Counter

def analyze_class_balance(df, target_col):
    """
    Analiza el balance de clases
    """
    class_counts = df[target_col].value_counts()
    class_proportions = df[target_col].value_counts(normalize=True)
    
    return {
        'counts': class_counts,
        'proportions': class_proportions
    }

def apply_balancing_techniques(X, y):
    """
    Aplica diferentes técnicas de balanceo
    """
    results = {}
    
    # Cluster Centroids
    cc = ClusterCentroids(random_state=42)
    X_cc, y_cc = cc.fit_resample(X, y)
    results['ClusterCentroids'] = Counter(y_cc)
    
    # NearMiss
    nm = NearMiss(version=1)
    X_nm, y_nm = nm.fit_resample(X, y)
    results['NearMiss'] = Counter(y_nm)
    
    # Tomek Links
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X, y)
    results['TomekLinks'] = Counter(y_tl)
    
    # Condensed Nearest Neighbours
    cnn = CondensedNearestNeighbour(random_state=42)
    X_cnn, y_cnn = cnn.fit_resample(X, y)
    results['CNN'] = Counter(y_cnn)
    
    return results