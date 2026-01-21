# ML Models module
import logging

logger = logging.getLogger(__name__)

from app.ml.models.base import BaseModel

# Optional imports for heavy ML dependencies
ProphetModel = None
LSTMModel = None
XGBoostModel = None

try:
    from app.ml.models.prophet_model import ProphetModel
except ImportError as e:
    logger.warning(f"ProphetModel not available: {e}")

try:
    from app.ml.models.lstm_model import LSTMModel
except ImportError as e:
    logger.warning(f"LSTMModel not available: {e}")

try:
    from app.ml.models.xgboost_model import XGBoostModel
except ImportError as e:
    logger.warning(f"XGBoostModel not available: {e}")

from app.ml.models.ensemble import EnsemblePredictor

__all__ = [
    "BaseModel",
    "ProphetModel",
    "LSTMModel",
    "XGBoostModel",
    "EnsemblePredictor",
]
