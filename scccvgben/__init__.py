__version__ = "0.1.0"

try:
    from .models import CCVGAE, ENCODER_REGISTRY
except ImportError:
    pass
