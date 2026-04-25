__version__ = "0.1.0"

try:
    from .models import ScCCVGBenModel, ENCODER_REGISTRY
except ImportError:
    pass
