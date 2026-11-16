__version__ = "0.1.1"

try:
    from .models import ScCCVGBenModel, ENCODER_REGISTRY
except ImportError:
    pass
