# Core models with stable dependencies
from . import Transformer, TimesNet
from . import DLinear, Informer
from . import PatchTST, TimesNettcn
# Optional models - gracefully skip if dependencies are missing


try:
    from . import Sundial
except ImportError:
    Sundial = None

try:
    from . import TimeMoE
except ImportError:
    TimeMoE = None

try:
    from . import Chronos
except ImportError:
    Chronos = None

try:
    from . import Moirai
except ImportError:
    Moirai = None

try:
    from . import TiRex
except ImportError:
    TiRex = None

try:
    from . import TimesFM
except ImportError:
    TimesFM = None

try:
    from . import Chronos2
except ImportError:
    Chronos2 = None

__all__ = [
    'Transformer', 'TimesNet', 'TimesNettcn',
    'DLinear', 'Informer', 'PatchTST'
]