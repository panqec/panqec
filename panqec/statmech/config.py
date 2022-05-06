from .loop2d import LoopModel2D, LoopModel2DIidDisorder
from .rbim2d import RandomBondIsingModel2D, Rbim2DIidDisorder


# Register your spin models here.
SPIN_MODELS = {
    'RandomBondIsingModel2D': RandomBondIsingModel2D,
    'LoopModel2D': LoopModel2D,
}
DISORDER_MODELS = {
    'Rbim2DIidDisorder': Rbim2DIidDisorder,
    'LoopModel2DIidDisorder': LoopModel2DIidDisorder,
}
