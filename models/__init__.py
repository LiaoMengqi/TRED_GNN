from .TRED_GNN2 import TRED_GNN2
from .TRED_GNN import TRED_GNN
from .TRED_GNN3 import TRED_GNN3
from .TRED_GNN4 import TRED_GNN4
from .Copymode import Copymode

model_dict = {
    "TRED_GNN2": TRED_GNN2,
    "TRED_GNN": TRED_GNN,
    "TRED_GNN3": TRED_GNN3,
    "TRED_GNN4": TRED_GNN4,
    "Copymode": Copymode
}
