
import sys
import types
import torch

# 先导入你项目里真正的类/函数定义
from src.models import PINNInverseMLP
from src.data_utils import PINNDataset, variable_length_collate


def _register_symbol_in_main(name: str, obj):
    """
    将 obj 注册到 __main__ 模块下，名称为 name。
    这样当 bundle 在保存时引用了 __main__.name，就能被找到。
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")
        sys.modules["__main__"] = main_mod
    if not hasattr(main_mod, name):
        setattr(main_mod, name, obj)


def _prepare_pickle_symbols():

    _register_symbol_in_main("PINNInverseMLP", PINNInverseMLP)
    _register_symbol_in_main("PINNDataset", PINNDataset)
    _register_symbol_in_main("variable_length_collate", variable_length_collate)


def load_model_bundle(model_path='./model/wtlt_big_pinn.pt'):

    _prepare_pickle_symbols()
    bundle = torch.load(model_path, map_location="cpu")
    model = bundle["model"]
    train_loader = bundle.get("train_loader")
    test_loader = bundle.get("test_loader")
    loss_history = bundle.get("loss_history", {})
    scalers = bundle.get("scalers", {})
    constants = bundle.get("constants", {})
    return model, train_loader, test_loader, loss_history, scalers, constants
