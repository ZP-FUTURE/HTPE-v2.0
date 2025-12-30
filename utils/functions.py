import importlib
import inspect

def load_model_class(identifier: str, prefix: str = "models."):
    """
    根据字符串动态加载模型类。
    """
    module_path, class_name = identifier.split('@')
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    return cls