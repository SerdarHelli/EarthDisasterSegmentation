

def instantiate_from_config(config,reload=False):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"],reload)(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)