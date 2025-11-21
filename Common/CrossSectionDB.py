import math
import yaml
import os

class CrossSectionDB:

    @staticmethod
    def Load(base_path, config_entry):
        cfg_paths = []
        if type(config_entry) == str:
            config_entry = [config_entry]
        for path in config_entry:
            full_path = path if path.startswith("/") else os.path.join(base_path, path)
            cfg_paths.append(full_path)
        return CrossSectionDB(cfg_paths)

    def __init__(self, cfg_paths):
        self.entries = {}
        self.values = {}
        for cfg_path in cfg_paths:
            old_keys = set(self.entries.keys())
            with open(cfg_path, "r") as file:
                cfg = yaml.safe_load(file)
            for key, entry in cfg.items():
                if key in old_keys:
                    continue
                self.addEntry(key, entry)


    def addEntry(self, entry_name, entry):
        if entry_name in self.entries:
            raise RuntimeError(f"CrossSectionDB: duplicate entry '{entry_name}'")
        if "crossSec" not in entry and "BR" not in entry:
            raise RuntimeError(f"CrossSectionDB: missing 'crossSec' or 'BR' for entry '{entry_name}'")
        if "crossSec" in entry and "BR" in entry:
            raise RuntimeError(f"CrossSectionDB: both 'crossSec' and 'BR' defined for entry '{entry_name}'")
        entry_type = 'BR' if "BR" in entry else 'crossSec'
        value = self.evaluateExpression(entry[entry_type], entry_name=entry_name)
        entry[f"{entry_type}_orig"] = entry[entry_type]
        entry["type"] = entry_type
        entry[entry_type] = value
        self.entries[entry_name] = entry
        self.values[entry_name] = value

    def getValue(self, entry_name):
        if entry_name not in self.values:
            raise RuntimeError(f"CrossSectionDB: unknown entry '{entry_name}'")
        return self.values[entry_name]

    def getEntry(self, entry_name):
        if entry_name not in self.entries:
            raise RuntimeError(f"CrossSectionDB: unknown entry '{entry_name}'")
        return self.entries[entry_name]

    def evaluateExpression(self, expr, entry_name=None):
        try:
            if type(expr) == int:
                result = float(expr)
            elif type(expr) == float:
                result = expr
            elif type(expr) == str:
                result = eval(expr, {}, self.values)
            else:
                raise RuntimeError(f"expression has invalid type {type(expr)}")
            if type(result) != float:
                raise RuntimeError(f"expression did not evaluate to a float")
            if math.isnan(result) or math.isinf(result):
                raise RuntimeError(f"expression evaluated to NaN or Inf")
            if result < 0:
                raise RuntimeError(f"expression evaluated to a negative value {result}")
            return result
        except Exception as e:
            msg = f"'{expr}'"
            if entry_name is not None:
                msg += f" for entry '{entry_name}'"
            raise RuntimeError(f"CrossSectionDB: error evaluating expression {msg}: {e}")
