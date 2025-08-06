from fanova import fANOVA
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
import numpy as np

def compute_fanova(self):
    cs = ConfigurationSpace()
    cs.add([
        UniformFloatHyperparameter("lr", lower=1e-5, upper=1e-1, log=True),
        UniformIntegerHyperparameter("batch_size", lower=16, upper=512, log=True),
        UniformIntegerHyperparameter("epochs", lower=1, upper=10)  # optional if fixed
    ])
    
    # Choose your dataset (e.g., current_confs at highest fidelity)
    data = [c for c in self.current_confs if c["conf"]["epochs"] == max(
        c["conf"]["epochs"] for c in self.current_confs)]
    
    if len(data) < 5:
        print("Too few points for reliable fANOVA.")
        return

    X = []
    y = []
    for c in data:
        X.append([
            c["conf"]["lr"],
            c["conf"]["batch_size"],
            # c["conf"]["epochs"],  # optional
        ])
        y.append(c["loss"])

    X = np.array(X)
    y = np.array(y)

    fanova_obj = fANOVA(X, y, config_space=cs)

    print("\nHyperparameter Importance (fANOVA):")
    for i, hp in enumerate(["lr", "batch_size"]):
        imp = fanova_obj.quantify_importance((i,))
        print(f"{hp}: {imp['individual importance']:.4f}")
