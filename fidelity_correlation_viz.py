from pathlib import Path
import yaml
import copy
from hydra import compose, initialize
from omegaconf import OmegaConf
from run import load_dataset
import sys
from hpo import neps_training_wrapper
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def extract_config_suffix(path: Path) -> int:
    if path.name.startswith("config_"):
        return path.name[7:]
    return -1

class CorrelationEvaluation:
    def __init__(self, result_dir: str):
        self.current_confs = []
        self.next_confs = []
        # self.get_configs(result_dir=result_dir)
    
    def get_configs(self, result_dir: str) -> list[dict]:
        result_dir = Path(result_dir)
        configs_path = result_dir / "configs"
        if not configs_path.is_dir():
            raise FileNotFoundError(f"Expected 'configs' inside {result_dir}")
        result = {}
        for cfg_dir in sorted(configs_path.glob("config_*"), key=extract_config_suffix):
            config_id = extract_config_suffix(cfg_dir)
            cfg_path = cfg_dir / "config.yaml"
            report_path = cfg_dir / "report.yaml"

            with cfg_path.open("r", encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh)

            with report_path.open("r", encoding="utf-8") as fh:
                report = yaml.safe_load(fh)
                if report["reported_as"] != "success":
                    print(f"skipping config {config_id}, beacause of not successful report")
                    continue
            result = {
                "config_id": config_id,
                "fidelity_id": '1',
                "loss": report.get("objective_to_minimize"),
                "test_roc_auc": report["extra"]["test_mean_roc_auc"],
                "conf": config_data,
            }
            self.current_confs.append(result)
        print(self.current_confs)
        
    def generate_lower_fidelity(self):
        overrides = sys.argv[1:]
        with initialize(config_path="./configs", version_base="1.3"):
            cfg = compose(config_name="train", overrides=overrides)
        args = OmegaConf.to_object(cfg)
        out_dir = Path(args["output_path"])
        out_dir.mkdir(parents=True, exist_ok=True) 
        # load datasets:
        dataset_classes, train_dfs, val_dfs, test_dfs, num_classes = load_dataset(
            dataset=args["dataset"],
            data_path=Path(args["data_path"]).absolute(),
            seed=args["seed"],
            val_size= args["val_size"],
            is_mtl=args["is_mtl"]
        )   
        func = neps_training_wrapper(args, dataset_classes, train_dfs, val_dfs, test_dfs, num_classes, out_dir)
    
        for conf in self.current_confs[2:]:
            lr = conf["conf"]["lr"]
            bs = conf["conf"]["batch_size"]
            res = func("bo-manual", lr, epochs=2, batch_size=bs)
            result = {
                "config_id": conf["config_id"],
                "fidelity_id": '0',
                "loss": res["objective_to_minimize"],
                "test_roc_auc": res["info_dict"]["test_mean_roc_auc"],
                "conf": {"lr": lr, "batch_size": bs, "epochs": 2},
            }
            print(result)
            self.next_confs.append(result)

    def calculate_spearman(self):
        current_dict = {c["config_id"]: c["loss"] for c in self.current_confs}
        next_dict = {n["config_id"]: n["loss"] for n in self.next_confs}
        
        common_ids = set(current_dict.keys()) & set(next_dict.keys())
        
        if len(common_ids) < 3:
            print("Not enough paired configs to calculate correlation.")
            return
        
        losses_current = [current_dict[cid] for cid in common_ids]
        losses_next = [next_dict[cid] for cid in common_ids]
        
        corr, p_val = spearmanr(losses_current, losses_next)
        print(f"Spearman Correlation: {corr:.4f}, P-value: {p_val:.4f}")

        # --- Plot ---
        plt.figure(figsize=(8, 6))
        plt.scatter(losses_next, losses_current, alpha=0.7, edgecolors='k')
        plt.xlabel("Loss at Low Fidelity (e.g., epochs=2)")
        plt.ylabel("Loss at High Fidelity (e.g., epochs=full)")
        plt.title(f"Spearman Correlation = {corr:.2f}, p = {p_val:.2f}")
        
        lims = [min(min(losses_next), min(losses_current)), max(max(losses_next), max(losses_current))]
        plt.plot(lims, lims, 'r--', label='y = x')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("fidelity_spearman_plot.png")
        plt.show()

evaluation = CorrelationEvaluation(result_dir="neps_results/bayesian_optimization2")
evaluation.current_confs = [
    {'config_id': '1', 'fidelity_id': '1', 'loss': 0.17888474056289183, 'test_roc_auc': 0.8913013371130526, 'conf': {'batch_size': 128, 'lr': 0.0016602558316662908}}, 
    {'config_id': '10', 'fidelity_id': '1', 'loss': 0.17750092703748166, 'test_roc_auc': 0.8926600552398952, 'conf': {'batch_size': 512, 'lr': 0.0012238940003729533}}, 
    {'config_id': '11', 'fidelity_id': '1', 'loss': 0.1769167400293451, 'test_roc_auc': 0.891741046948578, 'conf': {'batch_size': 512, 'lr': 0.0011426365907457028}}, 
    {'config_id': '12', 'fidelity_id': '1', 'loss': 0.18321727357609718, 'test_roc_auc': 0.8919240034341739, 'conf': {'batch_size': 512, 'lr': 0.0010063569210268667}}, 
    {'config_id': '13', 'fidelity_id': '1', 'loss': 0.17817838468720826, 'test_roc_auc': 0.8907833077217783, 'conf': {'batch_size': 512, 'lr': 0.0013530557875166545}}, 
    {'config_id': '14', 'fidelity_id': '1', 'loss': 0.21746529945311455, 'test_roc_auc': 0.8488866680536886, 'conf': {'batch_size': 512, 'lr': 0.00010000000000000009}}, 
    {'config_id': '15', 'fidelity_id': '1', 'loss': 0.17985642923836187, 'test_roc_auc': 0.8910422917584843, 'conf': {'batch_size': 128, 'lr': 0.0006555745631320265}}, 
    {'config_id': '16', 'fidelity_id': '1', 'loss': 0.1827792383620115, 'test_roc_auc': 0.8952341159612073, 'conf': {'batch_size': 128, 'lr': 0.0005268382753939339}}, 
    {'config_id': '17', 'fidelity_id': '1', 'loss': 0.17895749633186597, 'test_roc_auc': 0.8913222397770872, 'conf': {'batch_size': 512, 'lr': 0.0011914433003471351}}, 
    {'config_id': '18', 'fidelity_id': '1', 'loss': 0.18389082966519943, 'test_roc_auc': 0.8921839418319967, 'conf': {'batch_size': 512, 'lr': 0.0011493592827589317}}, 
    # {'config_id': '19', 'fidelity_id': '1', 'loss': 0.17642314925970393, 'test_roc_auc': 0.8925675782295688, 'conf': {'batch_size': 128, 'lr': 0.0012032754576359812}}, 
    # {'config_id': '2', 'fidelity_id': '1', 'loss': 0.18282907829798578, 'test_roc_auc': 0.8945045800525073, 'conf': {'batch_size': 128, 'lr': 0.0006527488003484905}}, 
    # {'config_id': '3', 'fidelity_id': '1', 'loss': 0.18263442043484057, 'test_roc_auc': 0.8930965203156135, 'conf': {'batch_size': 512, 'lr': 0.0025998274613680773}}, 
    # {'config_id': '4', 'fidelity_id': '1', 'loss': 0.17449483793517417, 'test_roc_auc': 0.8945334497898467, 'conf': {'batch_size': 256, 'lr': 0.0013948986052189753}}, 
    # {'config_id': '5', 'fidelity_id': '1', 'loss': 0.18358944244364428, 'test_roc_auc': 0.8924760652413659, 'conf': {'batch_size': 256, 'lr': 0.001166258942436676}}, 
    # {'config_id': '6', 'fidelity_id': '1', 'loss': 0.18893260637588372, 'test_roc_auc': 0.8923458755784975, 'conf': {'batch_size': 256, 'lr': 0.0014285977120224943}}, 
    # {'config_id': '7', 'fidelity_id': '1', 'loss': 0.19177405628918232, 'test_roc_auc': 0.8777193210962625, 'conf': {'batch_size': 256, 'lr': 0.008129418678346665}}, 
    # {'config_id': '8', 'fidelity_id': '1', 'loss': 0.1713982059490463, 'test_roc_auc': 0.8898501023932127, 'conf': {'batch_size': 512, 'lr': 0.0013332505309480815}}, 
    # {'config_id': '9', 'fidelity_id': '1', 'loss': 0.17152753101240492, 'test_roc_auc': 0.8924977618564417, 'conf': {'batch_size': 512, 'lr': 0.0012447474732961603}}
]
evaluation.next_confs = [
    {'config_id': '1', 'fidelity_id': '0', 'loss': 0.17217464985994402, 'test_roc_auc': 0.8995732605160586, 'conf': {'lr': 0.0016602558316662908, 'batch_size': 128, 'epochs': 2}},
    {'config_id': '10', 'fidelity_id': '0', 'loss': 0.20783682806455916, 'test_roc_auc': 0.868510265617712, 'conf': {'lr': 0.0012238940003729533, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '11', 'fidelity_id': '0', 'loss': 0.20627603708149922, 'test_roc_auc': 0.8660573827015976, 'conf': {'lr': 0.0011426365907457028, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '12', 'fidelity_id': '0', 'loss': 0.22184857943177272, 'test_roc_auc': 0.8468788329646686, 'conf': {'lr': 0.0010063569210268667, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '13', 'fidelity_id': '0', 'loss': 0.19008089902627712, 'test_roc_auc': 0.8793169628781503, 'conf': {'lr': 0.0013530557875166545, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '14', 'fidelity_id': '0', 'loss': 0.4426668467386955, 'test_roc_auc': 0.5687617535156861, 'conf': {'lr': 0.00010000000000000009, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '15', 'fidelity_id': '0', 'loss': 0.17905873015872997, 'test_roc_auc': 0.8907324590468962, 'conf': {'lr': 0.0006555745631320265, 'batch_size': 128, 'epochs': 2}},
    {'config_id': '16', 'fidelity_id': '0', 'loss': 0.17799142323596107, 'test_roc_auc': 0.8923654544988803, 'conf': {'lr': 0.0005268382753939339, 'batch_size': 128, 'epochs': 2}},
    {'config_id': '17', 'fidelity_id': '0', 'loss': 0.238660010670935, 'test_roc_auc': 0.8645269172837599, 'conf': {'lr': 0.0011914433003471351, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '18', 'fidelity_id': '0', 'loss': 0.2017313258636788, 'test_roc_auc': 0.8677997300044393, 'conf': {'lr': 0.0011493592827589317, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '2', 'fidelity_id': '0', 'loss': 0.1988311670822176, 'test_roc_auc': 0.934695713351391, 'conf': {'lr': 0.0006527488003484905, 'batch_size': 128, 'epochs': 2}},
    {'config_id': '3', 'fidelity_id': '0', 'loss': 0.19485783867393103, 'test_roc_auc': 0.9332788472015129, 'conf': {'lr': 0.0025998274613680773, 'batch_size': 512, 'epochs': 2}},
    {'config_id': '4', 'fidelity_id': '0', 'loss': 0.20344910918213444, 'test_roc_auc': 0.9320170417963671, 'conf': {'lr': 0.0013948986052189753, 'batch_size': 256, 'epochs': 2}},
    {'config_id': '5', 'fidelity_id': '0', 'loss': 0.2038836134453781, 'test_roc_auc': 0.9297483478520169, 'conf': {'lr': 0.001166258942436676, 'batch_size': 256, 'epochs': 2}},
    {'config_id': '6', 'fidelity_id': '0', 'loss': 0.20373584649244314, 'test_roc_auc': 0.9339254229230282, 'conf': {'lr': 0.0014285977120224943, 'batch_size': 256, 'epochs': 2}},

    
]
# evaluation.generate_lower_fidelity()
evaluation.calculate_spearman()
