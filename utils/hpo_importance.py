import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

current_confs = [{'config_id': '1', 'fidelity_id': '1', 'loss': 0.19364073629451783, 'test_roc_auc': 0.8588478467960694, 'conf': {'batch_size': 128, 'lr': 0.00045092595973983407}}, {'config_id': '10', 'fidelity_id': '1', 'loss': 0.3396419367747099, 'test_roc_auc': 0.547611774640498, 'conf': {'batch_size': 512, 'lr': 3.515948264976032e-05}}, {'config_id': '11', 'fidelity_id': '1', 'loss': 0.19843897559023604, 'test_roc_auc': 0.8567133179736883, 'conf': {'batch_size': 256, 'lr': 0.0021516126580536366}}, {'config_id': '12', 'fidelity_id': '1', 'loss': 0.28284905962384954, 'test_roc_auc': 0.7221284503617977, 'conf': {'batch_size': 512, 'lr': 0.05524750053882599}}, {'config_id': '13', 'fidelity_id': '1', 'loss': 0.3214419367747099, 'test_roc_auc': 0.6075501852241465, 'conf': {'batch_size': 256, 'lr': 4.924634413328022e-05}}, {'config_id': '14', 'fidelity_id': '1', 'loss': 0.19403969587835135, 'test_roc_auc': 0.857249773658971, 'conf': {'batch_size': 128, 'lr': 0.00055680115474388}}, {'config_id': '15', 'fidelity_id': '1', 'loss': 0.22364505802320922, 'test_roc_auc': 0.8190383892000572, 'conf': {'batch_size': 256, 'lr': 0.016404995694756508}}, {'config_id': '16', 'fidelity_id': '1', 'loss': 0.3322419367747098, 'test_roc_auc': 0.5665672537049105, 'conf': {'batch_size': 256, 'lr': 2.747994221863337e-05}}, {'config_id': '17', 'fidelity_id': '1', 'loss': 0.2220445778311324, 'test_roc_auc': 0.825794693589849, 'conf': {'batch_size': 256, 'lr': 0.012367508374154568}}, {'config_id': '18', 'fidelity_id': '1', 'loss': 0.29844193677470987, 'test_roc_auc': 0.6683942666245211, 'conf': {'batch_size': 128, 'lr': 5.0325561460340396e-05}}, {'config_id': '19', 'fidelity_id': '1', 'loss': 0.1866377751100441, 'test_roc_auc': 0.8594933236029448, 'conf': {'batch_size': 256, 'lr': 0.0002385831467108801}}, {'config_id': '2', 'fidelity_id': '1', 'loss': 0.3398506602641056, 'test_roc_auc': 0.6543937750103397, 'conf': {'batch_size': 256, 'lr': 0.06025729700922966}}, {'config_id': '20', 'fidelity_id': '1', 'loss': 0.2728499399759905, 'test_roc_auc': 0.747544571355976, 'conf': {'batch_size': 256, 'lr': 0.02889871411025524}}, {'config_id': '21', 'fidelity_id': '1', 'loss': 0.3316419367747099, 'test_roc_auc': 0.5668863469100471, 'conf': {'batch_size': 512, 'lr': 5.198559301788919e-05}}, {'config_id': '22', 'fidelity_id': '1', 'loss': 0.19583969587835137, 'test_roc_auc': 0.855309549918827, 'conf': {'batch_size': 128, 'lr': 0.0011583846062421799}}, {'config_id': '23', 'fidelity_id': '1', 'loss': 0.2922419367747099, 'test_roc_auc': 0.6888138314250001, 'conf': {'batch_size': 512, 'lr': 0.00019230223551858217}}, {'config_id': '24', 'fidelity_id': '1', 'loss': 0.30084193677470994, 'test_roc_auc': 0.6642730548561399, 'conf': {'batch_size': 128, 'lr': 4.855730367125943e-05}}, {'config_id': '3', 'fidelity_id': '1', 'loss': 0.3364419367747098, 'test_roc_auc': 0.5537057306290407, 'conf': {'batch_size': 512, 'lr': 4.020370033686049e-05}}, {'config_id': '4', 'fidelity_id': '1', 'loss': 0.21844521808723494, 'test_roc_auc': 0.828348314751052, 'conf': {'batch_size': 256, 'lr': 0.011100190691649914}}, {'config_id': '5', 'fidelity_id': '1', 'loss': 0.32064193677470987, 'test_roc_auc': 0.6206669458978151, 'conf': {'batch_size': 128, 'lr': 3.127250602119602e-05}}, {'config_id': '6', 'fidelity_id': '1', 'loss': 0.19024041616646659, 'test_roc_auc': 0.8617849137820144, 'conf': {'batch_size': 256, 'lr': 0.000403089914470911}}, {'config_id': '7', 'fidelity_id': '1', 'loss': 0.20464281712685084, 'test_roc_auc': 0.8423374037812732, 'conf': {'batch_size': 512, 'lr': 0.010275288484990597}}, {'config_id': '8', 'fidelity_id': '1', 'loss': 0.3042419367747099, 'test_roc_auc': 0.6552451494455115, 'conf': {'batch_size': 128, 'lr': 4.4588356104213744e-05}}, {'config_id': '9', 'fidelity_id': '1', 'loss': 0.2722511404561825, 'test_roc_auc': 0.7531377930680497, 'conf': {'batch_size': 256, 'lr': 0.03369069844484329}}]
def compute_shap_importance(current_confs):
    # Build DataFrame from configs
    df = pd.DataFrame([
        {
            "lr": c["conf"]["lr"],
            "batch_size": c["conf"]["batch_size"],
            "loss": c["loss"]
        }
        for c in current_confs
    ])
    
    X = df[["lr", "batch_size"]]
    y = df["loss"]

    # Fit a model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Explain with SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values = feature importance
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    print("\nHyperparameter Importance (SHAP):")
    print(importance_df)

    # Optional plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)  # save as PNG
    plt.close()

compute_shap_importance(current_confs)