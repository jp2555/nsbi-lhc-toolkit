import os, sys, json
import argparse
import logging
import warnings
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
import jax

import nsbi_common_utils

jax.config.update("jax_enable_x64", True)

print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Style
hep.style.use(hep.style.ATLAS)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and parameter fitting.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.pipeline.yaml",
        help="Path to the main pipeline configuration file."
    )
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_nll_plot(scan_data, output_dir, parameter_label):
    """Plots the NLL scans and saves the figure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    
    for item in scan_data:
        plt.plot(
            item['points'], 
            item['nll'], 
            label=item['label'], 
            linestyle=item['style'], 
            color=item['color'],
            linewidth=2
        )
    
    plt.xlabel(parameter_label, fontsize=20)
    plt.ylabel(r"$-2\Delta \ln L$", fontsize=20)
    plt.legend(fontsize=14, frameon=False)
    plt.ylim(0, 8)
    plt.xlim(0, 3)
    
    plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(4, color='gray', linestyle=':', alpha=0.5)
    
    hep.atlas.label(data=False, label="Internal", loc=0)
    
    output_path = os.path.join(output_dir, "nll_scan_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    args = parse_args()
    config_workflow = load_config(args.config)["parameter_fitting"]
    
    plots_dir = config_workflow["output"]["plots_dir"]
    
    logger.info("Starting Inference Pipeline")
    logger.info(f"Configurations: Hist={config_workflow['configs']['histogram']}, NSBI={config_workflow['configs']['nsbi']}")

    measurement = config_workflow["measurement"]
    scan_param = config_workflow["scan"]["parameter"]
    scan_range = tuple(config_workflow["scan"]["range"])
    scan_steps = config_workflow["scan"]["steps"]

    try:
        
        logger.info("Building Workspaces")

        hist_config_path = config_workflow["configs"]["histogram"]
        workspace_histogram = nsbi_common_utils.workspace_builder.WorkspaceBuilder(config_path=hist_config_path).build()

        nsbi_config_path = config_workflow["configs"]["nsbi"]
        workspace_nsbi = nsbi_common_utils.workspace_builder.WorkspaceBuilder(config_path=nsbi_config_path).build()
        
        logger.info("Initializing Models")

        model_hist = nsbi_common_utils.models.sbi_parametric_model(workspace=workspace_histogram, 
                                                   measurement_to_fit=measurement)
        
        model_nsbi = nsbi_common_utils.models.sbi_parametric_model(workspace=workspace_nsbi, 
                                                   measurement_to_fit=measurement)
        
        list_params, init_values = model_hist.get_model_parameters()
        num_unconstrained = model_hist.num_unconstrained_param
        
        inference_histogram = nsbi_common_utils.inference.inference(
            model_nll=model_hist.model,
            initial_values=init_values,
            list_parameters=list_params,
            num_unconstrained_params=num_unconstrained,
            model_grad=model_hist.model_grad
        )

        inference_nsbi = nsbi_common_utils.inference.inference(
            model_nll=model_nsbi.model,
            initial_values=init_values,
            list_parameters=list_params,
            num_unconstrained_params=num_unconstrained,
            model_grad=model_nsbi.model_grad
        )

        logger.info("\nPerforming Fits (Tables logged to file)")

        # freeze_params = ["JES", "TES"]
        freeze_params = []
        if len(freeze_params)>0:
            print(f"Freezing params {freeze_params} to nominal values, they will not be floated for fits.")
        
        print("\n" + "="*40)
        print(" NSBI FIT RESULTS ")
        print("="*40 + "\n")
        inference_nsbi.perform_fit(freeze_params=freeze_params)
        
        print("\n" + "="*40)
        print(" HISTOGRAM FIT RESULTS ")
        print("="*40 + "\n")
        inference_histogram.perform_fit(freeze_params=freeze_params)

        logger.info(f"\nRunning Profile Scans for {scan_param}")

        logger.info("Scanning Histogram Model")
        pts_hist, nll_hist, pts_stat_hist, nll_stat_hist = inference_histogram.perform_profile_scan(
            parameter_name=scan_param,
            freeze_params=freeze_params,
            bound_range=scan_range,
            fit_strategy=0,
            doStatOnly=True,
            size=scan_steps
        )

        logger.info("Scanning NSBI Model")
        pts_nsbi, nll_nsbi, pts_stat_nsbi, nll_stat_nsbi = inference_nsbi.perform_profile_scan(
            parameter_name=scan_param,
            freeze_params=freeze_params,
            bound_range=scan_range,
            fit_strategy=0,
            doStatOnly=True,
            size=scan_steps
        )

        
        logger.info(f"\nProfiled CI for {scan_param} (level=1.0 ~ 68%%; t_mu convention)")
        from ci import ci_from_scan
        _fmt = lambda v: "open" if v is None else f"{v:.3f}"
        ci_summary = {}
        for name, pts, nll, pts_s, nll_s in [
            ("nsbi", pts_nsbi, nll_nsbi, pts_stat_nsbi, nll_stat_nsbi),
            ("histogram", pts_hist, nll_hist, pts_stat_hist, nll_stat_hist),
        ]:
            lo, hi, half = ci_from_scan(pts, nll, level=1.0)
            lo_s, hi_s, half_s = ci_from_scan(pts_s, nll_s, level=1.0)
            infl = (half / half_s) if (half and half_s) else float("nan")
            ci_summary[name] = {"stat_syst": [lo, hi, half],
                                "stat_only": [lo_s, hi_s, half_s], "syst_inflation": infl}
            logger.info("  %-9s stat+syst=[%s,%s] (+/-%s) | stat-only +/-%s | syst inflation %s",
                        name, _fmt(lo), _fmt(hi), _fmt(half), _fmt(half_s),
                        ("x%.2f" % infl) if infl == infl else "n/a")
        with open(os.path.join(plots_dir, "mu_ci.json"), "w") as fh:
            json.dump(ci_summary, fh, indent=2)
        logger.info(f"CI summary -> {os.path.join(plots_dir, 'mu_ci.json')}")

        logger.info("\nGenerating Plots")

        plot_data = [
            {
                'points': pts_hist, 'nll': nll_hist, 
                'label': "Histogram Stat+Syst", 'style': "-", 'color': "blue"
            },
            {
                'points': pts_stat_hist, 'nll': nll_stat_hist, 
                'label': "Histogram Stat Only", 'style': "--", 'color': "blue"
            },
            {
                'points': pts_nsbi, 'nll': nll_nsbi, 
                'label': "NSBI Stat+Syst", 'style': "-", 'color': "black"
            },
            {
                'points': pts_stat_nsbi, 'nll': nll_stat_nsbi, 
                'label': "NSBI Stat Only", 'style': "--", 'color': "black"
            }
        ]
        
        parameter_label_latex = r'$\mu_{h\tau\tau}$' 
        plot_path = save_nll_plot(plot_data, plots_dir, parameter_label_latex)
        
        logger.info(f"Plot saved to: {plot_path}")
        logger.info("Inference workflow completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
