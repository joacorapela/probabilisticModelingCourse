import sys
import os.path
import random
import argparse
import configparser
import numpy as np
import pandas as pd

sys.path.append("../src")
import inference


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simRes_number", type=int,
                        help="simulation result number")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    parser.add_argument("--results_filenames_pattern", type=str,
                        default="../../results/{:08d}_smoothed.{:s}",
                        help="results filename pattern")
    args = parser.parse_args()

    simRes_number = args.simRes_number
    simRes_filename_pattern = args.simRes_filename_pattern
    results_filenames_pattern = args.results_filenames_pattern

    simRes_filename = simRes_filename_pattern.format(simRes_number, "npz")
    simRes = np.load(simRes_filename)

    simRes_metadata_filename = simRes_filename_pattern.format(
        simRes_number, "ini")
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simRes_metadata_filename)
    sim_params_filename = \
        simResConfig["simulation_params"]["sim_params_filename"]
    sim_params = np.load(sim_params_filename)

    filterRes = inference.filterLDS(
        y=simRes["y"], A=sim_params["A"], Q=sim_params["Q"],
        m0=sim_params["m0"], V0=sim_params["V0"], C=sim_params["C"],
        R=sim_params["R"])
    smoothRes = inference.smoothLDS(A=sim_params["A"],
                                    xnn=filterRes["xnn"],
                                    Vnn=filterRes["Vnn"],
                                    xnn1=filterRes["xnn1"],
                                    Vnn1=filterRes["Vnn1"],
                                    m0=sim_params["m0"],
                                    V0=sim_params["V0"])
    data = {"fpos1": filterRes["xnn"][0, 0, :],
            "fpos2": filterRes["xnn"][3, 0, :],
            "fvel1": filterRes["xnn"][1, 0, :],
            "fvel2": filterRes["xnn"][4, 0, :],
            "facc1": filterRes["xnn"][2, 0, :],
            "facc2": filterRes["xnn"][5, 0, :],
            "spos1": smoothRes["xnN"][0, 0, :],
            "spos2": smoothRes["xnN"][3, 0, :],
            "svel1": smoothRes["xnN"][1, 0, :],
            "svel2": smoothRes["xnN"][4, 0, :],
            "sacc1": smoothRes["xnN"][2, 0, :],
            "sacc2": smoothRes["xnN"][5, 0, :]}
    df = pd.DataFrame(data=data)

    # save results
    res_prefix_used = True
    while res_prefix_used:
        res_number = random.randint(0, 10**8)
        smoothed_metadata_filename = results_filenames_pattern.format(
            res_number, "ini")
        if not os.path.exists(smoothed_metadata_filename):
            res_prefix_used = False
    smoothed_data_filename = results_filenames_pattern.format(
        res_number, "csv")

    df.to_csv(smoothed_data_filename)
    print("Saved results to {:s}".format(smoothed_data_filename))
    smoothed_metadata = configparser.ConfigParser()
    smoothed_metadata["params"] = {"simRes_number": simRes_number}
    with open(smoothed_metadata_filename, "w") as f:
        smoothed_metadata.write(f)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
