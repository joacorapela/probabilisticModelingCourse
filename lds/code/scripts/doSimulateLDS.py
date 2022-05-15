import sys
import os.path
import configparser
import argparse
import random
import numpy as np
import plotly.graph_objs as go

sys.path.append("../src")
import simulation


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_params_filename",
                        help="Name of file with simulation parameters",
                        type=str,
                        default="../../data/00000000_simulation_params.npz")
    parser.add_argument("--num_samples",
                        help="Number of samples to simulate",
                        type=int, default=10000)
    parser.add_argument("--plot", action="store_false")
    parser.add_argument("--simRes_filename_pattern", type=str,
                        default="../../results/{:08d}_simulation.{:s}",
                        help="simulation results filename pattern")
    args = parser.parse_args()

    sim_params_filename = args.sim_params_filename
    num_samples = args.num_samples
    plot = args.plot
    simRes_filename_pattern = args.simRes_filename_pattern

    sim_params = np.load(sim_params_filename)

    A = sim_params["A"]
    C = sim_params["C"]
    Q = sim_params["Q"]
    R = sim_params["R"]
    m0 = sim_params["m0"]
    V0 = sim_params["V0"]

    x0, x, y = simulation.simulateLDS(N=num_samples, A=A, Q=Q, C=C, R=R,
                                      m0=m0, V0=V0)

    # save simulation results
    sim_prefix_used = True
    while sim_prefix_used:
        simRes_number = random.randint(0, 10**8)
        simRes_metadata_filename = simRes_filename_pattern.format(
            simRes_number, "ini")
        if not os.path.exists(simRes_metadata_filename):
            sim_prefix_used = False
    simRes_results_filename = simRes_filename_pattern.format(simRes_number,
                                                             "npz")

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"sim_params_filename":
                                         sim_params_filename}
    simResConfig["simulation_results"] = {"sim_results_filename":
                                          simRes_results_filename}
    with open(simRes_metadata_filename, "w") as f:
        simResConfig.write(f)
    print("Saving results to {:s}".format(simRes_results_filename))
    np.savez(simRes_results_filename, x=x, y=y)

    if plot:
        fig = go.Figure()
        trace_x = go.Scatter(x=x[0, :], y=x[3, :], mode="markers",
                             showlegend=True, name="state")
        trace_y = go.Scatter(x=y[0, :], y=y[1, :], mode="markers",
                             showlegend=True, name="observation", opacity=0.3)
        trace_start = go.Scatter(x=[x0[0]], y=[x0[1]], mode="markers",
                                 text="x0", marker={"size": 7},
                                 showlegend=False)
        fig.add_trace(trace_x)
        fig.add_trace(trace_y)
        fig.add_trace(trace_start)
        fig.show()

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
