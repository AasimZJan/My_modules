import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner
from eryn.ensemble import EnsembleSampler
from eryn.backends import HDFBackend
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import multiprocessing

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def get_start_state_from_prior(sampler_dict):
    """
    Generate initial sampler states by drawing from the prior distributions.

    Parameters
    ----------
    sampler_dict : dict
        Dictionary containing the sampler configuration.
        Expected keys:
            - 'models' : dict
                Keys are branch names, values are tuples of (ndims, nleaves_min, nleaves_max).
            - 'priors' : dict
                Keys are branch names, values are prior distributions with an `.rvs` method.
            - 'number_of_temperatures' : int
            - 'number_of_walkers' : int

    Returns
    -------
    dict
        A dictionary mapping branch names to NumPy arrays of shape:
        (number_of_temperatures, number_of_walkers, nleaves_max[branch_name]),
        containing samples drawn from the respective priors.
    """
    # get branch names and max number of leaves from the dictionary
    branch_names = list(sampler_dict['models'].keys())
    nleaves_max = {branch_name: sampler_dict['models'][branch_name][2] for branch_name in branch_names}

    # return inital state from priors
    return {name: sampler_dict['priors'][name].rvs(size=(
                sampler_dict['number_of_temperatures'],
                sampler_dict['number_of_walkers'],
                nleaves_max[name]
                )) 
            for name in branch_names
            }

def initiate_and_run_eryn_sampler(sampler_dictionary, verbose: bool = False, save_sampler_dictionary: bool = True):
    """
    Initialize and run an Eryn EnsembleSampler based on the given configuration.

    Parameters
    ----------
    sampler_dictionary : dict
        Dictionary containing configuration options for the sampler.

        Required keys
        -------------
        - priors : dict
            Priors for each branch.
        - log_likelihood_function : callable
            Log-likelihood function for the sampler.
        - extra_log_likelihood_arguments : tuple
            Additional positional arguments passed to the log-likelihood function.
        - models : dict
            Dictionary where keys are branch names and values are tuples of:
            (ndims, min_instances, max_instances).
        - reversible_jump_moves : bool or list
            Flag or list of reversible-jump moves.
        - moves : list
            Additional sampler moves.

        Optional keys (with defaults)
        ------------------------------
        - number_of_walkers : int, default 30
            Number of walkers per temperature.
        - number_of_temperatures : int, default 10
            Number of temperature levels for parallel tempering.
        - pool : bool, default False
            Whether to use multiprocessing pool for parallel likelihood evaluation.
        - pool_processes : int, default 4
            Number of processes to use if pool is enabled.
        - start_state : ndarray, default drawn from prior
            Initial state of the walkers.
        - iterations : int, default 1000
            Number of MCMC iterations.
        - burnin_iterations : int, default 6000
            Number of burn-in iterations.
        - thin_by : int, default 1
            Thinning factor for stored chain samples.
        - store : bool, default True
            Whether to store samples in the backend.
        - h5_file : str
            Path (including filename) to the HDF5 file for sampler backend storage. By default it saves the h5 file as eryn_output.h5 in current working directory.

    verbose : bool, optional
        If True, prints detailed setup information. Default is False.

    save_sampler_dictionary : bool, optional
        If True, saves a JSON copy of the sampler configuration (with non-serializable
        objects removed) in the same directory as the provided HDF5 file. Default is True.

    Returns
    -------
    sampler : EnsembleSampler
        The initialized and run Eryn EnsembleSampler instance.
    """
    # work on a copy of the dictionary
    sampler_dict = sampler_dictionary.copy()
    
    # check arguments and assign default values for ease
    defaults = {
        "pool": False,
        "pool_processes": 4,
        "start_state": get_start_state_from_prior(sampler_dict),
        "iterations": 1000,
        "burnin_iterations": 6000,
        "thin_by": 1,
        "store": True,
        "h5_file": os.path.join(os.getcwd(), "eryn_output.h5"),
        "number_of_walkers": 30,
        "number_of_temperatures": 10,
    }
    for key, val in defaults.items():
        print(f"Parsing through sampler dictionary")
        if key not in sampler_dict:
            sampler_dict[key] = val
            if verbose:
                print(f"\tSetting '{key}' to {val}")
    
    # handling some common mistakes
    if sampler_dict['thin_by'] <= 0:
        print(f'Warning: Invalid "thin_by" = {sampler_dict['thin_by']} (must be > 0), setting it to default of {defaults["thin_by"]}')
        sampler_dict['thin_by'] = defaults['thin_by']
    	
    # get branches and number of branches from the dictionary
    branch_names = list(sampler_dict['models'].keys())
    nbranches = len(branch_names)   
    # get number of dimensions for each model, max number of leaves and min number of leaves
    ndims = {branch_name: int(sampler_dict['models'][branch_name][0]) for branch_name in branch_names}
    nleaves_min = {branch_name: int(sampler_dict['models'][branch_name][1]) for branch_name in branch_names}
    nleaves_max = {branch_name: int(sampler_dict['models'][branch_name][2]) for branch_name in branch_names}
   
    # print for sanity checks
    if verbose:
        print(f"Eryn sampler initiated with {sampler_dict['number_of_temperatures']} temperatures and {sampler_dict['number_of_walkers']} walkers per temperature.")
        for name in branch_names:
            print(f'\tFor model "{name}", number of dimensions: {ndims[name]}, min/max instances: {nleaves_min[name]}/{nleaves_max[name]}')
    
    # save the sampler_dictionary if requested (with unserializable objects removed)
    if save_sampler_dictionary:
        sampler_dict_save = sampler_dict.copy()
        path = os.path.dirname(os.path.abspath(sampler_dict["h5_file"]))
        
        # remove objects that json doesn't like
        for key in ["priors", "moves", "log_likelihood_function", "extra_log_likelihood_arguments", "start_state"]:
            sampler_dict_save.pop(key, None)
        
        # save json
        filename = f'{path}/sampler_dict.json'
        try:
            with open(filename, 'w') as f:
                json.dump(sampler_dict_save, f)
            if verbose:
                print(f"Sampler dictionary successfully saved to {filename}")
        except IOError as e:
            print(f"Error saving file: {e}")  

    # Common kwargs for EnsembleSampler
    sampler_kwargs = dict(
        nwalkers = int(sampler_dict["number_of_walkers"]),
        ndims = ndims,
        log_like_fn = sampler_dict["log_likelihood_function"],
        priors=sampler_dict["priors"],
        nbranches=nbranches,
        branch_names=branch_names,
        tempering_kwargs=dict(ntemps=int(sampler_dict["number_of_temperatures"]), Tmax=np.inf),
        args=sampler_dict["extra_log_likelihood_arguments"],
        rj_moves=sampler_dict["reversible_jump_moves"],
        moves=sampler_dict["moves"],
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        backend=HDFBackend(sampler_dict["h5_file"]),
    )

    # Initialize sampler with or without multiprocessing
    if sampler_dict["pool"] is not False:
        with multiprocessing.Pool(processes=int(sampler_dict["pool_processes"])) as pool:
            sampler = EnsembleSampler(**sampler_kwargs, pool=pool)
            sampler.run_mcmc(
                sampler_dict["start_state"],
                sampler_dict["iterations"],
                burn=sampler_dict["burnin_iterations"],
                progress=True,
                thin_by=sampler_dict["thin_by"],
                store=sampler_dict["store"],
            )
    else:
        sampler = EnsembleSampler(**sampler_kwargs, pool=None)
        sampler.run_mcmc(
            sampler_dict["start_state"],
            sampler_dict["iterations"],
            burn=sampler_dict["burnin_iterations"],
            progress=True,
            thin_by=sampler_dict["thin_by"],
            store=sampler_dict["store"],
        )

    return sampler

def save_truths_as_json(truths_dict, save_path):
    """Save ground truth dictionary to JSON."""
    filename = f'{save_path}/truths_dict.json'
    try:
        with open(filename, 'w') as f:
            json.dump(truths_dict, f, cls=NumpyEncoder)
            print(f"Truths dictionary successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving file: {e}")

def plot_eryn_output(sampler_object, save_path, truth_file=None, plots_name_extension=1):
    """
    Generate and save diagnostic plots from an Eryn sampler run.

    The function produces, for each model:
        1. Histogram of the number of instances sampled in the cold chain.
        2. (If a truth file is provided) Log-posterior odds ratios vs. number of instances.
        3. Combined corner plot of all walker samples.
        4. Corner plots per walker.
        5. Log-likelihood distribution of all walkers.
        6. Log-likelihood distribution per walker.

    Parameters
    ----------
    sampler_object : EnsembleSampler
        An initialized and run Eryn EnsembleSampler object containing chain data.

    save_path : str
        Directory path where plots will be saved. Created if it does not exist.

    truth_file : str, optional
        Path to a JSON file containing the ground truth configuration.
        If provided, truth values will be used for reference in the plots.
    
    plots_name_extension : (str, float, int), optional
        Extension to plot name, helpful when plotting mutliple outputs for same data.

    Returns
    -------
    None
        The function saves plots as PNG files in the specified directory.
    """
    # create folder
    if not(os.path.exists(save_path)):
        print(f'Folder does not exist, creating one: {save_path}')
        os.mkdir(save_path)
    else:
        print(f'Folder exists in {save_path}')
    
    # identify is truth file exists
    use_truths = False
    if truth_file is not None:
        use_truths = True
        with open(truth_file, 'r') as f:
            truths_dict = json.load(f)

    # list of models for convinience
    models = list(sampler_object.get_nleaves().keys())

    # plot (per model) diagnostics 1) number of instances 2) ln posterior odds ratio w.r.t truth 3) Combined walker corner plot for the chain 4) Corner plot per walker 
    for model in models:
        # 1) Number of instances of model in the cold chain
        nleaves = sampler_object.get_nleaves()[model][:, 0].flatten() 
        bins = np.arange(nleaves.min()-0.5, nleaves.max()+1.5)
        plt.title(f'Model:{model}')
        plt.hist(nleaves, bins=bins, align='mid', rwidth=0.5, color='royalblue')
        if use_truths:
            plt.axvline(len(truths_dict[model]), linestyle='--', color='grey')
        plt.savefig(save_path + f'/leaves_plot_{model}_{plots_name_extension}.png', dpi=200, bbox_inches='tight')
        plt.clf()

        # 2) Plot log(posterior Odds) v/s N
        if use_truths:
            # number of instances the sampler sampled  N_truth instances of the model
            nleaves_base = len(np.argwhere(nleaves==len(truths_dict[model])).flatten())
            nleaves_base = max(nleaves_base, 1)  # avoid division by zero
            for i in np.arange(nleaves.max()+1):
                # number of instances the sampler sampled i instances of the model
                nleaves_here = len(np.argwhere(nleaves==i).flatten())
                nleaves_here = max(nleaves_here, 1) # avoid division by zero
                # to prevent error
                if nleaves_here == 0:
                    nleaves_here = 1
                plt.scatter(i, np.log(nleaves_here/nleaves_base), color='royalblue')
            plt.xlabel('N')
            plt.ylabel('ln(Odds)')
            plt.axvline(x = len(truths_dict[model]), linewidth = 1.0, linestyle = '--', color='grey')
            plt.savefig(save_path + f'/log_posterior_odds_plot_{model}_{plots_name_extension}.png', dpi=200, bbox_inches='tight')
            plt.clf()

        # 3) Plot samples from all walkers
        cold_chain = sampler_object.get_chain()[model][:, 0, :, :, :]
        # reshape and remove NaNs
        dimensions_here = cold_chain.shape[-1]
        combined_samples = cold_chain.reshape(-1, dimensions_here)
        combined_samples = combined_samples[~np.isnan(combined_samples[:, 0])]
        # plot and save
        fig = corner.corner(combined_samples, truth_color='black', color='royalblue', plot_datapoints=False, plot_density=True, no_fill_contours=True, contours=True, levels=[0.9])
        fig.savefig(save_path + f'/corner_plot_combined_walkers_{model}_{plots_name_extension}.png', dpi=200, bbox_inches='tight')
        
        # 4) Plot samples from each walkers
        nwalkers = cold_chain.shape[1]
        cmap = cm.get_cmap("tab20")
        labels = []
        for walker in np.arange(nwalkers):
            # get colors and label
            color = mcolors.to_hex(cmap(walker / nwalkers))
            label = f"Walker {walker}"
            labels.append(label)
            # reshape and remove NaNs
            samples_here = cold_chain[:, walker, :, :].reshape(-1, dimensions_here)
            samples_here = samples_here[~np.isnan(samples_here[:, 0])]
            if walker == 0:
                fig = corner.corner(samples_here, color=color, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=[0.9])
            else:
                corner.corner(samples_here, color=color, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=[0.9], fig=fig)
        fig.legend(labels,loc="upper right", bbox_to_anchor=(1.2, 1.0), title="Walkers")
        fig.savefig(save_path + f'/corner_plot_per_walker_{model}_{plots_name_extension}.png', dpi=200, bbox_inches='tight')

    # 5) Plot log likelihood from all walkers
    cold_chain_log_likelihood =  sampler_object.get_log_like()[:, 0, :]
    cold_chain_log_likelihood_flattened = cold_chain_log_likelihood.reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.set_xlabel('log-likelihood')
    ax.hist(cold_chain_log_likelihood_flattened, bins=50, histtype='step', color='royalblue', linewidth = 1.5, density=True)
    fig.savefig(save_path + f'/log_likelihood_distribution_combined_walkers_{plots_name_extension}.png', dpi=200, bbox_inches='tight')

    # 6) Plot log likelihood from each walker
    fig, ax = plt.subplots()
    ax.set_xlabel('log-likelihood')
    for walker in np.arange(nwalkers):
        # get colors and label
        color = mcolors.to_hex(cmap(walker / nwalkers))
        label = f"Walker {walker}"
        labels.append(label)
        ax.hist(cold_chain_log_likelihood[:, walker], bins=50, histtype='step', color=color, linewidth = 1.5, density=True)
    fig.legend(labels, loc="upper right", bbox_to_anchor=(1.2, 1.0), title="Walkers")
    fig.savefig(save_path + f'/log_likelihood_distribution_per_walker_{plots_name_extension}.png', dpi=200, bbox_inches='tight')

    # 7) Plot corner per N per model
    # for model in models:
    #     nleaves = sampler_object.get_nleaves()[model][:, 0, :].reshape(-1, 1)
    #     cold_chain = sampler_object.get_chain()[model][:, 0, :, :, :]
    #     # reshape samples
    #     max_nleaves, dimensions_here = cold_chain.shape[-2], cold_chain.shape[-1]
    #     combined_samples = cold_chain.reshape(-1, max_nleaves, dimensions_here)

    #     cmap = cm.get_cmap("tab20")
    #     labels = []
    #     for i in np.arange(1, max_nleaves+1):
    #         color = mcolors.to_hex(cmap(i / max_nleaves))
    #         label = f"Leaves {i}"
    #         labels.append(label)

    #         indices_here = np.argwhere(nleaves == i).flatten()
    #         samples_here = combined_samples[indices_here,:, :].reshape(-1, dimensions_here)
    #         samples_here = samples_here[~np.isnan(samples_here[:, 0])]
    #         try:
    #             if i == 1:
    #                 fig = corner.corner(samples_here, color=color, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=[0.9])
    #             else:
    #                 corner.corner(samples_here, color=color, plot_datapoints=False, plot_density=False, no_fill_contours=True, contours=True, levels=[0.9], fig=fig)
    #         except Exception as e:
    #             print(f'ERROR: {e}')
    #             continue
    #     fig.legend(labels, loc="upper right", bbox_to_anchor=(1.2, 1.0), title="Leaves")
    #     fig.savefig(save_path + f'/corner_plot_per_leaf_{model}_{plots_name_extension}.png', dpi=200, bbox_inches='tight')
