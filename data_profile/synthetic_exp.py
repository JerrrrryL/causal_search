import concurrent.futures
import os
import warnings
warnings.filterwarnings('ignore')
import time
import random
from sklearn.linear_model import LinearRegression
from discovery import *
import pickle
from sklearn.preprocessing import scale

def save_diagram(G, name, coeffs=None):
    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(G, dist=None, weight='weight', scale=2)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, font_size=10)
    if coeffs is not None:
        edge_labels = {k: round(v, 2) for k, v in coeffs.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(name + '.png', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    

def random_pair(n):
    i = random.randint(min(10, n//3), min(100, n//2))
    j = random.randint(i+1, min(n-1, i+n//2))
    return i, j


def find_non_descendants(G, treatment):
    # Find all descendants of the treatment node
    descendants = nx.descendants(G, treatment)
    
    # Add the treatment node itself to the descendants
    descendants.add(treatment)
    
    # Get all nodes in the graph
    all_nodes = set(G.nodes())
    
    # Find non-descendants by subtracting descendants from all nodes
    non_descendants = all_nodes - descendants
    
    return non_descendants


def accuracy_exp_iteration(dp, treatment, target, mi_thresholds, 
    approx, hist, factorized_hist, factor, join_key, device):
    parents = set(list(dp.G.predecessors(treatment)))
    gt = causal_effect(dp.D, treatment, target, parents)
    # dp.generate_partitions_from_D(treatment, target, join_key)
        
    results = {}
    
    for mi_threshold in mi_thresholds:
        dm = DataMarket(device)
        dm.add_seller(dp.data_corpus, "synthetic", [[dp.join_key]], dp.join_key_domain, 
                      [col for col in dp.data_corpus if col != dp.join_key])
        cd = ConDiscovery(dm, mi_threshold=mi_threshold, approx=approx, verbose=False,
                          hist=hist, factorized_hist=factorized_hist, factor=factor, device=device)
        est_suna, preprocess_time, end_to_end_time, search_time, update_cor_time, _, _ = cd.compute_treatment_effect(
            dp.data_in, [['join_key']], treatment, target)
        print(f"Treatment: {treatment}, Outcome: {target}, Error: {(gt - est_suna) ** 2}, Estimation: {est_suna}, Ground Truth: {gt}")
        # print(f"Set of confounders: {cd.conf_set[(treatment, target)]}")
        # print(f"Estimated causal effect: {causal_effect(dp.D, treatment, target, cd.conf_set[(treatment, target)])}")
        # print(f"Discovered set of confounders: {cd.conf_set}; Estimation: {est_suna}; Ground Truth is {gt}")
        # print(f"Is the adjustment set Valid: {dp.is_valid_Z(treatment, target, cd.conf_set[(treatment, target)])}")
        conf_set = cd.conf_set[(treatment, target)]
        results[mi_threshold] = {
            'se': (gt - est_suna) ** 2,
            'size': len(conf_set),
            'preprocess': preprocess_time, 
            'end_to_end': end_to_end_time, 
            'search': search_time, 
            'update_res': update_cor_time, 
        }
    return results, gt


def accuracy_exp(runs, num_nodes=[100], mi_thresholds=[0.02], 
    gpu=False, approx=False, hist=False, factorized_hist=False, factor=1
):
    random.seed(20)
    np.random.seed(20)
    if gpu: device = "cuda"
    else: device = "cpu"
    join_key = ['join_key']
    all_results_suna, all_results_nd, all_results_baseline = {}, {}, {}
    gt_dicts = {}
    iteration_pairs = [(num_node, run_num) for num_node in num_nodes for run_num in range(runs)]
    # print(iteration_pairs)

    for num_node, run_num in iteration_pairs:
        with open(f'experiment/datasets/synthetic/data_{num_node}_{run_num}.pkl', 'rb') as file:
            dp, treatment, target = pickle.load(file)
            result_suna, gt = accuracy_exp_iteration(
                dp, treatment, target, mi_thresholds, approx, 
                hist, factorized_hist, factor, join_key, device)
        
            if num_node not in all_results_suna:
                gt_dicts[num_node] = []
                all_results_suna[num_node] = {}

            gt_dicts[num_node].append(gt)

            for mi_threshold, res in result_suna.items():
                if mi_threshold not in all_results_suna[num_node]:
                    all_results_suna[num_node][mi_threshold] = {
                        'se': 0,
                        'size': 0,
                        'preprocess': 0, 
                        'end_to_end': 0, 
                        'search': 0, 
                        'update_res': 0,
                    }

                all_results_suna[num_node][mi_threshold]['se'] += res.get('se', 0)
                all_results_suna[num_node][mi_threshold]['size'] += res.get('size', 0)
                all_results_suna[num_node][mi_threshold]['preprocess'] += res.get('preprocess', 0)
                all_results_suna[num_node][mi_threshold]['end_to_end'] += res.get('end_to_end', 0)
                all_results_suna[num_node][mi_threshold]['search'] += res.get('search', 0)
                all_results_suna[num_node][mi_threshold]['update_res'] += res.get('update_res', 0)
        
    return all_results_suna, gt_dicts

def accuracy_exp_base(runs, num_nodes=[100], nd_only=False):
    all_results_nd, all_results_base = {}, {}
    for num_node in num_nodes:
        all_results_nd[num_node] = {'se': 0, 'size': 0, 'time': 0}
        all_results_base[num_node] = {'se': 0, 'size': 0, 'time': 0}
        datasets = []
        treatments = []
        treatment_inds = []
        for run_num in range(runs):
            with open(f'experiment/datasets/synthetic/data_{num_node}_{run_num}.pkl', 'rb') as file:
                res = pickle.load(file)
            datasets.append(res[0].D[res[0].ordered_nodes])
            treatments.append(res[1])
            treatment_inds.append(int(res[1][1:]))

        time_topo_list, time_full_list, est_parents_nodes_list, est_ND_nodes_list = parallel_direct_lingam(
            datasets, treatments, treatment_inds, nd_only=nd_only)
        for i in range(runs):
            with open(f'experiment/datasets/synthetic/data_{num_node}_{i}.pkl', 'rb') as file:
                res = pickle.load(file)
            dp, treatment, outcome = res[0], res[1], res[2]
            nd_var_est = [v for v in est_ND_nodes_list[i] if v != outcome]

            est_nd = causal_effect(
                dp.D, treatment, outcome, set(nd_var_est))
            
            parents = set(list(dp.G.predecessors(treatment)))
            gt = causal_effect(dp.D, treatment, outcome, parents)
            
            all_results_nd[num_node]['se'] += (gt-est_nd)**2
            all_results_nd[num_node]['size'] += len(est_ND_nodes_list[i])
            all_results_nd[num_node]['time'] += time_topo_list[i]
            
            if not nd_only:
                par_var_est = [v for v in est_parents_nodes_list[i] if v != outcome]
                est_base = causal_effect(
                    dp.D, treatment, outcome, set(par_var_est))

                all_results_base[num_node]['se'] += (gt-est_base)**2
                all_results_base[num_node]['size'] += len(est_parents_nodes_list[i])
                all_results_base[num_node]['time'] += time_full_list[i]
    
    return all_results_nd, all_results_base

def gen_synthetic_data(runs, num_nodes, num_samples):
    random.seed(0)
    np.random.seed(0)
    os.makedirs('experiment/datasets/synthetic', exist_ok=True)
    for run_num in range(runs):
        for num_node in num_nodes:
            dp = DataProfile()
            dp.generate_G(num_node)
            dp.generate_D_from_G(num_samples=num_samples)

            treatment_ind, target_ind = random_pair(num_node)
            treatment, target = dp.ordered_nodes[treatment_ind], dp.ordered_nodes[target_ind]
            # gt = dp.get_ground_truth(treatment, outcome)
            dp.generate_partitions_from_D(treatment, target, ['join_key'])
            res = (dp, treatment, target)

            with open(f'experiment/datasets/synthetic/data_{num_node}_{run_num}.pkl', 'wb') as file:
                pickle.dump(res, file)
