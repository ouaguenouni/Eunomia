print("begin")
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Eunomia.preferences import *
from Eunomia.additive_functions import *
from Eunomia.alternatives import *
from Eunomia.sampling import *
from Eunomia.mcmc import *
from Eunomia.degree import *
from Eunomia.experiments import *
print("import sucessfull")

TEST_NAME = "T2330"
REPLACE = False

n_vals = [4,5,6,7,8,9,10]
sigma_weights = 10
sigma_w_vals = [1, 1e-1, 1e-2]
sigma_p_vals = [1, 1e-1, 1e-2]
n_samples = 1000

for n in n_vals:
  for k in range(1,n-1):
    for m in list(2**i for i in range(2,n)):
      for sigma_w in sigma_w_vals:
        for sigma_p in sigma_p_vals:

          run_d = {
              "n":n,
              "k":k,
              "m":m,
              "sigma_w":sigma_w,
              "sigma_p":sigma_p,
              "n_samples" : n_samples
          }

          found = find_experiment_file(run_d, TEST_NAME)

          if found:
            date = found
            print(f"Found file: {file_name}")
            
            if REPLACE:
              print('Replacing it...')
            else:
              continue
          else:
            file_name = compute_experiment_file_name(run_d, TEST_NAME)
            print("Registering the results in : ",file_name)


          theta = generate_additive_theta(n,k)
          weights = generate_normal_weights(theta, sigma_weights)
          alternatives = generate_random_alternatives_matrix(m,n)
          ranks = compute_ws_ranks(alternatives, theta, weights)
          t_sv = compute_semivalues(n, theta, weights, lambda x:1)
          preferences = PreferenceModel(alternatives, ranks)
          data = preferences.generate_preference_matrix(theta)
          data = torch.tensor(data).float()
          t = time.time()
          model = posterior_sampling_model(data, sigma_w = sigma_w, sigma_p = sigma_p)
          diag, sampled_weights, sigmas = sample_model(model, data , "w", "sigma", warmup_steps = 200, num_samples = n_samples, return_diag = True)
          t = time.time() - t
          accs_d = get_acc_distribution(data, sampled_weights, sigmas)

          predicted_rankings = [np.argsort(compute_semivalues(n, theta, weights, lambda x:1))[::-1] for weights in sampled_weights]
          kt_d = get_kt_distribution(predicted_rankings, np.argsort(t_sv))

          file_path = record_experiment_results(run_d, TEST_NAME)
          run_d["time"] = t
          run_d["weights"] = weights.tolist()
          run_d["predicted_rankings"] = [i.tolist() for i in predicted_rankings]
          run_d["accuracy_distribution"] = [i.tolist() for i in accs_d] 
          run_d["kt_d"] = [i.tolist() for i in kt_d]
          run_d["diag"] = dict(diag)
          print("diag: ", diag)
          with open(file_path, 'w') as file:
            yaml.dump(run_d, file, default_flow_style=False)
