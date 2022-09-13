best_params_dict = {'Cora': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 128, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Cora', 'decay': 0.00507685443154266, 'directional_penalty': None, 'dropout': 0.046878964627763316, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 0.5, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.022924849756740397, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'adaptive_heun', 'metric': 'accuracy', 'mix_features': False, 'name': 'cora_beltrami_splits', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 1000, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'threshold_type': 'addD_rvR', 'time': 18.294754260552843, 'tol_scale': 821.9773048827274, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Citeseer': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 32, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Citeseer', 'decay': 0.1, 'directional_penalty': None, 'dropout': 0.7488085003122172, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 250, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 128, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.6803233752085334, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.5825086997804176, 'lr': 0.00863585231323069, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'Citeseer_beltrami_1_KNN', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_class': 6, 'num_feature': 3703, 'num_init': 2, 'num_nodes': 2120, 'num_samples': 400, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 4, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'threshold_type': 'addD_rvR', 'time': 7.874113442879092, 'tol_scale': 2.9010446330432815, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Pubmed': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 16, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'cosine_sim', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Pubmed', 'decay': 0.0018236722171703636, 'directional_penalty': None, 'dropout': 0.07191100715473969, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 600, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 1, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.014669345840305131, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'test_acc', 'mix_features': False, 'name': None, 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 8, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'threshold_type': 'addD_rvR', 'time': 12.942327880200853, 'tol_scale': 1991.0688305523001, 'tol_scale_adjoint': 16324.368093998313, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False, 'folder': 'pubmed_linear_att_beltrami_adj2', 'index': 0, 'run_with_KNN': False, 'change_att_sim_type': False, 'reps': 1, 'max_test_steps': 100,  'earlystopxT': 5.0, 'pos_enc_csv': False, 'pos_enc_type': 'GDC'},
                    'CoauthorCS': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': False, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 8, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'CoauthorCS', 'decay': 0.004738413087298854, 'directional_penalty': None, 'dropout': 0.6857774850321, 'dt': 0.001, 'dt_min': 1e-05, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_add': 0.05, 'edge_sampling_epoch': 5, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_rmv': 0.05, 'edge_sampling_space': 'pos_distance', 'edge_sampling_sym': False, 'epoch': 250, 'exact': False, 'fa_layer': False, 'fc_out': False, 'feat_hidden_dim': 128, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.0001, 'gpus': 1, 'grace_period': 20, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 16, 'input_dropout': 0.5275042493231822, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.7181389780997276, 'lr': 0.0009342860080741642, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'CoauthorCS_final_tune_posencGDC', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 4, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'pos_dist_quantile': 0.001, 'pos_enc_csv': False, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'symmetric_attention': False, 'threshold_type': 'addD_rvR', 'time': 3.126400580172773, 'tol_scale': 9348.983916372074, 'tol_scale_adjoint': 6599.1250595331385, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Computers': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': False, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.572918052062338, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Computers', 'decay': 0.007674669913252157, 'directional_penalty': None, 'dropout': 0.08732611854459256, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5973137276937647, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.0035304663972281548, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'computer_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1.7138583550928912, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.249016177876166, 'tol_scale': 127.46369887079446, 'tol_scale_adjoint': 443.81436775321754, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Photo': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': False, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.9282359956104751, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'pearson', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Photo', 'decay': 0.004707800883497945, 'directional_penalty': None, 'dropout': 0.46502284638600183, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 64, 'input_dropout': 0.42903126506740247, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.005560726683883279, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'photo_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0.05783612585280118, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.5824027975386623, 'tol_scale': 2086.525473167121, 'tol_scale_adjoint': 14777.606112557354, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'ogbn-arxiv': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.8105268910037231, 'attention_dim': 32, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'ogbn-arxiv', 'decay': 0, 'directional_penalty': None, 'dropout': 0.11594990901233933, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 2, 'heat_time': 3.0, 'hidden_dim': 162, 'input_dropout': 0, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.21964773835397075, 'leaky_relu_slope': 0.2, 'lr': 0.005451476553977102, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 10000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'arxiv_beltrami_hard_att', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': False, 'num_init': 2, 'num_samples': 200, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'patience': 100, 'pos_enc_hidden_dim': 98, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW64', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.6760155951687636, 'tol_scale': 11353.558848254957, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False}
                    }

