## **Steps to reproduce the results described in "Dynamic PA power mode DPD on base of 2-dimensional Chebyshev polynomials"**

- Use **master** branch for simulations reproduction and results visualization.

- Paper experiments/classic_single_layer/Dynamic_DPD.pdf describes DPD performance achieved by classic 1D and 2D Chebyshev polynomials-based non-linear classic models. Folder experiments/classic_single_layer includes run files and experimental results provided in article.

- **Reproduction of paper simulations:**
    1. Copy config.yaml file from: 
    - experiments/classic_single_layer/1_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm/36_param_4_slot_61_cases_27_delay (for reproduction of 1D Chebyshev polynomials results) or 
    - experiments/classic_single_layer/10_pow_dim_lin_scale_corr_fraq_del_aligned_gain_mw_m16_0dBm/22_param_4_slot_61_cases_4_delay (for reproduction of 2D Chebyshev polynomials results) into classic_single_layer folder.
    2. Choose simulations parameters:
    - train_type within config file (SGD, LS, Mixed Newton), 
    - device to calculate simulation on.
    - Pay attention to **chunk_num** parameter, which determines the number of chunks to divide whole signal into: SGD updates parameters every chunk, LS and Mixed Newton accumulate hessian and gradient among all chunks after that updates parameters. **chunk_num** is provided to save GPU memory while model training.
    - Choose trial_name. If trial_name equals None, then experiments names are chosen automatically, otherwise run.py creates folder with the name, set to trial_name by user.
    - Pay attention, that is overwrite_file is True, then run.py overwrites results in case if folder with given trial_name name already exists.
    3. After model training run forward_path.py. It generates:
    - numpy.array PA input (x.npy), PA non-linear distortion output (d.npy) and model output (y.npy) along whole dataset (all 61 PA output power cases 0.069 W - 0.912 W). In order to check PA output in DPD-on mode calculate x + d - y signal. In order to check PA output in DPD-off mode calculate x + d signal. 
    - ACLR performance .pkl files, which include dictionary with "ACLR" and "power_linear" keys. Pay attention, that ACLR is provided in logarithmic scale (dB), whereas power_linear shows PA **output** power in linear scale (W).

- All results, provided in paper Dynamic_DPD.pdf could be visualized by experiments/classic_single_layer/plot_graphs/plot_graphs.py file.

- Currently classic_single_layer experimental folder is ready for exploitation and reproduction. The other experimental folders are under debug.

- NN-based models from model/cvcnn.py, model/rvcnn.py currently debugged for static DPD-task and could be modified for dynamic DPD simulations in further research.