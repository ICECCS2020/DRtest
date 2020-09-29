# DRtest
This is the repository for submission of ICECCS2020
This repository contains the source code for DRtest, a Python library for AI system analysis.

## Documents
- datasets: It contains three well-known datasets including mnist, cifar10 and svhn.
- nmutant_model: It contains several well-use model structure including lenet, resnet and googlenet.
- nmutant_attack: It contains four state-of-the-art attack methods including fgsm, jsma, cw and blackbox.
- nmutant_detection: It contains several state-of-art defense methods including bayesian uncertainty, density estimation, local intrinsic demensionality and input mutation.
- coverage_criteria: It contains six coverage metrics including neuron coverage, k-multisection neuron coverage, neuron boundary coverage, strong neuron activation coverage, top-k neuron coverage and top-k neuron patterns.
- attack_metrics: It contains ten utility metrics of attacks from three aspects including misclassification, imperceptibility and robustness.
- defense_metrics: It contains four utility metrics of defenses including classification accuracy variance, classification rectify/sacrifice ratio, classification confidence variance and classification output stability.
- robustness: It contains two utility metrics of robustness including clever_score and local lipschitz_constance.

### Coverage Criteria
- neuron_coverage(datasets, model, samples_path, de=False)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples, if want to use the original testing data, set "test".
    - de: If the target model is a defense-enhanced model, set 'True'.

- multi_testing_criteria(datasets, model, samples_path, std_range, k_n, k_l, de='False')
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples, if want to use the original testing data, set "test". 
    - std_range: The parameter to difine boundary with std.
    - k_n: The number of sections for neuron output.
    - k_l: The number of top-k neurons in one layer.
    - de: If the target model is a defense-enhanced model, set 'True'.

### Attack Metrics
- acac(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples. 

- actc(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.

- ald(datasets, model, samples_path, p)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.
    - p: p-norm distance.

- ass(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.

- mr(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.
    
- nte(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.

- psd(datasets, model, samples_path, n)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.
    - n: Region size.

- rgb(datasets, model, samples_path, radius)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.
    - radius: The Gaussian radius.

- rgb(datasets, model, samples_path)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - sample_path: The path of adversarial samples.
    
### Defense Metrics
- ccv(datasets, model, de_model)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - de_model: The name of defense-enhanced model.

- cos(datasets, model, de_model)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - de_model: The name of defense-enhanced model.
    
- cr(datasets, model)
    - datasets: The name of target dataset.
    - model: The name of target model.
    - return classification accuracy variance and classification rectify/sacrifice ratio

### Robustness(clever_score)
- collect_gradients(transform, batch_size, dataset, nthreads, model_name, sample_norm, fix_dirty_bug, firstimg, activation, order, compute_slope, ids, numimg, Nsamps, target_type, Niters, save, seed, de)
    - trainsform: input transformation function (defend_reduce, defend_jpeg, defend_png)
    - batch_size: batch size to run model. 0: use default batch size.
    - dataset: choose dataset.
    - nthreads: number of threads for generating random samples in sphere.
    - model_name: Select model.
    - sample_norm: norm of sampling ball (l2, l1 or li).
    - fix_dirty_bug: do not use (UNSUPPORTED).
    - firstimg: start from which image in dataset.
    - activation: activation functions.
    - sample_norm: norm of sampling ball (l2, l1 or li).
    - order: 1: first order bound, 2: second order bound for twice differentiable activations.
    - compute_slope: collect slope estimate.
    - ids: use a filelist of image IDs in CSV file for attack (UNSUPPORTED).
    - numimg: number of test images to load from dataset.
    - Nsamps: number of samples per iterations.
    - target_type: Binary mask for selecting targeted attack classes. bit0: top-2, bit1: random, bit2: least likely, bit3: use --ids override (UNSUPPORTED), bit4: use all labels (for untargeted).
    - Niters: number of iterations. NITERS maximum gradient norms will be collected. A larger value will give a more accurate estimate
    - save: results output path.
    - seed: random seed.
    - de: If model is retrained.
- clever_score(data_folder, reduce_op, user_type, use_slope, untargeted, num_samples, num_images, shape_reg, nthreads, plot_dir, method)
    - data_folder: data folder path.
    - reduce_op: report min of all CLEVER scores instead of avg.
    - user_type: replace user type with string, used for ImageNet data processing.
    - use_slope: report slope estimate. To use this option, collect_gradients.py needs to be run with --compute_slope.
    - untargeted: process untargeted attack results (for MNIST and CIFAR).
    - num_samples: the number of samples to use. Default 0 is to use all samples.
    - num_images: number of images to use, 0 to use all images.
    - shape_reg: to avoid the MLE solver in Scipy to diverge, we add a small regularization (default 0.01 is sufficient).
    - nthreads: number of threads (default is len(c_init)+1).
    - plot_dir: output path for weibull fit figures (empty to disable).
    - method: Fitting algorithm. Please use mle_reg for best results.
### Robustness(Lipschitz Constant)
- local_lipschitz(datasets, model_name, de, attack)
    - datasets: dataset of model trained with.
    - model_name: name of model.
    - de: If trained with adversarial examples.
    - attack: attack to use with adversarial training.
    
    
    
    
    
    
