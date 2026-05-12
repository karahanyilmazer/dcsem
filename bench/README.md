# BENCH  

## What is it?
BENCH (Bayesian EstimatioN of CHange) is a toolbox implemented in python for identifying and estimating changes in the parameters of a biophysical model between two groups of data (e.g. patients and controls). It is an alternative to model inversion, where one estimates the parameters for each group separately and compares the estimations between the groups (e.g. using a GLM). The advantage is that BENCH allows for using over-parameterised models where the model inversion approaches fail because of parameter degeneracies. Currently, BENCH only supports microstructural models of diffusion MRI, but it is extendable to other domains.    

## When is it useful?
Bench is useful when the aim is comparing two groups of data in terms of parameters of a biophysical model; but not estimating the parameters per se. It is particularly advantageous when the model has more free parameters than what can be estimated given the measurements.  

## How does it work?
Using simulated data, we generate models of change that can predict how a baseline measurement changes if any of the parameters of the model changes. Then, in a bayesian framework, we estimate which of the "change models" can better explain the observed change between two groups. For more details about the method, please refer to the [paper](https://www.sciencedirect.com/science/article/pii/S1053811922005638?via%3Dihub).

## How to install it?
You can use the following command to install BENCH: 

```buildoutcfg
pip install git+https://git.fmrib.ox.ac.uk/hossein/bench.git
```

## What are the required inputs to run BENCH?
As bench is an alternative to model fitting, anything that is needed to fit models to groups of subjects is needed for bench as well. This includes:

1. Preprocessed diffusion MRI data for each subject. 
2. Transformations from native diffusion space to a common structural space, e.g. MNI.
3. A ROI mask in the common space that specifies which voxels to analyse. 
4. b-val and b-vec files (the same format as accepted in [FDT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT)).
5. `design.mat` and `contrast.mat` files generated with [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) 's `Glm` tool (`Glm_gui` on Mac). Refer to the GLM section below for details.

## How to use it?
BENCH runs in four main steps that as explained below:

### 1. Train change models:
This step is for training change models for parameters of a specific biophysical model with a given acquisition protocol. It doesn't require any data, and once the training is done the models can be used to investigate any data with the same acquisition protocol. Generally, this step requires a forward model that can map parameters to measurements as well as prior distributions for each parameter (e.g. specifying their range). Currently, a handful of forward models have been implemented. Please refer to the paper for details about the forward models and the priors for the parameters. You can add your own model or update the priors in [diffusion_models.py](bench/diffusion_models.py).

To train models of change you need to run the following command:

```
bench diff-train --model <model_name> 
                 --bval <path_to_bvalue_file> 
                 --output <name_of_output_file>
```

run ``bench diff-train --help`` to see the full list of optional parameters. This command generates a pickle file that contains trained machine learning models.


### 2. Compute summary measurements:
This stage estimates rotationally invariant summary measurements from dMRI data for each subject. All subjects should have the same b-values, which are used for training models of change, but the b-vectors and ordering of measurements can vary across subjects.

Run the following command to estimate the summary measurements:
``` 
    bench diff-summary --data <list of all subjects diffusion data> 
                       --xfm <list of all subjects warp files from native to standard space>
                       --bval <a single bval or a list of bval file.>
                       --bvec <list of bvecs for all subject>
                       --mask <ROI mask in standard space>
                       --study-dir <path to save summary measurements>
```
This command will make a `SummaryMeasurements` directory inside the specified `study_dir` that contains summary measurements per subject, numbered based on the ordering of input from 0 to the number of subjects.
  
### 3. Run a GLM:
This steps runs a group GLM to compute the baseline measurements, the average change between groups and the noise covariance matrices. 

```
bench glm --design-mat <Design matrix for the group glm>  
          --design-con <Design contrast for the group glm>
          --mask <ROI mask in standard space>
          --study-dir <study directory>
```
The design matrix must have one row per subject (with the same order as the input for the previous step) and arbitrary number of columns. The contrast matrix must have two rows where the first one should correspond to the baseline measurement and the second one to the group difference. In the simplest case, where there is no confounding variables the design matrix has two columns that have group memberships and the contrast is `[1 0, -1 1]`. We strongly recommend generating the design and contrast matrices using the FSL's `Glm` tool (`Glm_gui` on a Mac).   

This step produces a directory in `study-dir` that contains `data.nii.gz`, `delta_data.nii.gz`, `variances.nii.gz`, `valid_mask.nii.gz`. The valid mask contains all the voxels where all subjects had valid data, which might be smaller than the user-provided mask if voxels from the user-provided mask lie outside of the brain masks of any subject.

### 4. Inference:
This final stage computes the posterior probability, in each voxel, for any of the change models trained in the first stage using the results of the GLM. 
```
bench inference --model <change model file> 
                --study-dir <study directory>
```

This stage produces a `Results` folder in the study directory that contains a folder for each forward model, e.g. `study_dir/Results/watson_noddi/`.  

## What are the outputs?
The results contain these outputs:
1. one probability map per parameter of the forward model named `pname_probability.nii.gz`. This contains the per voxel probability that change in that parameter can explain the observed change in the data between the two groups. 
2. one estimated amount of change map per parameter of the forward model named `pname_amount.nii.gz`, which contains the estimated amount of change for the corresponding parameter (under the assumption that it caused the change).
3. best explaining model of change in each voxel `inferred_change.nii.gz`. This shows the index of the parameter that can best explain the observed change. The ordering is matched with the order of appearance in the prior distributions in [diffusion_models.py](bench/diffusion_models.py).


## Use in non-diffusion models and data
We designed BENCH to be as modular as possible, meaning that any stage is a separate module that can be replaced by user defined code. Particularly to apply it to other domains one needs to provide the followings:
1. A biophysical model that maps parameters to summary measurements. (a callable function)
2. Prior distribution of the parameters of the model (a dictionary with keys being the parameters and values scipy stats distribution objects)
3. A script that computes summary measurements from raw data.

You can embed the above in the corresponding python files and use the rest of the command line as described above, or use the python APIs which give you more flexibility to implement your own use case.

To learn more about how to apply BENCH to your own data you can access to [practical for applying BENCH to diffusion data](https://users.fmrib.ox.ac.uk/~hossein/bench/).

Feedbacks or suggestions for improving the toolbox are highly appreciated. [email address](mailto:hossein.rafipoor@ndcn.ox.ac.uk)

