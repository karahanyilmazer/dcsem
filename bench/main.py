#!/usr/bin/env python3
"""
This module is to parse inputs from commandline and call the proper functions from other modules.
"""

import argparse
import os
from file_tree import FileTree
import numpy as np
from scipy import stats as st
from bench import (
    change_model,
    glm,
    spherical_harmonics,
    diffusion_models,
    acquisition,
    image_io,
    model_inversion,
    dti,
)
from fsl.utils.fslsub import submit
import importlib.util


def main(argv=None):
    """
    Wrapper function to parse the input from commandline and run the requested pipeline.

    :param argv: string from command line containing all required inputs
    :return: saves the output images to the specified path
    """
    args = parse_args(argv)
    args.func(args)


def parse_args(argv):
    """
    Parses the commandline input anc checks for the consistency of inputs
    :param argv: input string from commandline
    :return: arg namespce from argparse
    :raises: if the number of provided files do not match with other arguments
    """

    parser = argparse.ArgumentParser("BENCH: Bayesian EstimatioN of CHange")
    parser.set_defaults(func=print_avail_commands)

    subparsers = parser.add_subparsers(dest="commandname")
    diff_train_parser = subparsers.add_parser("diff-train")
    diff_summary_parser = subparsers.add_parser("diff-summary")
    submit_summary_parser = subparsers.add_parser("submit-summary")
    diff_normalise_parser = subparsers.add_parser("diff-normalise")
    glm_parser = subparsers.add_parser("glm")
    inference_parser = subparsers.add_parser("inference")

    # train arguments:
    train_required = diff_train_parser.add_argument_group("required arguments")
    available_models = list(diffusion_models.prior_distributions.keys())
    train_required.add_argument(
        "-m",
        "--model",
        help=f"name of the forward model. Available models:\n{available_models}",
        required=True,
    )
    train_required.add_argument(
        "-o", "--output", help="name of the trained model", required=True
    )

    train_optional = diff_train_parser.add_argument_group("optional arguments")
    train_optional.add_argument(
        "-n",
        "--samples",
        default=10000,
        type=int,
        help="number of training samples (default=10000)",
        required=False,
    )
    train_optional.add_argument(
        "-b", "--bvals", help="b-values for training", required=False
    )

    train_optional.add_argument(
        "--bvecs",
        default=None,
        help="Gradient directions. (default: 64 uniform samples over sphere)",
        required=False,
    )
    train_optional.add_argument(
        "--b0-thresh", default=0.05, type=float, help="threshold for b0 (default=1)"
    )
    train_optional.add_argument(
        "--pmean",
        default=2,
        type=int,
        help="polynomial degree for mean (default=2)",
        required=False,
    )
    train_optional.add_argument(
        "--pcov",
        default=1,
        type=int,
        help="polynomial degree for variance (default=1)",
        required=False,
    )
    train_optional.add_argument(
        "--summarytype",
        type=str,
        help="type of summary measurements. Either sh (spherical harmonic model)"
        " or dt (diffusion tensor model) (default sh)",
        required=False,
    )

    train_optional.add_argument(
        "--sh-degree",
        default=2,
        type=int,
        help=" maximum degree for summary measures (must be even numbers, default=2)",
        required=False,
    )
    train_optional.add_argument(
        "--alpha",
        default=0.0,
        type=float,
        help="regularisation weight for training regression models(default=0)",
        required=False,
    )
    train_optional.add_argument(
        "--change-vecs",
        help="text file for defining vectors of change (refer to documentations)",
        default=None,
        required=False,
    )

    train_optional.add_argument(
        "--verbose",
        help="flag for printing optimisation outputs",
        dest="verbose",
        action="store_true",
        default=False,
    )
    diff_train_parser.set_defaults(func=train_from_cli)

    # fit summary arguments:
    submit_summary_parser.add_argument(
        "--file-tree", help="file-tree text file", required=True
    )
    submit_summary_parser.add_argument(
        "--mask", help="mask in standard space.", required=True
    )
    submit_summary_parser.add_argument(
        "--output", help="Parent directory to store the output.", required=True
    )

    submit_summary_parser.add_argument(
        "--summarytype",
        default="sh",
        type=str,
        help="type of summary measurements. either sh (spherical harmonics)"
        " or dt (diffusion tensor) (default sh)",
        required=False,
    )

    submit_summary_parser.add_argument(
        "--sh-degree",
        default=2,
        help=" degree for spherical harmonics (even number)",
        required=False,
        type=int,
    )
    submit_summary_parser.add_argument(
        "--b0-thresh", default=0.05, type=float, help="b0-threshhold (default=0.05)"
    )
    submit_summary_parser.add_argument("--logdir", help=" log directory")
    submit_summary_parser.set_defaults(func=submit_summary)

    # single subject summary:
    diff_summary_parser.add_argument("--data", required=True)
    diff_summary_parser.add_argument("--bvecs", required=True)
    diff_summary_parser.add_argument("--bvals", required=True)
    diff_summary_parser.add_argument("--mask", required=True)
    diff_summary_parser.add_argument(
        "--xfm", help="Transformation from diffusion to mask", required=False
    )
    diff_summary_parser.add_argument("--output", required=True)
    diff_summary_parser.add_argument("--sh-degree", default=2, type=int, required=False)
    diff_summary_parser.add_argument("--summarytype", default="sh", required=False)
    diff_summary_parser.add_argument(
        "--b0-thresh", default=0.05, type=float, required=False
    )
    diff_summary_parser.add_argument(
        "--normalise", dest="normalise", action="store_true"
    )

    diff_summary_parser.set_defaults(func=summary_from_cli)

    # normalization args
    diff_normalise_parser.add_argument(
        "--study-dir",
        default=None,
        help="Path to the un-normalised summary measurements",
        required=True,
    )

    # glm arguments:
    glm_parser.add_argument(
        "--summarydir", help="Path to summary measurements folder", required=True
    )
    glm_parser.add_argument("--mask", help="Path to the mask", required=True)
    glm_parser.add_argument(
        "--designmat", help="Design matrix for the group glm", required=False
    )
    glm_parser.add_argument(
        "--designcon", help="Design contrast for the group glm", required=False
    )
    glm_parser.add_argument(
        "--subjectlist",
        help="List of subjects in order of the design matrix rows.",
        required=True,
    )
    glm_parser.add_argument("--paired", dest="paired", action="store_true")
    glm_parser.add_argument("--output", help="Output directory", required=True)
    glm_parser.set_defaults(func=glm_from_cli)

    # inference arguments:
    inference_parser.add_argument(
        "--changemodel",
        help="Path to a trained model of change (output of bench diff-train)",
        default=None,
        required=True,
    )
    inference_parser.add_argument("--glmdir", help="Path to the glm dir", required=True)
    inference_parser.add_argument(
        "--output", help="Path to the output dir", required=True
    )
    inference_parser.add_argument(
        "--mask",
        help="Path to the mask (if none passed uses the valid_mask in glm folder)",
        default=None,
        required=False,
    )
    inference_parser.add_argument(
        "--integral-bound",
        help="The maximum value for integrating over the amount of change. (default=1)",
        default=1.0,
        required=False,
    )
    inference_parser.set_defaults(func=inference_from_cli)

    args = parser.parse_args(argv)

    return args


def print_avail_commands(argv=None):
    print("BENCH: Bayesian EstimatioN of CHange")
    print("usage: bench <command> [options]")
    print("")
    print(
        "available commands: diff-train, diff-summary,diff-single-summary,diff-normalise,glm,inference"
    )


def load_objects_from_file(path, keys, defaults):
    """
    Dynamically load a module from a given file path and return the
    required objects in it. If the default value is None and the object is not found it returns an error
    """

    # Load the module from the given path
    spec = importlib.util.spec_from_file_location("external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    res = []
    # Retrieve the objects
    for k, d in zip(keys, defaults):
        if hasattr(module, k):
            obj = getattr(module, k)
        elif d is not None:
            obj = d
        else:
            raise ValueError(
                'The input file must contain a function named "forward_model".'
            )
        res.append(obj)

    return res


def train_from_cli(args):
    available_models = list(diffusion_models.prior_distributions.keys())
    if args.model in available_models:
        funcdict = {
            name: f
            for (name, f) in diffusion_models.__dict__.items()
            if name in available_models
        }
        forward_model = funcdict[args.model]
        prior_dist = diffusion_models.prior_distributions[args.model]
        bvals = acquisition.read_bvals(args.bvals)
        if args.bvecs is None:
            bvecs = np.array(
                diffusion_models.spherical2cart(
                    *diffusion_models.uniform_sampling_sphere(len(bvals))
                )
            ).T
        else:
            bvecs = acquisition.read_bvecs(args.bvecs)

        if args.summarytype is None:
            args.summarytype = "sh"

        func, measurement_names = change_model.summary_decorator(
            model=forward_model,
            bvals=bvals,
            bvecs=bvecs,
            summary_type=args.summarytype,
            sh_degree=int(args.sh_degree),
        )
        kwargs = {"noise_std": 0.0}

        if args.summarytype == "dt":
            normaliser = dti.normalise_summaries
        elif args.summarytype == "sh":
            normaliser = spherical_harmonics.normalise_summaries
        else:
            normaliser = change_model.default_normaliser

    elif (
        os.path.exists(args.model)
        and os.path.isfile(args.model)
        and args.model.endswith(".py")
    ):
        (
            forward_model,
            prior_dist,
            summary_extractor,
            measurement_names,
            normaliser,
            kwargs,
        ) = load_objects_from_file(
            args.model,
            (
                "forward_model",
                "priors",
                "extract_summary",
                "measurement_names",
                "normaliser",
                "kwargs",
            ),
            (None, None, lambda **x: x, [], change_model.default_normaliser, dict()),
        )

        func = lambda **params: summary_extractor(
            forward_model(**params, **kwargs), **kwargs
        )
        try:
            sm = func(**change_model.sample_params(prior_dist, 1))
            if len(measurement_names) == 0:
                measurement_names = [f"sm_{i + 1}" for i in range(len(sm))]
        except TypeError as e:
            raise ValueError(
                f"The prior distribution does not match with the function arguments: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred when trying to sample and call the function: {e}"
            )

    else:
        model_names = ", ".join(list(diffusion_models.prior_distributions.keys()))
        raise ValueError(
            f"model {args.model} is not defined in the library nor is a valid python file. "
            f"Current defined models are:\n {model_names}"
        )

    p_names = [i for p in prior_dist.keys() for i in ([p] if isinstance(p, str) else p)]
    print(f"Parameters of {args.model} are:{p_names}")
    print(
        "The model is trained on the following summary measurements:", measurement_names
    )

    if args.change_vecs is not None:
        with open(args.change_vecs, "r") as reader:
            args.change_vecs = [line.rstrip() for line in reader]

    trainer = change_model.Trainer(
        forward_model=func,
        kwargs=None,
        measurement_names=measurement_names,
        priors=prior_dist,
        normaliser=normaliser,
        change_vecs=args.change_vecs,
    )

    ch_model = trainer.train(
        n_samples=int(args.samples),
        mu_poly_degree=int(args.pmean),
        sigma_poly_degree=int(args.pcov),
        alpha=float(args.alpha),
        parallel=True,
        verbose=args.verbose,
    )

    ch_model.forward_model_name = forward_model.__name__
    ch_model.save(path="", file_name=args.output)
    print("All change models were trained successfully.")


def summary_from_cli(args):
    """
    Wrapper function that parses the input from commandline
    :param args: list of strings containing all required parameters for fit_summary_single_subj()
    """
    xfm = image_io.NNMapping(args.data, args.mask, args.xfm)
    data = xfm.load_native(args.data)
    valid_idx = data.var(axis=1) > 0

    acq = acquisition.Acquisition.from_bval_bvec(args.bvals, args.bvecs, args.b0_thresh)
    if args.summarytype == "sh":
        summaries = spherical_harmonics.fit_shm(
            data[valid_idx], acq, sh_degree=args.sh_degree
        )
        names = spherical_harmonics.summary_names(acq, sh_degree=args.sh_degree)
        if args.normalise:
            print("Summary measures are normalised by b0_mean.")
            summaries = spherical_harmonics.normalise_summaries(names, summaries)
            names = [f"{n}/b0" for n in names[1:]]

    elif args.summarytype == "dt":
        summaries = dti.fit_dti(data[valid_idx], acq)
        names = dti.summary_names(acq)
        if args.normalise:
            print("Summary measures are normalised by b0_mean.")
            summaries = dti.normalise_summaries(names, summaries)
            names = [f"{n}_norm" for n in names[1:]]

    elif (
        os.path.exists(args.summarytype)
        and os.path.isfile(args.summarytype)
        and args.summarytype.endswith(".py")
    ):
        func, names, normaliser = load_objects_from_file(
            args.summarytype,
            ("summary_extractor", "summary_names", "normaliser"),
            (None, [], lambda x: x),
        )
        summaries = func(data[valid_idx])
        if args.normalise:
            print("Summary measures are normalised by the provided function.")
            summaries = normaliser(summaries)

    all_summaries = np.zeros((data.shape[0], len(names)))
    all_summaries[valid_idx] = summaries

    # write to nifti:
    os.makedirs(args.output, exist_ok=True)
    for i, n in enumerate(names):
        xfm.write_std(all_summaries[:, i], f"{args.output}/{n}")

    print(f"Summary measurements are computed.")


def submit_summary(args):
    """
    submits jobs to cluster (or runs serially) to compute summary measurements for the given subjects
    :param args: namespace form parseargs output that contains all addresses to the required files
    (masks, transformations, diffusion data, bvalues, and bvecs) and output folder
    :return: job ids (the results are saved to files once the jobs are done)
    """
    tree = FileTree.read(args.file_tree).update_glob("data")
    subject_jobs = []
    os.makedirs(args.output, exist_ok=True)
    if args.logdir is None:
        args.logdir = args.output

    for subject_tree in tree.iter("data"):
        subj = os.path.basename(subject_tree.get("subjectname"))
        cmd = (
            "bench",
            "diff-summary",
            "--data",
            subject_tree.get("data"),
            "--bvals",
            subject_tree.get("bvals"),
            "--bvecs",
            subject_tree.get("bvecs"),
            "--xfm",
            subject_tree.get("diff2std"),
            "--mask",
            args.mask,
            "--summarytype",
            args.summarytype,
            "--sh-degree",
            str(args.sh_degree),
            "--b0-thresh",
            str(args.b0_thresh),
            "--output",
            f"{args.output}/{subj}",
        )
        # main(cmd[1:])
        subject_jobs.append(submit(cmd, logdir=args.logdir, job_name=f"bench.summary"))

    print(f"Jobs for calculating summary measurements are submitted for all subjects.")


def glm_from_cli(args):
    """
    Runs glm on the summary meausres.
    :param args: output from argparse, should contain desing matrix anc contrast addresss, summary_dir and masks
    :return:
    """
    assert args.paired or (args.designmat is not None and args.designcon is not None)

    os.makedirs(args.output, exist_ok=True)
    with open(args.subjectlist, "r") as file:
        subjs = file.readlines()
    subjs = [s.strip() for s in subjs]
    summaries, invalid_vox, summary_names = image_io.read_summary_images(
        summary_dir=args.summarydir,
        mask=args.mask,
        subject_list=subjs,
        summary_names=None,
    )
    summaries = summaries[:, invalid_vox == 0, :]

    if args.paired:
        data, delta_data, sigma_n = glm.group_glm_paired(summaries)
    else:
        data, delta_data, sigma_n = glm.group_glm(
            summaries, args.designmat, args.designcon
        )

    image_io.write_glm_results(
        data, delta_data, sigma_n, summary_names, args.mask, invalid_vox, args.output
    )
    with open(f"{args.output}/summary_names.txt", "w") as f:
        for item in summary_names:
            f.write("%s\n" % item)

    if any(["md" in x for x in summary_names]):
        y_norm, dy_norm, sigma_n_norm = dti.normalise_summaries(
            data, delta_data, sigma_n, names=summary_names
        )
    elif any(["l2" in summary_names]):
        y_norm, dy_norm, sigma_n_norm = spherical_harmonics.normalise_summaries(
            summary_names, data, delta_data, sigma_n
        )
    else:
        y_norm, dy_norm, sigma_n_norm = data, delta_data, sigma_n

    image_io.write_glm_results(
        y_norm,
        dy_norm,
        sigma_n_norm,
        summary_names,
        args.mask,
        invalid_vox,
        args.output + "/normalised",
    )

    print(f"GLM is done. Results are stored in {args.output}")


def inference_from_cli(args):
    """

    :param args:
    :return:
    """
    data, delta_data, sigma_n, summary_names, coords = image_io.read_glm_results(
        args.glmdir, mask=args.mask
    )
    # perform inference:
    ch_mdl = change_model.ChangeModel.load(args.changemodel)

    idx = [summary_names.index(s) for s in ch_mdl.measurement_names]
    data, delta_data, sigma_n = (
        data[:, idx],
        delta_data[:, idx],
        sigma_n[:, idx, :][:, :, idx],
    )
    posteriors, predictions, peaks, bad_samples = ch_mdl.infer(
        data,
        delta_data,
        sigma_n,
        integral_bound=float(args.integral_bound),
        parallel=True,
    )

    # save the results:
    image_io.write_inference_results(
        args.output,
        ch_mdl.model_names,
        predictions,
        posteriors,
        peaks,
        f"{args.glmdir}/valid_mask",
        coords,
    )

    dv, offset, deviation = ch_mdl.estimate_quality_of_fit(
        data, delta_data, sigma_n, predictions, peaks
    )

    image_io.write_nifti(
        deviation[:, np.newaxis],
        coords,
        f"{args.glmdir}/valid_mask",
        f"{args.output}/sigma_deviations.nii.gz",
    )

    with open(f"{args.output}/model_names.txt", "w") as f:
        for i, item in enumerate(ch_mdl.model_names):
            f.write(f"{i}:{item}\n")

    print(f"Analysis completed successfully.")


def submit_invert(args):
    os.makedirs(args.output, exist_ok=True)
    pe_dir = f"{args.output}/pes/{args.model}"
    os.makedirs(pe_dir, exist_ok=True)

    py_file_path = os.path.realpath(__file__)
    task_list = list()
    for subj_idx, (x, d, bval, bvec) in enumerate(
        zip(args.xfm, args.data, args.bval, args.bvecs)
    ):
        task_list.append(
            f"python {py_file_path} {subj_idx} {d} {x} {bvec} {bval} {args.mask} {args.model} {pe_dir}"
        )

    # if 'SGE_ROOT' in os.environ.keys():
    #     print('Submitting jobs to SGE ...')
    #     with open(f'{args.output}/tasklist.txt', 'w') as f:
    #         for t in task_list:
    #             f.write("%s\n" % t)
    #         f.close()
    #
    #         job_id = run(f'fsl_sub -t {args.output}/tasklist.txt '
    #                      f'-T 240 -N bench_inversion -l {pe_dir}/log')
    #         print('jobs submitted to SGE ...')
    #         fslsub.hold(job_id)
    # else:
    #     os.system('; '.join(task_list))

    else:
        print("parameter estimates already exist in the specified path")

    # apply glm:
    if args.design_mat is not None:
        param_names = diffusion_models.prior_distributions[args.model].keys()
        fit_results, invalids = image_io.read_pes(pe_dir, args.mask)
        x = glm.loadDesignMat(args.design_mat)
        if not fit_results.shape[0] == x.shape[0]:
            raise ValueError(
                f"Design matrix with {x.shape[0]} subjects does not match with "
                f"loaded parameters for {fit_results.shape[0]} subjects."
            )

        pe1 = fit_results[x[:, 0] == 1, :, : len(param_names)].mean(axis=0)
        pe2 = fit_results[x[:, 1] == 1, :, : len(param_names)].mean(axis=0)

        varpe1 = fit_results[x[:, 0] == 1, :, : len(param_names)].var(axis=0)
        varpe2 = fit_results[x[:, 1] == 1, :, : len(param_names)].var(axis=0)

        z_values = (pe2 - pe1) / np.sqrt(
            varpe1 / np.sqrt(x[:, 0].sum()) + varpe2 / np.sqrt(x[:, 1].sum())
        )
        p_values = st.norm.sf(abs(z_values)) * 2  # two-sided

        for d, p in zip(p_values, param_names):
            fname = f"{args.output}/zmaps/{p}"
            image_io.write_nifti(d, args.mask, fname=fname, invalids=invalids)
        print(f"Analysis completed sucessfully, the z-maps are stored at {args.output}")


def invert_from_cli(
    subj_idx, diff_add, xfm_add, bvec_add, bval_add, mask_add, mdl_name, output_add
):
    print("diffusion data address:" + diff_add)
    print("xfm address:" + xfm_add)
    print("bvecs address: " + bvec_add)
    print("bvals address: " + bval_add)
    print("mask address: " + mask_add)
    print("model name: " + mdl_name)
    print("output path: " + output_add)

    bvals = np.genfromtxt(bval_add)
    bvals = np.round(bvals / 1000, 1)
    bvecs = np.genfromtxt(bvec_add)
    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    def_field_dir = f"{output_add}/def_fields/"
    os.makedirs(def_field_dir, exist_ok=True)
    def_field = f"{def_field_dir}/{subj_idx}.nii.gz"
    data, valid_vox = spherical_harmonics.sample_from_native_space(
        diff_add, xfm_add, mask_add, def_field
    )
    data = data / 1000
    params, stds = model_inversion.map_fit(data, 0.01, mdl_name, bvals, bvecs)
    print(f"subject {subj_idx} parameters estimated.")

    # write down [pes, vpes] to 4d files
    fname = f"{output_add}/subj_{subj_idx}.nii.gz"
    image_io.write_nifti(params, mask_add, fname, np.logical_not(valid_vox))
    print(f"Model fitted to subject {subj_idx} data.")


if __name__ == "__main__":
    print("This is BENCH user interface.")
