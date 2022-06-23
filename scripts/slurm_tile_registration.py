import argparse
import configparser
from os import mkdir
from os.path import exists, join


def config_to_dict(config):
    default = config["DEFAULT"]
    register_tiles = config["REGISTER_TILES"]
    mesh_conf = config["MESH_INTEGRATION_CONFIG"]
    warp_conf = config["WARP_CONFIG"]
    kwargs = {
        "sbem_experiment": default["sbem_experiment"],
        "block": default["block"],
        "grid_index": int(default["grid_index"]),
        "start_section": int(register_tiles["start_section"]),
        "end_section": int(register_tiles["end_section"]),
        "batch_size": int(register_tiles["batch_size"]),
        "stride": int(register_tiles["stride"]),
        "overlaps_x": tuple(int(o) for o in register_tiles["overlaps_x"].split(",")),
        "overlaps_y": tuple(int(o) for o in register_tiles["overlaps_y"].split(",")),
        "min_overlap": int(register_tiles["min_overlap"]),
        "min_peak_ratio": float(register_tiles["min_peak_ratio"]),
        "min_peak_sharpness": float(register_tiles["min_peak_sharpness"]),
        "max_deviation": float(register_tiles["max_deviation"]),
        "max_magnitude": float(register_tiles["max_magnitude"]),
        "min_patch_size": int(register_tiles["min_patch_size"]),
        "max_gradient": float(register_tiles["max_gradient"]),
        "reconcile_flow_max_deviation": float(
            register_tiles["reconcile_flow_max_deviation"]
        ),
        "n_workers": int(register_tiles["n_workers"]),
        "dt": float(mesh_conf["dt"]),
        "gamma": float(mesh_conf["gamma"]),
        "k0": float(mesh_conf["k0"]),
        "k": float(mesh_conf["k"]),
        "num_iters": int(mesh_conf["num_iters"]),
        "max_iters": int(mesh_conf["max_iters"]),
        "stop_v_max": float(mesh_conf["stop_v_max"]),
        "dt_max": float(mesh_conf["dt_max"]),
        "prefer_orig_order": mesh_conf["prefer_orig_order"] == "True",
        "start_cap": float(mesh_conf["start_cap"]),
        "final_cap": float(mesh_conf["final_cap"]),
        "remove_drift": mesh_conf["remove_drift"] == "True",
        "use_clahe": warp_conf["use_clahe"] == "True",
        "kernel_size": int(warp_conf["kernel_size"]),
        "clip_limit": float(warp_conf["clip_limit"]),
        "nbins": int(warp_conf["nbins"]),
    }

    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    kwargs = config_to_dict(config)

    run_dir = config["SLURM"]["run_dir"]
    if not exists(run_dir):
        mkdir(run_dir)

    if not exists(join(run_dir, "tile_registration.config")):
        with open(join(run_dir, "tile_registration.config"), "w") as f:
            config.write(f)

    n_sections_per_job = 75
    for i, section_start in enumerate(
        range(kwargs["start_section"], kwargs["end_section"], n_sections_per_job)
    ):
        with open(join(run_dir, f"tile_registration_{i}.config"), "w") as f:
            config["REGISTER_TILES"]["start_section"] = str(section_start)
            config["REGISTER_TILES"]["end_section"] = str(
                min(section_start + n_sections_per_job, kwargs["end_section"])
            )
            config.write(f)

        job_file = join(run_dir, f"slurm_job_{i}.sh")
        with open(job_file, "w") as f:
            f.writelines("#!/bin/bash\n")
            f.writelines(f"#SBATCH --account={config['SLURM']['account']}\n")
            f.writelines(f"#SBATCH --job-name={config['SLURM']['job_name']}\n")
            f.writelines("#SBATCH --exclude=pcl1002\n")
            f.writelines(
                f"#SBATCH --cpus-per-task=" f"{config['SLURM']['cpus_per_task']}\n"
            )
            f.writelines(f"#SBATCH --ntasks={config['SLURM']['ntasks']}\n")
            f.writelines(f"#SBATCH --partition={config['SLURM']['partition']}\n")
            f.writelines(f"#SBATCH --mem={config['SLURM']['mem']}\n")
            f.writelines(f"#SBATCH --gres={config['SLURM']['gres']}\n")
            f.writelines(f"#SBATCH --mail-user={config['SLURM']['mail_user']}\n")
            f.writelines("#SBATCH --mail-type=ALL\n")
            f.writelines("\n")
            f.writelines("START=$(date +%s)\n")
            f.writelines("STARTDATE=$(date -Iseconds)\n")
            f.writelines(
                'echo "[INFO] [$STARTDATE] [$$] Starting SOFIMA tile registration with job ID $SLURM_JOB_ID"\n'
            )
            f.writelines(
                'echo "[INFO] [$STARTDATE] [$$] Running in $(' 'hostname -s)"\n'
            )
            f.writelines(
                'echo "[INFO] [$STARTDATE] [$$] Working directory: ' '$(pwd)"\n'
            )
            f.writelines("\n")
            cudnn_dir = config["SLURM"]["cudnn_dir"]
            f.writelines(f"export " f"CPATH=$CPATH:{cudnn_dir}/local/include/\n")
            f.writelines(
                "export " f"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{cudnn_dir}/local/lib/\n"
            )
            cuda = config["SLURM"]["cuda"]
            f.writelines(f"export PATH=$(echo $PATH | sed 's/cuda/" f"{cuda}/g')\n")
            sofima_dir = config["SLURM"]["sofima_dir"]
            f.writelines(f"export PYTHONPATH='{sofima_dir}':$PYTHONPATH\n")
            f.writelines("\n")
            conda_env = config["SLURM"]["conda_env"]
            f.writelines(
                f"{conda_env}/bin/python "
                f"{run_dir}/run_tile_registration.py --config "
                f"{run_dir}/tile_registration_{i}.config\n"
            )
            f.writelines("EXITCODE =$?\n")
            f.writelines("\n")
            f.writelines("END=$(date +%s)\n")
            f.writelines("ENDDATE=$(date -Iseconds)\n")
            f.writelines(
                'echo "[INFO] [$ENDDATE] [$$] Workflow finished '
                'with code $EXITCODE"\n'
            )
            f.writelines(
                'echo "[INFO] [$ENDDATE] [$$] Workflow execution '
                'time \\(seconds\\) : $(( $END-$START ))"\n'
            )

        # cmd = f"sbatch {job_file}"
        # p = subprocess.Popen(cmd.split())
        # results, errors = p.communicate()
        # print(results)
        # print(errors)
        # time.sleep(1)


if __name__ == "__main__":
    main()
