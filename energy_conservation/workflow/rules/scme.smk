rule run_scme:
    input:
        xyz_file="",
    output:
        xyz_file="",
        initial_data="",
        final_data="",
    threads: 1
    params:
        method="VelocityVerlet",
        n_iter=lambda wc: INPUT_DICT[wc.sample].n_iter,
        fmax=0.01,
        timestep=lambda wc: INPUT_DICT[wc.sample].dt,
        temperature=300,
        pbc=[False, False, False],
        logfile="",
    script:
        "scripts/run_scme.py"