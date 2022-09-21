def get_experiment_name(linking_pipeline, split):
    cand_approach = linking_pipeline.myranker.method
    if linking_pipeline.myranker.method == "deezymatch":
        cand_approach += "+" + str(
            linking_pipeline.myranker.deezy_parameters["num_candidates"]
        )
        cand_approach += "+" + str(
            linking_pipeline.myranker.deezy_parameters["selection_threshold"]
        )
    link_approach = linking_pipeline.mylinker.method
    if linking_pipeline.mylinker.method == "reldisamb":
        link_approach += "+" + str(linking_pipeline.mylinker.rel_params["ranking"])
    experiment_name = linking_pipeline.mylinker.rel_params["training_data"]
    if linking_pipeline.mylinker.rel_params["training_data"] == "lwm":
        experiment_name += "_" + cand_approach
        experiment_name += "_" + link_approach
        experiment_name += "_" + split
    if linking_pipeline.mylinker.rel_params["training_data"] == "aida":
        experiment_name += "_" + cand_approach
        experiment_name += "_" + link_approach

    return experiment_name
