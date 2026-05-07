import gnn4s

def main():
    # configuration
    config = gnn4s.load_config('./config/mol_doc_pdb_train.yaml')
    
    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)

    # Configuring file system resource sharing strategy
    gnn4s.utils.set_sharing_strategy('file_system')
    # Setting file descriptor limits in multi-process training
    gnn4s.utils.setrlimit(64000)
    # Configuring optimization options for the CuDNN library
    gnn4s.utils.use_cudnn_benchmark(config.cudnn_benchmark)

    # dataset
    load_fn = gnn4s.data.load_lm_embeddings
    process_fn = gnn4s.data.process_complex
    t_to_sigma = gnn4s.data.TimeToSigma(
        tr_sigma_min=config.diffusion.tr_sigma_min, tr_sigma_max=config.diffusion.tr_sigma_max,
        rot_sigma_min=config.diffusion.rot_sigma_min, rot_sigma_max=config.diffusion.rot_sigma_max,
        tor_sigma_min=config.diffusion.tor_sigma_min, tor_sigma_max=config.diffusion.tor_sigma_max)
    transform_fn = gnn4s.data.NoiseTransform(t_to_sigma=t_to_sigma,
        no_torsion=config.diffusion.no_torsion, all_atom=config.data.all_atom)
    train_dataset = gnn4s.data.PDBDataset(
        dirt=config.data.dirt, load_fn=load_fn, process_fn=process_fn,
        transform=transform_fn, cache_path=config.data.cache_dirt, split_path=config.data.split_dirt,
        esm_embd_path=config.data.esm_embd_path, complex_limit=config.data.complex_limit, 
        worker_num=config.data.worker_num, 
        matching=(not config.diffusion.no_torsion), keep_original=True, 
        conformer_num=config.data.conformer_num, remove_hs=config.data.remove_hs, 
        rec_cutoff_radius=config.data.rec_radius, ca_nb_num_max=config.data.ca_nb_num_max,
        all_atom=config.data.all_atom, atom_cutoff_radius=config.data.atom_radius, 
        atom_nb_num_max=config.data.atom_nb_num_max)

    # model
    timestep_embd = gnn4s.model.TimeStepEmbedding(
        config.model.embd_type, config.model.sigma_embd_dim, config.model.embd_scale)
    lm_embd_type = 'esm' if config.data.esm_embd_path is not None else None
    score_model = gnn4s.model.TensorProductModel(
        t_to_sigma=t_to_sigma, timestep_embd_func=timestep_embd, ns=config.model.ns, nv=config.model.nv, 
        sigma_embd_dim=config.model.sigma_embd_dim, dist_embd_dim=config.model.dist_embd_dim,
        cro_dist_embd_dim=config.model.cro_dist_embd_dim, lm_embd_type=lm_embd_type,
        lig_radius_max=config.model.max_radius, cro_dist_max=config.model.cro_dist_max, 
        dropout_rate=config.model.dropout_rate, dynamic_cro_max=config.model.dynamic_cro_max, 
        use_second_order_repr=config.model.use_second_order_repr,
        use_batch_norm=(not config.model.no_batch_norm), conv_layer_num=config.model.conv_layer_num,
        no_torsion=config.diffusion.no_torsion, scale_by_sigma=config.model.scale_by_sigma,
        conf_mode=False, device=config.device)
    score_model = gnn4s.model.to_data_parallel(score_model, config.device, config.no_parallel)
    gnn4s.model.count_parameters_number(score_model)

    # loss
    loss = gnn4s.train.LossScoreProduct(score_model, t_to_sigma,
        tra_weight=config.diffusion.tr_weight, rot_weight=config.diffusion.rot_weight,
        tor_weight=config.diffusion.tor_weight, no_torsion=config.diffusion.no_torsion)
    
    # train
    optimizer = gnn4s.train.AdamOptimizer(score_model.parameters(), lr=config.train.lr)
    scheduler = gnn4s.train.ExponentialLR(optimizer, gamma=0.999)
    trainer = gnn4s.train.TrainerDiffDock(train_dataset, config.train.batch_size,
        config.data.worker_num, config.data.pin_memory,
        score_model, loss, optimizer, scheduler, config.train.epoch_num, 
        use_ema=config.train.use_ema, device=config.device, logger=logger)
    trainer.train()

if __name__ == '__main__':
    main()