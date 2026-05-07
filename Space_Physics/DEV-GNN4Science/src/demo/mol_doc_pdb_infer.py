import gnn4s

def diffdock_inference(data_path, output_dirt, inference_sample_size):
    # configuration
    config = gnn4s.load_config('./config/mol_doc_pdb_infer.yaml')
    
    # logger
    logger = gnn4s.utils.Logger(config.log.dirt)
    logger.log_config(config)

    # load checkpoint
    ckpt_dict = gnn4s.utils.load_ckpt(f'{config.ckpt.dirt}/{config.ckpt.name}.pth')
    ckpt_config = ckpt_dict['config']
    logger.log_config(vars(ckpt_config))

    # dataset and dataloader
    rec_path_list, lig_description = gnn4s.data.load_rec_lig_csv(data_path)
    load_fn = gnn4s.data.load_lm_embeddings
    process_fn = gnn4s.data.process_complex
    test_dataset = gnn4s.data.PDBDataset(config.data.dirt, load_fn=load_fn, process_fn=process_fn,
        cache_path=config.data.cache_dirt, esm_embd_path=config.data.esm_embd_dirt, 
        worker_num=ckpt_config.data.worker_num, lig_description=lig_description, require_lig=True,
        matching=False, keep_original=False, remove_hs=ckpt_config.data.remove_hs, 
        keep_local_structure=config.data.keep_local_structure,
        rec_path_list=rec_path_list, rec_cutoff_radius=ckpt_config.data.rec_radius,
        ca_nb_num_max=ckpt_config.data.ca_nb_num_max,
        all_atom=ckpt_config.data.all_atom_score, atom_cutoff_radius=ckpt_config.data.atom_radius,
        atom_nb_num_max=ckpt_config.data.atom_nb_num_max)
    test_loader = gnn4s.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if 'confi_model_state_dict' in ckpt_dict:
        confi_test_dataset = gnn4s.data.PDBDataset(
            dirt=config.data.dirt, load_fn=load_fn, process_fn=process_fn,
            cache_path=config.data.cache_dirt, esm_embd_path=config.data.esm_embd_dirt,
            worker_num=ckpt_config.data.worker_num, lig_description=lig_description, require_lig=True,
            matching=False, keep_original=False, remove_hs=ckpt_config.data.remove_hs,
            rec_path_list=rec_path_list, rec_cutoff_radius=ckpt_config.data.rec_radius,
            ca_nb_num_max=ckpt_config.data.ca_nb_num_max,
            all_atom=ckpt_config.data.all_atom_confi, atom_cutoff_radius=ckpt_config.data.atom_radius,
            atom_nb_num_max=ckpt_config.data.atom_nb_num_max)

    # score model
    csm = ckpt_config.score_model
    t_to_sigma = gnn4s.data.TimeToSigma(
        ckpt_config.diffusion.tr_sigma_min, ckpt_config.diffusion.tr_sigma_max,
        ckpt_config.diffusion.rot_sigma_min, ckpt_config.diffusion.rot_sigma_max,
        ckpt_config.diffusion.tor_sigma_min, ckpt_config.diffusion.tor_sigma_max)
    timestep_embd = gnn4s.model.TimeStepEmbedding(csm.embd_type, csm.sigma_embd_dim, csm.embd_scale)
    score_model = gnn4s.model.TensorProductModel(
        t_to_sigma=t_to_sigma, timestep_embd_func=timestep_embd,
        ns=csm.ns, nv=csm.nv, sigma_embd_dim=csm.sigma_embd_dim,
        dist_embd_dim=csm.dist_embd_dim, cro_dist_embd_dim=csm.cro_dist_embd_dim, 
        lm_embd_type=csm.lm_embd_type, lig_radius_max=csm.max_radius,
        cro_dist_max=csm.cro_dist_max, dropout_rate=csm.dropout_rate,
        dynamic_cro_max=csm.dynamic_cro_max, use_second_order_repr=csm.use_second_order_repr,
        use_batch_norm=(not csm.no_batch_norm), conv_layer_num=csm.conv_layer_num,
        no_torsion=ckpt_config.diffusion.no_torsion, scale_by_sigma=csm.scale_by_sigma, 
        conf_mode=False, device=config.device)
    score_model = gnn4s.model.load_state_dict(score_model, ckpt_dict['score_model_state_dict'], 
        'eval', config.device, config.no_parallel)
    
    # confidence model
    if 'confi_model_state_dict' in ckpt_dict:
        ccm = ckpt_config.confi_model
        timestep_embd = gnn4s.model.TimeStepEmbedding(ccm.embd_type, ccm.sigma_embd_dim, ccm.embd_scale)
        confi_output_num = len(ccm.rmsd_class_cutoff) + 1 if (
            hasattr(ccm, 'rmsd_class_cutoff') and isinstance(ccm.rmsd_class_cutoff, list)) else 1
        confi_model = gnn4s.model.TensorProductModelALL(
            t_to_sigma=t_to_sigma, timestep_embd_func=timestep_embd,
            ns=ccm.ns, nv=ccm.nv, sigma_embd_dim=ccm.sigma_embd_dim, dist_embd_dim=ccm.dist_embd_dim,
            cro_dist_embd_dim=ccm.cro_dist_embd_dim, lm_embd_type=ccm.lm_embd_type, 
            lig_radius_max=ccm.max_radius, cro_dist_max=ccm.cro_dist_max,
            dropout_rate=ccm.dropout_rate, dynamic_cro_max=ccm.dynamic_cro_max, 
            use_second_order_repr=ccm.use_second_order_repr, use_batch_norm=(not ccm.no_batch_norm),
            conv_layer_num=ccm.conv_layer_num, no_torsion=ckpt_config.diffusion.no_torsion,
            scale_by_sigma=ccm.scale_by_sigma, conf_mode=True, 
            conf_output_num=confi_output_num, device=config.device)
        confi_model = gnn4s.model.load_state_dict(confi_model, 
            ckpt_dict['confi_model_state_dict'], 'eval', config.device, config.no_parallel)
    else:
        confi_model = None

    # sampler
    transform = gnn4s.data.NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=False, all_atom=False)
    sampler = gnn4s.diffusion.Sampler(t_to_sigma=t_to_sigma, transform=transform, 
        score_model=score_model, confi_model=confi_model, no_torsion=ckpt_config.diffusion.no_torsion,
        score_all_atom=ckpt_config.data.all_atom_score, confi_all_atom=ckpt_config.data.all_atom_confi,
        tr_sigma_min=ckpt_config.diffusion.tr_sigma_min, tr_sigma_max=ckpt_config.diffusion.tr_sigma_max,
        rot_sigma_min=ckpt_config.diffusion.rot_sigma_min, rot_sigma_max=ckpt_config.diffusion.rot_sigma_max,
        tor_sigma_min=ckpt_config.diffusion.tor_sigma_min, tor_sigma_max=ckpt_config.diffusion.tor_sigma_max,
        no_random=ckpt_config.diffusion.no_random, use_ode=ckpt_config.diffusion.use_ode, 
        no_final_step_noise=ckpt_config.diffusion.no_final_step_noise, device=config.device)

    generator = gnn4s.diffusion.GeneratorDiffDock(test_loader, confi_test_dataset, confi_model, sampler,
        config.inference.step_num, config.data.sample_per_complex, ccm.rmsd_class_cutoff, logger)
    generator.generate(inference_sample_size, output_dirt)

def main():
    # path for storing dataset
    data_path = './dataset/pdbbind/inference/protein_ligand.csv'
    # directory for storing output
    output_dirt = './dataset/pdbbind/output'
    # number of generated ligand
    inference_sample_size = 6
    
    diffdock_inference(data_path, output_dirt, inference_sample_size)

if __name__ == "__main__":
    main()