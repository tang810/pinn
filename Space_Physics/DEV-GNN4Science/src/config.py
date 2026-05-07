import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DEV-GNN4Science standardized runner")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "train"])
    parser.add_argument("--data", type=str, default="simul", choices=["simul", "real"])
    parser.add_argument("--data_path", type=str, default="data/simul_gnn_graph.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument(
        "--task",
        type=str,
        default="cylinder_flow",
        choices=[
            "airfoil",
            "cardiovascular",
            "cylinder_flow",
            "flag_simple",
            "mol_doc_pdb",
            "mol_gen_qm9",
        ],
    )
    parser.add_argument("--delegate_original", action="store_true")
    return parser.parse_args()

