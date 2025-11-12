import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda device number"
    )
    parser.add_argument("--data", type=str, default="", help="data path")
    parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
    parser.add_argument("--channels", type=int, default=64, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=250, help="number of nodes")
    parser.add_argument("--input_len", type=int, default=6, help="input_len")
    parser.add_argument("--output_len", type=int, default=1, help="out_len")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay rate"
    )
    parser.add_argument("--print_every", type=int, default=50, help="")
    parser.add_argument("--epochs", type=int, default=200, help="100~500")
    parser.add_argument(
        "--es_patience", type=int, default=60, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="learning rate"
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.2, help="learning rate decay"
    )
    parser.add_argument(
        "--lr_decay_patience", type=int, default=30, help="learning rate decay patience"
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=5e-5, help="minimum learning rate"
    )
    parser.add_argument(
        "--gpt_layers", type=int, default=6, help="number of gpt layers"
    )
    parser.add_argument("--U", type=int, default=1, help="U")
    parser.add_argument("--model", type=str, default="", help="model name")
    parser.add_argument(
        "--load_model", type=str, default=None, help="path to pretrained model"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"),
        help="save path",
    )
    parser.add_argument("--gpt_path", type=str, default="", help="pretrained gpt path")
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default="gpt2",
        choices=["gpt2", "qwen3-0.6b"],
        help="which pretrained model to use (gpt2/qwen3-0.6b)",
    )
    parser.add_argument("--use_lora", type=str, default="True", help="use lora")
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora alpha")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--train_ratio",
        type=int,
        default=100,
        help="percentage of training data to use (10, 20, ..., 100)",
    )
    parser.add_argument(
        "--is_transfer",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="whether to use transfer learning",
    )
    parser.add_argument(
        "--source_nodes",
        type=int,
        default=None,
        help="number of nodes in source domain",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="path to pretrained model (without extension)",
    )
    args = parser.parse_args()
    return args
