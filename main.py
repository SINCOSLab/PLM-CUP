import torch
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
from colorama import init, Fore, Style
from utils.metrics import metric, MAE_torch, MAPE_torch, RMSE_torch, WMAPE_torch
from utils.data_loader import load_dataset
from utils.seed import seed_it
from utils.trainer import trainer
from utils.config import get_args
import importlib
import shutil
import hashlib
import base64

PEMS_NODES = [358, 883, 170, 307]


def extract_source_domain(pretrained_model_path):
    """Extract source domain information from pretrained model path"""
    if not pretrained_model_path:
        return None

    path_parts = pretrained_model_path.split("/")
    for i, part in enumerate(path_parts):
        if part in ["logs", "finish", "running"]:
            continue
        if any(
            dataset in part.upper()
            for dataset in ["PEMS", "CHBike", "NYCTaxi", "HK", "NJ", "SH"]
        ):
            return part

    if "logs" in path_parts:
        logs_idx = path_parts.index("logs")
        if logs_idx + 1 < len(path_parts):
            return path_parts[logs_idx + 1]

    return None


def encode_prefix(
    data_name,
    model,
    gpt_layers,
    U,
    use_lora,
    current_time,
    source_domain=None,
    pretrain_model=None,
):
    if source_domain:
        if pretrain_model:
            original = f"{data_name}_{model}_{pretrain_model}_{gpt_layers}_{U}_{use_lora}_{source_domain}_{current_time}"
        else:
            original = f"{data_name}_{model}_{gpt_layers}_{U}_{use_lora}_{source_domain}_{current_time}"
    else:
        if pretrain_model:
            original = f"{data_name}_{model}_{pretrain_model}_{gpt_layers}_{U}_{use_lora}_{current_time}"
        else:
            original = f"{data_name}_{model}_{gpt_layers}_{U}_{use_lora}_{current_time}"

    hash_obj = hashlib.md5(original.encode())
    encoded = base64.b64encode(hash_obj.digest()).decode("utf-8")
    encoded = encoded[:12].replace("+", "p").replace("/", "s")

    return encoded, original


def main(retry_count=0):
    init()
    args = get_args()
    seed_it(args.seed)
    dataloader = load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size,
        args.train_ratio,
    )
    scaler = dataloader["scaler"]

    if os.path.isdir(args.data):
        data_name = os.path.basename(os.path.normpath(args.data))
        parent_dir = os.path.basename(os.path.dirname(args.data))

        data_mapping = {
            ("CHBike", "pick"): ("CHBike_pick", 250),
            ("CHBike", "drop"): ("CHBike_drop", 250),
            ("NYCTaxi", "pick"): ("NYCTaxi_pick", 266),
            ("NYCTaxi", "drop"): ("NYCTaxi_drop", 266),
            ("bike_pick",): ("CHBike_pick", 250),
            ("bike_drop",): ("CHBike_drop", 250),
            ("taxi_pick",): ("NYCTaxi_pick", 266),
            ("taxi_drop",): ("NYCTaxi_drop", 266),
            ("PEMS03",): ("PEMS03", 358),
            ("PEMS04",): ("PEMS04", 307),
            ("PEMS07",): ("PEMS07", 883),
            ("PEMS08",): ("PEMS08", 170),
            ("NJ",): ("NJ", 225),
            ("HK",): ("HK", 225),
            ("SH",): ("SH", 225),
        }

        key = (
            (parent_dir, data_name)
            if parent_dir in ["CHBike", "NYCTaxi"]
            else (data_name,)
        )
        if key in data_mapping:
            data_name, args.num_nodes = data_mapping[key]

        # Handle the difference between QY and QY-n
        # If parent_dir is QY, add -1 suffix to data_name
        # If parent_dir is QY-n, add -n suffix to data_name
        if parent_dir.startswith("QY"):
            original_data_name = data_name
            if parent_dir == "QY":
                # QY -> xx-1
                data_name = f"{data_name}-1"
            elif "-" in parent_dir:
                # QY-n -> xx-n
                suffix = parent_dir.split("-")[1]
                data_name = f"{data_name}-{suffix}"
            print(
                f"{Fore.CYAN}Dataset naming: {parent_dir}/{original_data_name} -> {data_name}{Style.RESET_ALL}"
            )

    device = torch.device(args.device)
    current_time = time.strftime("%m-%d-%H-%M-%S")  # Remove year

    # If transfer learning, extract source domain information
    source_domain = None
    if args.is_transfer and args.pretrained_model_path:
        source_domain = extract_source_domain(args.pretrained_model_path)

    encoded_prefix, original_prefix = encode_prefix(
        data_name,
        args.model,
        args.gpt_layers,
        args.U,
        args.use_lora,
        current_time,
        source_domain,
        args.pretrain_model,
    )
    file_prefix = f"_{encoded_prefix}"
    save_dir = os.path.join(os.path.dirname(__file__), args.save)
    model_path = os.path.join(save_dir, data_name, args.model)

    if source_domain:
        timestamp_folder = f"{current_time}_from_{source_domain}"
    else:
        timestamp_folder = current_time

    # Create path including pretrained model
    pretrain_model_clean = args.pretrain_model.replace("-", "_").replace(".", "_")

    # Convert True/False to use_lora/not_use_lora
    lora_status = "use_lora" if str(args.use_lora).lower() == "true" else "not_use_lora"

    running_path = os.path.join(
        model_path,
        "running",
        f"{pretrain_model_clean}",
        lora_status,
        f"{args.gpt_layers}_{args.U}",
        timestamp_folder,
    )
    finish_path = os.path.join(
        model_path,
        "finish",
        f"{pretrain_model_clean}",
        lora_status,
        f"{args.gpt_layers}_{args.U}",
        timestamp_folder,
    )
    code_path = os.path.join(running_path, "code")
    if not os.path.exists(running_path):
        os.makedirs(running_path, exist_ok=True)
    if not os.path.exists(code_path):
        os.makedirs(code_path, exist_ok=True)

    # Create model first to avoid double loading
    try:
        model_module = importlib.import_module("models." + args.model)
        model_class = getattr(model_module, args.model)

        model_kwargs = {
            "device": device,
            "input_dim": args.input_dim,
            "channels": args.channels,
            "num_nodes": args.num_nodes,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "dropout": args.dropout,
            "gpt_path": args.gpt_path,
            "pretrain_model": args.pretrain_model,
            "gpt_layers": args.gpt_layers,
            "U": args.U,
            "small_value_threshold": 1.0,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }

        if args.is_transfer:
            model_kwargs.update(
                {
                    "is_transfer": args.is_transfer,
                    "source_nodes": args.source_nodes,
                    "pretrained_model_path": args.pretrained_model_path,
                }
            )

        model = model_class(**model_kwargs)
    except ImportError:
        raise ImportError(
            f"{Fore.RED}Model {args.model} not found in models directory{Style.RESET_ALL}"
        )
    except AttributeError:
        raise AttributeError(
            f"{Fore.RED}Model {args.model} not found in {args.model}.py{Style.RESET_ALL}"
        )

    # Write configuration using the already created model
    config_path = os.path.join(running_path, "config.txt")
    with open(config_path, "w") as f:
        f.write("Configuration Parameters:\n")
        f.write("=" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("=" * 50 + "\n")

        f.write("\nRuntime Environment:\n")
        f.write("=" * 50 + "\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Device Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Device Count: {torch.cuda.device_count()}\n")
        f.write(f"Start Time: {current_time}\n")
        f.write(f"Pretrained Model: {args.pretrain_model}\n")
        f.write(
            f"GPT Layers: {args.gpt_layers} {'(all layers)' if args.gpt_layers == -1 else ''}\n"
        )
        if retry_count > 0:
            f.write(f"\nRetry Information:\n")
            f.write(f"Retry Count: {retry_count}\n")
            f.write(f"Original Learning Rate: {args.learning_rate}\n")
            f.write(
                f"Actual Learning Rate: {args.learning_rate * (0.5**retry_count)}\n"
            )
            f.write(f"Original Min Learning Rate: {args.min_learning_rate}\n")
            f.write(
                f"Actual Min Learning Rate: {args.min_learning_rate * (0.5**retry_count)}\n"
            )
        f.write("=" * 50 + "\n")

        try:
            f.write("\nNetwork Structure:\n")
            f.write("=" * 50 + "\n")
            model_str = str(model)
            formatted_model_str = "\n".join(
                "    " + line for line in model_str.split("\n")
            )
            f.write(formatted_model_str)
            f.write("\n")
            f.write("=" * 50 + "\n")

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            f.write("\nModel Parameters:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write("=" * 50 + "\n")

        except Exception as e:
            f.write("\nError getting model structure:\n")
            f.write(str(e) + "\n")
            f.write("=" * 50 + "\n")

        f.write("\nPrefix Mapping:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Encoded Prefix: {encoded_prefix}\n")
        f.write(f"Original Prefix: {original_prefix}\n")
        f.write("=" * 50 + "\n")

    loss = 9999999
    epochs_since_best_mae = 0
    his_loss = []
    val_time = []
    train_time = []
    result = []
    best_model_state = None  # Store best model state in memory

    # Create trainer

    # Calculate actual learning rates
    actual_lr = args.learning_rate * (0.5**retry_count)
    actual_min_lr = args.min_learning_rate * (0.5**retry_count)

    engine = trainer(
        scaler,
        actual_lr,
        args.weight_decay,
        args.lr_decay,
        args.lr_decay_patience,
        actual_min_lr,
        model,
        device,
        is_pems=args.num_nodes in PEMS_NODES,
        use_domain_shift_regularization=True,
        domain_shift_config={
            "method": "simple_cosine",
            "phi": 0.01,
            "detach_features": False,
        },
    )

    if args.load_model is not None and os.path.exists(args.load_model):
        try:
            engine.model.load_state_dict(torch.load(args.load_model))
        except Exception as e:
            pass

    # Start training

    for i in range(1, args.epochs + 1):
        engine.set_epoch(i)
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        train_ds = []

        t1 = time.time()

        train_progress = tqdm(
            dataloader["train_loader"],
            total=len(dataloader["train_loader"]),
            desc=f"Epoch {i}/{args.epochs}",
        )
        for batch_x, batch_y in train_progress:
            trainx = batch_x.to(device)
            trainx = trainx.transpose(1, 3)
            trainy = batch_y.to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            if any(np.isnan(m) for m in metrics):
                print(f"{Fore.RED}NaN detected in metrics: {metrics}{Style.RESET_ALL}")
                if args.epochs // 2 < i:
                    return True
                return False
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            if len(metrics) > 4:
                ds_value = metrics[4]
                train_ds.append(ds_value)
            else:
                ds_value = 0.0
                train_ds.append(ds_value)

            train_progress.set_postfix(
                {
                    "Loss": f"{train_loss[-1]:.4f}",
                    "RMSE": f"{train_rmse[-1]:.4f}",
                    "MAPE": f"{train_mape[-1]:.4f}",
                    "WMAPE": f"{train_wmape[-1]:.4f}",
                    "DS": f"{ds_value:.4f}",
                }
            )
        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()

        val_progress = tqdm(
            dataloader["val_loader"],
            total=len(dataloader["val_loader"]),
            desc=f"Validation {i}/{args.epochs}",
        )
        for batch_x, batch_y in val_progress:
            testx = batch_x.to(device)
            testx = testx.transpose(1, 3)
            testy = batch_y.to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

            val_progress.set_postfix(
                {
                    "Loss": f"{np.mean(valid_loss):.4f}",
                    "RMSE": f"{np.mean(valid_rmse):.4f}",
                    "MAPE": f"{np.mean(valid_mape):.4f}",
                    "WMAPE": f"{np.mean(valid_wmape):.4f}",
                }
            )

        s2 = time.time()

        val_time.append(s2 - s1)

        engine.scheduler.step(np.mean(valid_loss))

        current_lr = engine.optimizer.param_groups[0]["lr"]

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            epoch=int(i),
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            train_ds=np.mean(train_ds),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        mtrain_ds = np.mean(train_ds)

        if mvalid_loss < loss:
            loss = mvalid_loss
            best_model_state = engine.model.state_dict().copy()
            bestid = i
            epochs_since_best_mae = 0
        else:
            epochs_since_best_mae += 1

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{running_path}/train{file_prefix}.csv", index=False)

        if epochs_since_best_mae >= args.es_patience and i >= 80:
            break

    # Training completed
    print(
        f"Training completed. Best epoch: {bestid}, Valid loss: {round(his_loss[bestid - 1], 4)}"
    )

    # Test on best model
    if best_model_state is not None:
        engine.model.load_state_dict(best_model_state)

    outputs = []
    realy = dataloader["y_test"].to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for x, y in dataloader["test_loader"]:
        testx = x.to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []
    test_result = []

    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)

        test_m = dict(
            horizon=i + 1,
            test_mae=metrics[0],
            test_rmse=metrics[2],
            test_mape=metrics[1],
            test_wmape=metrics[3],
        )
        test_result.append(pd.Series(test_m))

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    test_m = dict(
        horizon="avg",
        test_mae=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_result.append(pd.Series(test_m))

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{running_path}/test_results.csv", index=False)

    print(
        f"Test MAE: {np.mean(amae):.4f}, RMSE: {np.mean(armse):.4f}, MAPE: {np.mean(amape):.4f}, WMAPE: {np.mean(awmape):.4f}"
    )

    # Save best model
    if best_model_state is not None:
        torch.save(
            best_model_state,
            os.path.join(running_path, "best_model.pth"),
        )

    # Save domain shift log
    ds_log_path = os.path.join(running_path, "domain_shift_log.json")
    engine.save_ds_log(ds_log_path)

    if not args.is_transfer:
        # Load best model state before saving components
        if best_model_state is not None:
            engine.model.load_state_dict(best_model_state)

        engine.model.save_component_A(
            os.path.join(running_path, f"best_epoch_{bestid}_component_A.pth")
        )
        engine.model.save_component_B(
            os.path.join(running_path, f"best_epoch_{bestid}_component_B.pth")
        )

    if not os.path.exists(finish_path):
        os.makedirs(finish_path, exist_ok=True)

    for file in os.listdir(running_path):
        src = os.path.join(running_path, file)
        dst = os.path.join(finish_path, file)
        shutil.move(src, dst)

    os.rmdir(running_path)

    return True


if __name__ == "__main__":
    MAX_RETRIES = 3
    retry_count = 0
    while retry_count < MAX_RETRIES:
        t1 = time.time()
        result = main(retry_count=retry_count)
        t2 = time.time()
        if result:
            print("Total time spent: {:.4f}".format(t2 - t1))
            break
        else:
            retry_count += 1
            print(f"Retrying... ({retry_count}/{MAX_RETRIES})")
