import os.path

import torch
import torchvision
import ultralytics
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from pathlib import Path
import yaml
# from ultralytics.data.build import build_dataloader
from codebase.models import AutoEncoder
from codebase.utils import visualize_keypoints_predicted, visualize_keypoints_gt, dotdict, build_dataloader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from ultralytics.utils import ASSETS
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import ASSETS
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
import wandb
import torchvision
from ultralytics.models.yolo.pose import PosePredictor
import argparse
import torchvision.transforms as transforms
import numpy as np
from piq import ssim, psnr, vif_p
import random
import pickle

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/project_ghent/Ergo/configs/keypoints_final2.yaml")
    parser.add_argument("--pose_model", type=str, default="yolov8x-pose.pt",
                        help='which ultralytics pose model to use, we recommend 8 as 11 seems more unstable (not all weights can be loaded)')
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size_train", type=int, default=8)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument("--learning_rate_obfuscator", type=float, default=0.001)
    parser.add_argument("--learning_rate_deobfuscator", type=float, default=0.001)
    parser.add_argument("--save_path_obfuscator", type=str, default="obfuscator.pth")
    parser.add_argument("--save_path_deobfuscator", type=str, default="deobfuscator.pth")

    # obfuscator architecture hyperparameters
    parser.add_argument("--internal_expansion", type=int, default=6)
    parser.add_argument("--first_n_filters", type=int, default=32)
    parser.add_argument("--upscale_factor", nargs=3, type=int, default=[2, 2, 4])
    parser.add_argument('--random_mapping', action='store_true', default=False,
                        help='Use to use the random map variant of the obfuscator')

    parser.add_argument("--reconstruction_weight", type=float, default=20, help='Weight of the reconstruction factor in the loss function')
    parser.add_argument('--weight_decay_obfuscator', type=float, default=0.01)
    parser.add_argument('--weight_decay_deobfuscator', type=float, default=0.01)

    # use when starting from a trained model
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Use to finetune an existing obfuscator and deobfuscator, be sure to specify them in load_path_(de)obfuscator')
    parser.add_argument('--load_path_obfuscator', type=str, default='models/obfuscator/rand_map_coco.pt')
    parser.add_argument('--load_path_deobfuscator', type=str, default='models/deobfuscator/rand_map_coco.pt')

    parser.add_argument('--augment_train_data', action='store_true', default=False,
                        help='Wether to use the yolo data augmentations for training models')
    parser.add_argument('--nowandb', action='store_true', default=False,
                        help="Use when you don't want to log to weights and biases")
    parser.add_argument('--wandb_project', default='privacyErgo', help='Wandb project name')

    args = parser.parse_args()
    print(args)

    normalize = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.nowandb else 'online')

    # load in YOLO model
    model = YOLO(args.pose_model.replace("pt", "yaml")).load(args.pose_model)
    # model = YOLO(args.pose_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device).train()
    # args = dict(model='args.pose_model', source=ASSETS)

    # load in dataset
    conf = yaml.safe_load(Path(args.config_path).read_text())
    images_path_train = os.path.join(conf["path"], conf["train"])
    images_path_val = os.path.join(conf["path"], conf["val"])
    dataset_train = YOLODataset(img_path=Path(images_path_train), data=conf, task="pose",
                                augment=args.augment_train_data)
    dataset_train_eval = YOLODataset(img_path=Path(images_path_train), data=conf, task="pose", augment=False)
    dataset_val = YOLODataset(img_path=Path(images_path_val), data=conf, task="pose", augment=False)

    # use a datasampler
    sampler = None
    dataloader = build_dataloader(dataset_train, batch=args.batch_size_train, workers=2, sampler=sampler, shuffle=True)
    dataloader_val = build_dataloader(dataset_val, batch=args.batch_size_val, workers=2, shuffle=True)
    dataloader_train_eval = build_dataloader(dataset_train_eval, batch=args.batch_size_val, workers=2, shuffle=True)

    # load in obfuscator and deobfuscator model
    obfuscator = AutoEncoder(internal_expansion=args.internal_expansion, first_n_filters=args.first_n_filters,
                             upscale_factor=args.upscale_factor, random_map=args.random_mapping)
    deobfuscator = AutoEncoder(internal_expansion=6, first_n_filters=24, upscale_factor=[2, 2, 2])
    obfuscator = obfuscator.to(device)
    deobfuscator = deobfuscator.to(device)

    if args.finetune:
        obfuscator.load_state_dict(torch.load(args.load_path_obfuscator))
        deobfuscator.load_state_dict(torch.load(args.load_path_deobfuscator))

    # init validation model
    val_model = YOLO(args.pose_model)
    val_model.model.to(device).eval()
    args_valid = dict(model=args.pose_model, source=ASSETS, data=args.config_path, task="pose")
    validator = PoseValidator(args=args_valid, dataloader=dataloader_val)
    validator.data = check_det_dataset(validator.args.data)
    validator.device = device

    train_model = YOLO(args.pose_model)
    train_model.model.to(device).eval()
    train_validator = PoseValidator(args=args_valid, dataloader=dataloader_train_eval)
    train_validator.data = check_det_dataset(train_validator.args.data)
    train_validator.device = device

    pose_loss = ultralytics.utils.loss.v8PoseLoss(model.model)
    pose_loss.hyp = dotdict(pose_loss.hyp)
    reconstruction_criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(obfuscator.parameters(), lr=args.learning_rate_obfuscator, eps=1e-4,
                                  weight_decay=args.weight_decay_obfuscator)
    optimizer_deobfuscator = torch.optim.AdamW(deobfuscator.parameters(), lr=args.learning_rate_deobfuscator, eps=1e-4,
                                               weight_decay=args.weight_decay_deobfuscator)

    scheduler_obfuscator = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)
    scheduler_deobfuscator = torch.optim.lr_scheduler.StepLR(
        optimizer_deobfuscator, step_size=10, gamma=0.1)

    # disable gradients for the pose estimation model
    for param in model.model.parameters():
        param.requires_grad = False

    scaler = GradScaler()

    for epoch in range(args.num_epochs):
        # train obfuscator
        obfuscator = obfuscator.train()
        deobfuscator = deobfuscator.train()

        train_validator.init_metrics(model)
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            # load in data, move to device
            data["img"] = normalize(data["img"].to(device, non_blocking=True).float() / 255)
            for key in data.keys():
                if type(data[key]) == torch.Tensor:
                    data[key] = data[key].to(device)

            optimizer.zero_grad()
            optimizer_deobfuscator.zero_grad()

            with autocast(enabled=True):
                # obfuscate image
                obfuscated = obfuscator(data["img"])

                reg_loss = 0
                if args.random_mapping:
                    obfuscated, rand_map = obfuscated[0], obfuscated[1]
                    reg_loss = torch.mean(torch.abs(rand_map))

                # deobfuscate image
                reconstructed = deobfuscator(obfuscated)

                if epoch % 2 == 0:
                    # calculate loss
                    pose_preds = model.model(obfuscated)
                    loss = pose_loss(pose_preds, data)
                    # print(loss.sum())

                reconstruction_loss = reconstruction_criterion(reconstructed,
                                                               data["img"]) * args.reconstruction_weight + reg_loss
                deobfuscator_loss = reconstruction_loss

            if epoch % 2 == 0:
                # train the obfuscator
                obfuscator_loss = loss[0].sum() - reconstruction_loss
                print(obfuscator_loss)
                scaler.scale(obfuscator_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # train the deobfuscator
                scaler.scale(deobfuscator_loss).backward()
                scaler.step(optimizer_deobfuscator)
                scaler.update()

            if i % 10 == 0:
                wandb.log({"loss_pose": loss[0].sum().item(), "loss_reconstruction": reconstruction_loss.item(),
                           "obfuscator_loss": obfuscator_loss.item(), "deobfuscator_loss": deobfuscator_loss.item()})
        # advance schedulers
        scheduler_obfuscator.step()
        scheduler_deobfuscator.step()

        # log metrics
        if epoch % 2 == 0:
            # do a run for train accuracy
            train_validator.init_metrics(train_model)
            train_model.model.eval()
            for i, data in tqdm(enumerate(dataloader_train_eval), total=len(dataloader_train_eval)):
                data["img"] = normalize(data["img"].to(device, non_blocking=True).float() / 255)
                with torch.no_grad():
                    obfuscated = obfuscator(data["img"])

                if args.random_mapping:
                    obfuscated, rand_map = obfuscated[0], obfuscated[1]

                data['img'] = obfuscated.clone() * 255
                data = train_validator.preprocess(data)

                with torch.no_grad():
                    preds = train_model.model(data['img'])
                preds = train_validator.postprocess(preds)
                for key in data.keys():
                    if type(data[key]) == torch.Tensor:
                        data[key] = data[key].to(device)

                train_validator.update_metrics(preds, data)

            stats = train_validator.get_stats()
            # prepend "train_" to the keys
            stats = {f"train_{key}": value for key, value in stats.items()}
            wandb.log(stats)

            # validation run
            validator.init_metrics(val_model)

            avg_ssim = 0
            avg_psnr = 0
            avg_vif = 0

            for i, data in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
                original = data['img'].clone().to(device)
                data['img'] = normalize(data['img'].to(device) / 255)
                with torch.no_grad():
                    obfuscated = obfuscator(data["img"])
                # now we should save the obfuscated images
                if args.random_mapping:
                    obfuscated, rand_map = obfuscated[0], obfuscated[1]

                # update iq metrics

                avg_ssim += ssim(original / 255, obfuscated)
                avg_psnr += psnr(original / 255, obfuscated)
                avg_vif += vif_p(original / 255, obfuscated)

                data['img'] = obfuscated.clone() * 255
                data = validator.preprocess(data)

                preds = val_model.model(data['img'])
                preds = validator.postprocess(preds)
                for key in data.keys():
                    if type(data[key]) == torch.Tensor:
                        data[key] = data[key].to(device)

                validator.update_metrics(preds, data)

            stats = validator.get_stats()
            wandb.log(stats)
            images_normal = wandb.Image(torchvision.utils.make_grid(original))
            images_obfuscated = wandb.Image(torchvision.utils.make_grid(obfuscated))
            with torch.no_grad():
                reconstructed = deobfuscator(obfuscated)
            images_deobfuscated = wandb.Image(torchvision.utils.make_grid(reconstructed))
            data['img'] = original
            keypoints_normal = visualize_keypoints_gt(data)
            predictions_obf = val_model(obfuscated)
            keypoints_obfuscated = visualize_keypoints_predicted(obfuscated, predictions_obf)
            keypoints_normal = wandb.Image(torchvision.utils.make_grid(keypoints_normal))
            keypoints_obfuscated = wandb.Image(torchvision.utils.make_grid(keypoints_obfuscated))

            if args.random_mapping:
                print(rand_map.shape)
                print(rand_map.repeat(1, 3, 1, 1).shape)
                maps = wandb.Image(torchvision.utils.make_grid(rand_map.repeat(1, 3, 1, 1)))
                wandb.log({"random_map": maps})
            wandb.log({"normal": images_normal, "obfuscated": images_obfuscated, "deobfuscated": images_deobfuscated,
                       "keypoints_normal": keypoints_normal,
                       "keypoints_obfuscated": keypoints_obfuscated, "ssim": avg_ssim / len(dataloader_val),
                       "psnr": avg_psnr / len(dataloader_val), "vif": avg_vif / len(dataloader_val)
                       })

    torch.save(obfuscator.state_dict(), args.save_path_obfuscator)
    torch.save(deobfuscator.state_dict(), args.save_path_deobfuscator)
