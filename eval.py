# python script to do evaluation of an obfuscation model
import os
import torch
import torchvision
import ultralytics
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from pathlib import Path
import yaml
from ultralytics.data.build import build_dataloader
from codebase.models import AutoEncoder
from codebase.utils import visualize_keypoints_predicted, visualize_keypoints_gt, dotdict
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
from piq import ssim, psnr, vif_p, LPIPS
from semsim import get_model, calculate_distance

# perceptanon model
from models import PerceptAnonHA1, PerceptAnonHA2

# ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def obfuscate_image(image, type="gaussian_blur", **kwargs):
    obfuscated = None
    if type == 'no_obfuscation':
        return image
    if type == "gaussian_blur":
        obfuscated = torchvision.transforms.functional.gaussian_blur(image, kernel_size=[kwargs["kernel_size"],
                                                                                         kwargs["kernel_size"]])
    elif type == "noise":
        noise = torch.randn_like(image) * kwargs["std"]
        obfuscated = torch.clip(image + noise,0,1)

    elif type == "pixelate":
        im_size = (image.shape[1], image.shape[2]) if len(image.shape) == 3 else (image.shape[2], image.shape[3])
        im_size_reduced = [int(im_size[0] / kwargs["pix_reduction"]), int(im_size[1] / kwargs["pix_reduction"])]
        obfuscated = torchvision.transforms.functional.resize(image, im_size_reduced)
        obfuscated = torchvision.transforms.functional.resize(obfuscated, im_size)

    elif type == "two-step":
        # first use yolo model to detect persons and then blur them
        with torch.no_grad():
            preds = yolo_model(image, verbose=False)
        obfuscated = image.clone()
        
        for i in range(len(preds)):
            for box in preds[i].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())

                # if box is too small, skip
                if x2 - x1 < 30 or y2 - y1 < 30:
                    continue
                obfuscated[i,:,  y1:y2,x1:x2] = torchvision.transforms.functional.gaussian_blur(image[i,:,  y1:y2,x1:x2],
                                                                                               kernel_size=[kwargs["kernel_size"],
                                                                                                            kwargs["kernel_size"]])
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    #norm_ip(obfuscated, float(obfuscated.min()), float(obfuscated.max()))
    return torch.clip(obfuscated, 0,1)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/project_ghent/Ergo/configs/keypoints_final2.yaml")
    parser.add_argument("--pose_model", type=str, default="yolo11x-pose.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--internal_expansion", type=int, default=6)
    parser.add_argument("--first_n_filters", type=int, default=32)
    parser.add_argument("--upscale_factor", nargs=3, type=int, default=[2, 2, 4])
    parser.add_argument('--load_path_obfuscator', type=str, default='models/obfuscator/scratch_norand.pt')
    parser.add_argument('--load_path_deobfuscator', type=str, default='models/deobfuscator/scratch_norand_large.pt')
    parser.add_argument('--random_mapping', action='store_true', default=False)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--obfuscation_type', type=str, default="obfuscator")
    parser.add_argument('--pix_reduction', type=int, default=2)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--kernel_size', type=int, default=21)
    parser.add_argument('--noise_type', type=str, default='uniform')

    args = parser.parse_args()

    log_image_quality = True
    # load config
    wandb.init(project="eval_ergo", config=args, mode='disabled' if not args.wandb else 'online')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    yolo_model = YOLO("yolo11n.pt", verbose=False)
    yolo_model.model.to(device).eval()
    
    # load in dataset
    conf = yaml.safe_load(Path(args.config_path).read_text())
    images_path = os.path.join(conf["path"], conf["test"])
    dataset = YOLODataset(img_path=Path(images_path), data=conf, task="pose", augment=False)
    dataloader = build_dataloader(dataset, batch=args.batch_size, workers=1, shuffle=False)

    if args.obfuscation_type == "obfuscator":
        # load in obfuscator and deobfuscator model
        obfuscator = AutoEncoder(internal_expansion=args.internal_expansion, first_n_filters=args.first_n_filters,
                                 upscale_factor=args.upscale_factor, random_map=args.random_mapping, noise_type=args.noise_type)
        deobfuscator = AutoEncoder(internal_expansion=6, first_n_filters=24, upscale_factor=[2, 2, 2])
        obfuscator.load_state_dict(torch.load(args.load_path_obfuscator))
        deobfuscator.load_state_dict(torch.load(args.load_path_deobfuscator))
        obfuscator = obfuscator.to(device).eval()
        deobfuscator = deobfuscator.to(device).eval()

    # load in YOLO model
    val_model = YOLO(args.pose_model)
    val_model.model.to(device).eval()
    args_valid = dict(model=args.pose_model, source=ASSETS, data=args.config_path, task="pose")
    validator = PoseValidator(args=args_valid, dataloader=dataloader)
    validator.data = check_det_dataset(validator.args.data)
    validator.device = device
    semsim_model = get_model()
    semsim_model = semsim_model.to(device)

    # loop over test data
    validator.init_metrics(val_model)

    avg_ssim = 0
    avg_psnr = 0
    avg_vif = 0
    avg_lpips = 0
    avg_semsim = 0
    avg_perceptanon_ha1 = 0
    avg_perceptanon_ha2 = 0

    avg_ssim_deobf = 0
    avg_psnr_deobf = 0
    avg_vif_deobf = 0
    avg_lpips_deobf = 0
    avg_semsim_deobf = 0
    avg_perceptanon_deobf_ha1 = 0
    avg_perceptanon_deobf_ha2 = 0


    # perceptanon model
    model_ha1 = PerceptAnonHA1(10, False, True).get_model('resnet50')
    checkpoint = torch.load('/project_ghent/perceptanon/ha1_rn50_clf_labels10.pth.tar', map_location=device)
    model_ha1.load_state_dict(checkpoint['state_dict'])
    model_ha1 = model_ha1.to(device)

    model_ha2 = PerceptAnonHA2('resnet18', 10, False, True)
    checkpoint = torch.load('/project_ghent/perceptanon/results/ckpts/HA2/resnet18/all/labels10.pth.tar',
                            map_location=device)
    model_ha2.load_state_dict(checkpoint['state_dict'])
    model_ha2 = model_ha2.to(device)

    model_ha1.eval()
    model_ha2.eval()

    loss_lpips = LPIPS().to(device)
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        original = data['img'].clone().to(device)

        #data['img'] = normalize(data['img'].to(device) / 255)
        with torch.no_grad():
            if args.obfuscation_type == "obfuscator":
                # first adjust the saturation
                data['img'] = torchvision.transforms.functional.adjust_saturation(data['img'], 1.5)
                data['img'] = normalize(data['img'].to(device) / 255)
                obfuscated = obfuscator(data["img"])
            else:
                data['img'] = data['img'].to(device) / 255
                obfuscated = obfuscate_image(data["img"], type=args.obfuscation_type, **vars(args))

        # now we should save the obfuscated images
        if args.random_mapping:
            obfuscated, rand_map = obfuscated[0], obfuscated[1]

        # update iq metrics
        if log_image_quality:
            avg_ssim += ssim(original / 255, obfuscated)
            avg_psnr += psnr(obfuscated, original / 255)
            avg_vif += vif_p(original / 255, obfuscated)
            avg_lpips += loss_lpips(original / 255, obfuscated)
            #avg_semsim += calculate_distance(semsim_model, original/255, obfuscated)
            with torch.no_grad():
                avg_perceptanon_ha1 += torch.mean(torch.argmax(model_ha1(test_transforms(obfuscated.clone())), axis=1).float())
                avg_perceptanon_ha2 += torch.mean(torch.argmax(model_ha2( test_transforms(original / 255), test_transforms(obfuscated.clone())), axis=1).float())


        data['img'] = obfuscated.clone() #* 255
        #data = validator.preprocess(data)

        with torch.no_grad():
            preds = val_model.model(data['img'])
        preds = validator.postprocess(preds)
        for key in data.keys():
            if type(data[key]) == torch.Tensor:
                data[key] = data[key].to(device)

        validator.update_metrics(preds, data)
        #stats = validator.get_stats()
        #print(stats)

        #del preds, obfuscated, data
        if args.obfuscation_type == "obfuscator" and log_image_quality:
            with torch.no_grad():
                reconstructed = deobfuscator(obfuscated)

            avg_ssim_deobf += ssim(original / 255, reconstructed)
            avg_psnr_deobf += psnr(reconstructed, original / 255)
            avg_vif_deobf += vif_p(original / 255, reconstructed)
            avg_lpips_deobf += loss_lpips(original / 255, reconstructed)
            #avg_semsim_deobf += calculate_distance(semsim_model, original/255, reconstructed)

            # with torch.no_grad():
            #     avg_perceptanon_deobf_ha1 += model_ha1(test_transforms(reconstructed.clone()))
    stats = validator.get_stats()
    print(stats)
    wandb.log(stats)
    images_normal = wandb.Image(torchvision.utils.make_grid(original))
    images_obfuscated = wandb.Image(torchvision.utils.make_grid(obfuscated, normalize=False))
    if args.obfuscation_type == "obfuscator" and log_image_quality:
        images_deobfuscated = wandb.Image(torchvision.utils.make_grid(reconstructed))
    data['img'] = original
    keypoints_normal = visualize_keypoints_gt(data)
    predictions_obf = val_model(torch.clip(obfuscated, 0, 1))
    keypoints_obfuscated = visualize_keypoints_predicted(obfuscated, predictions_obf)
    keypoints_normal = wandb.Image(torchvision.utils.make_grid(keypoints_normal))
    keypoints_obfuscated = wandb.Image(torchvision.utils.make_grid(keypoints_obfuscated, normalize=False))

    if args.random_mapping:
        print(rand_map.shape)
        print(rand_map.repeat(1, 3, 1, 1).shape)
        maps = wandb.Image(torchvision.utils.make_grid(rand_map.repeat(1, 3, 1, 1)))
        wandb.log({"random_map": maps})
    wandb.log({"normal": images_normal, "obfuscated": images_obfuscated,
               "keypoints_normal": keypoints_normal,
               "keypoints_obfuscated": keypoints_obfuscated, "ssim": avg_ssim / len(dataloader),
               "psnr": avg_psnr / len(dataloader), "vif": avg_vif / len(dataloader), "lpips": avg_lpips / len(dataloader),
               "semsim" : avg_semsim / len(dataloader), 'percept_anon_ha1_obf':avg_perceptanon_ha1 /i, 'percept_anon_ha2_obf':avg_perceptanon_ha2 /i
               })
    if args.obfuscation_type == "obfuscator" and log_image_quality:
        wandb.log({"ssim_deobf": avg_ssim_deobf / len(dataloader), "psnr_deobf": avg_psnr_deobf / len(dataloader),
                   "vif_deobf": avg_vif_deobf / len(dataloader),
                   "lpips_deobf": avg_lpips_deobf / len(dataloader), "deobfuscated": images_deobfuscated,
                    "semsim_deobf" : avg_semsim_deobf / len(dataloader), "perceptanon_deobf_ha1": avg_perceptanon_deobf_ha1 / i, "perceptanon_deobf_ha2": avg_perceptanon_deobf_ha2 / i
                   })
    print({"normal": images_normal, "obfuscated": images_obfuscated,
               "keypoints_normal": keypoints_normal,
               "keypoints_obfuscated": keypoints_obfuscated, "ssim": avg_ssim / len(dataloader),
               "psnr": avg_psnr / len(dataloader), "vif": avg_vif / len(dataloader), "lpips": avg_lpips / len(dataloader),
               "semsim" : avg_semsim / len(dataloader), 'percept_anon_ha1_obf':avg_perceptanon_ha1 /i, 'percept_anon_ha2_obf':avg_perceptanon_ha2 /i
               })
