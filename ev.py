import argparse

import torch

from architectures.preact_resnet import PreActResNet18
from utils.tiny_imagenet import TINYIMAGENETUtils

from utils.attack_utils import AttackUtils

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/path/to/datasets/', type=str)
parser.add_argument('--epsilon_test', default=8, type=int,
                    help='''Epsilon to be used at test time (only for final model,
                    if computing loss during training epsilon train is used).
                    If set to None, default, the same args.epsilon will be used for test and train.''')
parser.add_argument('--model-path', type=str, default='/kaggle/input/vit-n-fgsm/ex1/ex1/model.pth')

args = parser.parse_args()

model_test = PreActResNet18(kernel_size=7, stride=2, padding=3, num_classes=args.num_classes).cuda()

model_test.load_state_dict(torch.load(args.model_path))
model_test.float()
model_test.eval()

data_utils = TINYIMAGENETUtils()
attack_utils = AttackUtils(data_utils.lower_limit, data_utils.upper_limit, data_utils.std)
(train_loader, test_loader, robust_test_loader,
 valid_loader, train_idx, valid_idx) = data_utils.get_indexed_loaders(args.data_dir,
                                                                      128,
                                                                      valid_size=0,
                                                                      robust_test_size=-1)

print("start evaluating...")
pgd_loss, pgd_acc = attack_utils.evaluate_pgd(robust_test_loader, model_test, 50, 10, epsilon=args.epsilon_test)
test_loss, test_acc = attack_utils.evaluate_standard(test_loader, model_test)
print("pgd acc: ", pgd_acc)
print("test acc: ", test_acc)
