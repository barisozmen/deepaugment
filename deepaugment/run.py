
import sys
from os.path import dirname, realpath
file_path = realpath(__file__)
dir_of_file = dirname(file_path)
sys.path.insert(0, dir_of_file)

from deepaugment import DeepAugment

deepaug = DeepAugment("cifar10", config={"child_epochs":1})

# best_policies = deepaug.optimize(1)

# best_policies_2 = deepaug.optimize(1)

print ("best-policies:", best_policies)
print ("best-policies-2:", best_policies_2)

print ("End")





