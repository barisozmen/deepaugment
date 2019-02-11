from deepaugment import DeepAugment

deepaug = DeepAugment("cifar10", config={"child_epochs":1})

best_policies = deepaug.optimize(1)

best_policies_2 = deepaug.optimize(1)

print ("best-policies:", best_policies)
print ("best-policies-2:", best_policies_2)

print ("End")





