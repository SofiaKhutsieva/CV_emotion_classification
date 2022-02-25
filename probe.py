from build_features import build_features

data_dir = './data'
dataloaders, dataset_sizes, class_names = build_features(data_dir)
print(class_names)