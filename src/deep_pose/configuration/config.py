import configparser
import argparse
import os

def get_configuration():
    parser = argparse.ArgumentParser(description="Resnet DeepPose Training")
    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to the .ini file with file path configurations")
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config_path)

    required_config_params = ['train_img_dir', 'train_ann_file', 'val_img_dir', 'val_ann_file']
    for param in required_config_params:
        if param not in config.options("DeepPose"):
            print("ERROR: One or more required config options not present in config file")
            print("Ensure TRAIN_IMG_DIR, TRAIN_ANN_FILE, VAL_IMG_DIR, VAL_ANN_FILE are set")
            exit(1)

    for param in required_config_params:
        if not os.path.exists(config.get("DeepPose", param)):
            print("ERROR: Dataset path not found")
            print("Ensure TRAIN_IMG_DIR, TRAIN_ANN_FILE, VAL_IMG_DIR, VAL_ANN_FILE are set to the correct directories in the config file")
            exit(1)
    
    return config.get("DeepPose", required_config_params[0]), config.get("DeepPose", required_config_params[1]), \
        config.get("DeepPose", required_config_params[2]), config.get("DeepPose", required_config_params[3])

# For testing
if __name__ == '__main__':
    print(get_configuration())