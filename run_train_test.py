import argparse
import yaml
import json
from train_test_classification_model import run_train_test_model


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        print("cfg: ", cfg)
        print("")
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', default="", type=str, help='Path of the config file to use')
    parser.add_argument('--config_string', default="", type=str, help='String with the json configuration file')
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    config_string = opt.config_string
    print('config_string: ', config_string)

    if path_config_file != "":
        cfg = load_config(path_config_file)
    elif config_string != "":
        cfg = json.loads(config_string)

    # 2 - run train and test
    do_train = cfg["model"].get("do_train", 1.0) > 0.0
    do_test = cfg["model"].get("do_test", 1.0) > 0.0
    run_train_test_model(cfg=cfg,
                         do_train=do_train,
                         do_test=do_test)
