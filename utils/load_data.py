import json
import yaml
import warnings
def is_number(string):
    return string.isdigit()

def load_mosaic_info(label_name="all", file_name="reltr_custom_data_info.json"):
    print("loading... mosaic_info")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    mosaic_data_info_base_dir = dataset_info['dataset']['mosaic']['sgg_dir']
    mosaic_data_info_dir = mosaic_data_info_base_dir + f'/{file_name}'
    with open(mosaic_data_info_dir, 'r') as f:
        mosaic_data_info = json.load(f)
    print("load sucessfully")
    if(label_name == "all"):
        return mosaic_data_info
    elif(label_name == "idx_to_files"):
        return mosaic_data_info["idx_to_files"]
    elif(label_name == "ind_to_predicates"):
        return mosaic_data_info["ind_to_predicates"]
    elif(label_name == "ind_to_classes"):
        return mosaic_data_info["ind_to_classes"]
    else:
        warnings.warn("check load mosaic info input!!!!!!!")

def load_mosaic_prediction(label_name="all", file_name="reltr_custom_prediction.json"):
    print("loading... mosaic_prediction")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    mosaic_data_prediction_dir = dataset_info['dataset']['mosaic']['sgg_dir']
    mosaic_prediction_dir = mosaic_data_prediction_dir + f'/{file_name}'
    with open(mosaic_prediction_dir, 'r') as f:
        mosaic_data_prediction = json.load(f)
    print("load sucessfully")
    if(label_name == "all"):
        return mosaic_data_prediction
    elif(is_number(label_name)):
        return mosaic_data_prediction[label_name]
    else:
        warnings.warn("check load mosaic prediction input!!!!!!!")

def load_privacy_info(label_name="all", file_name="reltr_custom_data_info.json"):
    print("loading... load_privacy_info")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    
    privacy_data_info_base_dir = dataset_info['dataset']['privacy_dataset_1000']['sgg_dir']+'/privacy'
    privacy_data_info_dir = privacy_data_info_base_dir + f'/{file_name}'
    print(privacy_data_info_dir)
    with open(privacy_data_info_dir, 'r') as f:
        privacy_data_info = json.load(f)
    print("load sucessfully")
    if(label_name == "all"):
        return privacy_data_info
    elif(label_name == "idx_to_files"):
        return privacy_data_info["idx_to_files"]
    elif(label_name == "ind_to_predicates"):
        return privacy_data_info["ind_to_predicates"]
    elif(label_name == "ind_to_classes"):
        return privacy_data_info["ind_to_classes"]
    else:
        warnings.warn("check load privacy info input!!!!!!!")

def load_public_info(label_name="all", file_name="custom_data_info.json"):
    print("loading... load_public_info")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    
    public_data_info_base_dir = dataset_info['dataset']['privacy_dataset_1000']['sgg_dir']+'/public'
    public_data_info_dir = public_data_info_base_dir + f'/{file_name}'
    with open(public_data_info_dir, 'r') as f:
        public_data_info = json.load(f)
    print("load sucessfully")
    if(label_name == "all"):
        return public_data_info
    elif(label_name == "idx_to_files"):
        return public_data_info["idx_to_files"]
    elif(label_name == "ind_to_predicates"):
        return public_data_info["ind_to_predicates"]
    elif(label_name == "ind_to_classes"):
        return public_data_info["ind_to_classes"]
    else:
        warnings.warn("check load public info input!!!!!!!")

def load_privacy_prediction(label_name="all", file_name="reltr_custom_prediction.json"):
    print("loading... load_privacy_prediction")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    
    privacy_data_prediction_dir = dataset_info['dataset']['privacy_dataset_1000']['sgg_dir']+'/privacy'
    privacy_prediction_dir = privacy_data_prediction_dir + f'/{file_name}'
    with open(privacy_prediction_dir, 'r') as f:
        privacy_data_prediction = json.load(f)
    print("load sucessfully")
    if(label_name == "all"):
        return privacy_data_prediction
    elif(is_number(label_name)):
        return privacy_data_prediction[label_name]
    else:
        warnings.warn("check load privacy prediction input!!!!!!!")

def load_public_prediction(label_name="all", file_name="custom_prediction.json"):
    print("loading... load_public_prediction")
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    print("load sucessfully")
    public_data_prediction_dir = dataset_info['dataset']['privacy_dataset_1000']['sgg_dir']+'/public'
    public_prediction_dir = public_data_prediction_dir + f'/{file_name}'
    with open(public_prediction_dir, 'r') as f:
        public_data_prediction = json.load(f)
    if(label_name == "all"):
        return public_data_prediction
    elif(is_number(label_name)):
        return public_data_prediction[label_name]
    else:
        warnings.warn("check load public prediction input!!!!!!!")
    
if __name__ == "__main__":
    print("test")