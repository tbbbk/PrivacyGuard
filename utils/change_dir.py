import json
import yaml
import argparse
from tqdm import tqdm

def change_privacy_dataset_1000(image_dir, sgg_dir):
    # print("privacy_dataset_1000")
    # This is the old directory, please remember to modify it, and the new directory is specified in the dataset YAML file
    old_dir = "/media/zhuohang/Disk21/SGG-GCN/datasets/image_Dataset/privacy_dataset_1000"
    new_dir = image_dir
    private_dir = sgg_dir + '/privacy/origin_custom_data_info.json'
    
    # Read JSON file
    with open(private_dir, 'r') as file:
        data = json.load(file)
    
    # Modify content
    new_data = []
    for file_path in data['idx_to_files']:
        new_path = file_path.replace(old_dir, new_dir)
        new_data.append(new_path)

    # Update data dictionary
    data['idx_to_files'] = new_data

    # Save to another location
    output_path = sgg_dir + '/privacy/output_origin_custom_data_info.json'
    with open(output_path, 'w') as file:
        json.dump(data, file)

    public_dir = sgg_dir + '/public/origin_custom_data_info.json'
    
    # Read JSON file
    with open(public_dir, 'r') as file:
        data = json.load(file)
    
    # Modify content
    new_data = []
    for file_path in data['idx_to_files']:
        new_path = file_path.replace(old_dir, new_dir)
        new_data.append(new_path)

    # Update data dictionary
    data['idx_to_files'] = new_data

    # Save to another location
    output_path = sgg_dir + '/public/output_origin_custom_data_info.json'
    with open(output_path, 'w') as file:
        json.dump(data, file)


def change_fake_img_1000(image_dir, sgg_dir):
    # print("fake_img_1000")
    # This is the old directory, please remember to modify it, and the new directory is specified in the dataset YAML file
    # removed
    old_dir = "/home/zhuohang/Documents/code/dataset/dataset/fakeimg_1000"
    new_dir = image_dir
    for i in tqdm(range(1,15)):
        new_sgg_dir = sgg_dir + f'/{i}/custom_data_info.json'
        
        # Read JSON file
        with open(new_sgg_dir, 'r') as file:
            data = json.load(file)
        
        # Modify content
        new_data = []
        for file_path in data['idx_to_files']:
            new_path = file_path.replace(old_dir, new_dir)
            new_data.append(new_path)

        # Update data dictionary
        data['idx_to_files'] = new_data

        # Save to another location
        output_path = sgg_dir + f'/{i}/output_custom_data_info.json'
        with open(output_path, 'w') as file:
            json.dump(data, file)

def change_mosaic(image_dir, sgg_dir):
    print("mosaic")
    # This is the old directory, please remember to modify it, and the new directory is specified in the dataset YAML file
    old_dir = "/media/zhuohang/Disk21/SGG-GCN/datasets/image_Dataset/mosaic"
    new_dir = image_dir
    private_dir = sgg_dir + '/origin_custom_data_info.json'
    
    # Read JSON file
    with open(private_dir, 'r') as file:
        data = json.load(file)
    
    # Modify content
    new_data = []
    for file_path in data['idx_to_files']:
        new_path = file_path.replace(old_dir, new_dir)
        new_data.append(new_path)

    # Update data dictionary
    data['idx_to_files'] = new_data
    output_path = sgg_dir + '/output_origin_custom_data_info.json'
    with open(output_path, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    with open('./configs/dataset.yaml') as f:
        dataset_info = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="privacy_dataset_1000",help='the dataset want to change(privacy_dataset_1000, mosaic)')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    if(dataset_name == "privacy_dataset_1000"):
        image_dir = dataset_info['dataset']['privacy_dataset_1000']['img_dir']
        sgg_dir = dataset_info['dataset']['privacy_dataset_1000']['sgg_dir']
        print("privacy_dataset_1000 changing...")
        change_privacy_dataset_1000(image_dir, sgg_dir)
    elif(dataset_name == "reltr_privacy_dataset_1000"):
        image_dir = dataset_info['dataset']['reltr_privacy_dataset_1000']['img_dir']
        sgg_dir = dataset_info['dataset']['reltr_privacy_dataset_1000']['sgg_dir']
        print("privacy_dataset_1000 changing...")
        change_privacy_dataset_1000(image_dir, sgg_dir)
    elif(dataset_name == "mosaic"):
        image_dir = dataset_info['dataset']['mosaic']['img_dir']
        print(image_dir)
        sgg_dir = dataset_info['dataset']['mosaic']['sgg_dir']
        # print("we have not written this")
        change_mosaic(image_dir, sgg_dir)
    else:
        print("Error! We have not implemented anything about the custom dataset.")