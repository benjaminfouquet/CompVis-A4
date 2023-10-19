import csv
import glob
import numpy as np


def create_paired_embeddings_dict(all_full_encodings, ALL_FILES_PATH, LEFT_FILES_PATH, RIGHT_FILES_PATH):
    """
    Create a list with all the file names and the associated embeddings
    """
    embeddings_dict = {}

    all_file_names = glob.glob(ALL_FILES_PATH)
    # Loop over all the file names to create a dict with the file names and the corresponding embeddings
    for index, file_name in enumerate(all_file_names):
        base_name = file_name.split("/")[-1]  # Extract the filename from the path
        prefix = base_name[:-4]  # Get the three letters before "jpg"
        embeddings_dict[prefix] = all_full_encodings[index]

    left_file_names = glob.glob(LEFT_FILES_PATH)
    left_images_names = [file_name.split("/")[-1][:-4] for file_name in left_file_names]

    right_file_names = glob.glob(RIGHT_FILES_PATH)
    right_images_names = [file_name.split("/")[-1][:-4] for file_name in right_file_names]

    paired_embeddings = []
    # Loop over all the file names in embeddings_dict and check if the file is in left_file_names or right_file_names
    for image_name, embeddings in embeddings_dict.items():
        pair_dict = {}
        if image_name in left_images_names:
            pair_dict['left'] = embeddings
            pair_dict['left_name'] = image_name
        elif image_name in right_images_names:
            pair_dict['right'] = embeddings
            pair_dict['right_name'] = image_name
        else:
            print('Missing image')
        paired_embeddings.append(pair_dict)

    return paired_embeddings


def merge_dicts_with_csv(csv_filename, dict_list):
    """
    Create a dict containing all the data in the following format: {'left', 'right', 'left_name', 'right_name'}
    """
    result = {'left': [], 'right': [], 'left_name': [], 'right_name': []}

    # Read the CSV file into a dictionary with left as the key
    csv_data = {}
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_data[row['left']] = row

    for item in dict_list:
        if 'left_name' in item:
            left_name = item['left_name']
            if left_name in csv_data:
                left_data = csv_data[left_name]
                right_name = left_data.get('right', None)

                if right_name:
                    # Find the dictionary in the input list with the corresponding 'right_name'
                    matching_dict = next((d for d in dict_list if d.get('right_name') == right_name), None)

                    if matching_dict:
                        # Merge the matching dictionary from the input list with the CSV dictionary
                        merged_dict = {**left_data, **matching_dict}

                        # Add the left and right image names
                        merged_dict['left_image'] = left_name
                        merged_dict['right_image'] = right_name

                        result['left'].append(item['left'])
                        result['right'].append(matching_dict['right'])
                        result['left_name'].append(left_name)
                        result['right_name'].append(right_name)
    result['left'] = np.array(result['left'])
    result['right'] = np.array(result['right'])
    return result