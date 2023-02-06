import pandas as pd
import cv2
import albumentations as A
import os
import shutil
from sklearn.model_selection import train_test_split


path_train_csv = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train.csv"
path_augmented_train_csv = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/final_train_val_split_augmented/augmented_train.csv"
path_val_csv = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/final_train_val_split_augmented/val.csv"
path_images = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_images_processed_no_pad512"
path_augmented_images = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/final_train_val_split_augmented/train_val_augmented_no_pad_512"

CHECK_FOLDER = os.path.isdir(path_augmented_images)
if not CHECK_FOLDER:
    print("Creo la directory '{}'".format(path_augmented_images))
    os.makedirs(path_augmented_images)

n_applications = 25


# Functions
transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.CLAHE(p=0.7),
    A.RandomGamma(p=0.7),
    A.Flip(p=0.7),
    A.Transpose(p=0.7),
])


def train_val_split(path_train_csv):

    train_df = pd.read_csv(path_train_csv)
    print("Cancer distribution in train set")
    count_target = train_df.groupby('cancer').size()
    percent_target=count_target.apply(lambda x: x/len(train_df)*100)

    print('cancer absolute distribution')
    print(round(count_target,3))
    print("")
    print('cancer relative distribution')
    print(round(percent_target,2))

    print("-------------------")

    print("laterality distribution in train set")
    count_target = train_df.groupby('laterality').size()
    percent_target=count_target.apply(lambda x: x/len(train_df)*100)

    print('laterality absolute distribution')
    print(round(count_target,3))
    print("")
    print('laterality relative distribution')
    print(round(percent_target,2))


    print("-------------------")
    print("-------------------")
    print("Train test split")
    print("-------------------")
    print("-------------------")


    #train val split stratified respect to the cancer label
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0, stratify=train_df[['cancer', 'laterality']])

    #reset index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    count_target = train_df.groupby('cancer').size()
    percent_target=count_target.apply(lambda x: x/len(train_df)*100)

    print('cancer absolute distribution')
    print(round(count_target,3))
    print("")
    print('cancer relative distribution')
    print(round(percent_target,2))

    print("-------------------")

    count_target = train_df.groupby('laterality').size()
    percent_target = count_target.apply(lambda x: x/len(train_df)*100)
    print('laterality absolute distribution')
    print(round(count_target,3))
    print("")
    print('laterality relative distribution')
    print(round(percent_target,2))

    print("-------------------")
    print("-------------------")

    count_target = val_df.groupby('cancer').size()
    percent_target=count_target.apply(lambda x: x/len(val_df)*100)
    print('cancer absolute distribution in val set')
    print(round(count_target,3))
    print("")
    print('cancer relative distribution')
    print(round(percent_target,2))

    print("-------------------")

    count_target = val_df.groupby('laterality').size()
    percent_target=count_target.apply(lambda x: x/len(val_df)*100)
    print('laterality absolute distribution')
    print(round(count_target,3))
    print("")
    print('laterality relative distribution')
    print(round(percent_target,2))

    return train_df, val_df


def augment_train_set(path_dataset_in, path_dataset_out, train_df, path_augmented_csv_train, n_applications):

    number_syntetic_images = 0
    print("train_df info BEFORE")
    print(train_df.info())
    # change image_id from int64 to string field
    train_df['image_id'] = train_df['image_id'].astype(str)

    # iter over the row of the csv
    for index, row in train_df.iterrows():
        patient_id = row['patient_id']
        image_id = row['image_id']
        laterality = row['laterality']
        view = row['view']
        age = row['age']
        implant = row['implant']
        cancer = row['cancer']

        img_path = os.path.join(path_dataset_in, str(patient_id), str(image_id) + '.png')
        new_dir = os.path.join(path_dataset_out, str(patient_id))
        CHECK_FOLDER = os.path.isdir(new_dir)
        if not CHECK_FOLDER:
            # print("Creo la directory '{}'".format(new_dir))
            os.makedirs(new_dir)

        # copy the original image in both cases cancer 0 and 1
        file_name = img_path.split("/")[-1]
        dst_file = os.path.join(new_dir, file_name)
        if not os.path.isfile(dst_file):
            shutil.copy2(img_path, dst_file)

        if cancer == 1:
            # generate new synthetic images
            image = cv2.imread(img_path)

            for i in range(n_applications):
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented_image = transform(image=image)['image']
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                new_file_name = str(image_id) + '_' + str(i) + '.png'
                dst_file = os.path.join(new_dir, new_file_name)
                cv2.imwrite(dst_file, augmented_image)
                train_df = train_df.append({'patient_id': patient_id,
                                            'image_id': str(image_id) + '_' + str(i),
                                            'laterality': laterality,
                                            'view': view,
                                            'age': age,
                                            'implant': implant,
                                            'cancer': cancer,
                                            }, ignore_index=True)
                number_syntetic_images += 1

    # apply shuffling
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    # save new train fold dataset
    print("number_syntetic_images: ", number_syntetic_images)

    count_target = train_df.groupby('cancer').size()
    percent_target = count_target.apply(lambda x: x / len(train_df) * 100)
    print('Cancer absolute distribution after train augmentation')
    print(round(count_target, 3))
    print("")
    print('Cancer relative distribution after train augmentation')
    print(round(percent_target, 2))

    print("-------------------")

    count_target = train_df.groupby('laterality').size()
    percent_target = count_target.apply(lambda x: x / len(train_df) * 100)
    print('laterality absolute distribution after train augmentation')
    print(round(count_target, 3))
    print("")
    print('laterality relative distribution after train augmentation')
    print(round(percent_target, 2))

    print("save augmented train csv")
    train_df.to_csv(path_augmented_csv_train, index=False)
    print("train_df info AFTER")
    print(train_df.info())


def copy_val_set(path_dataset_in, path_dataset_out, val_df, path_val_df):
    print(val_df.info())
    # change image_id from int64 to string field
    val_df['image_id'] = val_df['image_id'].astype(str)
    copied_img = 0

    # iter over the row of the csv
    for index, row in val_df.iterrows():
        patient_id = row['patient_id']
        image_id = row['image_id']

        img_path = os.path.join(path_dataset_in, str(patient_id), str(image_id) + '.png')
        new_dir = os.path.join(path_dataset_out, str(patient_id))
        CHECK_FOLDER = os.path.isdir(new_dir)
        if not CHECK_FOLDER:
            # print("Creo la directory '{}'".format(new_dir))
            os.makedirs(new_dir)

        # copy the original image in both cases cancer 0 and 1
        file_name = img_path.split("/")[-1]
        dst_file = os.path.join(new_dir, file_name)
        if not os.path.isfile(dst_file):
            shutil.copy2(img_path, dst_file)
            copied_img += 1

    print("Copied images: ", copied_img)
    print("Save augmented csv")
    val_df.to_csv(path_val_df, index=False)



# 1 train val split
print("1 - train val split")
train_df, val_df = train_val_split(path_train_csv)

print("**********************")
print("**********************")
print("2 - augment train images when cancer = 1")

augment_train_set(path_images, path_augmented_images, train_df, path_augmented_train_csv, n_applications)

print("**********************")
print("**********************")
print("3 - copy val dataset")

copy_val_set(path_images, path_augmented_images, val_df, path_val_csv)
