import pandas as pd
import cv2
import albumentations as A
import os
import shutil


path_train_df_0 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_fold_0.csv"
path_train_df_1 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_fold_1.csv"
path_train_df_2 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_fold_2.csv"

path_test_df_0 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/test_fold_0.csv"
path_test_df_1 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/test_fold_1.csv"
path_test_df_2 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/test_fold_2.csv"

path_augmented_train_df_0 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_augmented_fold_0.csv"
path_augmented_train_df_1 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_augmented_fold_1.csv"
path_augmented_train_df_2 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_augmented_fold_2.csv"

path_images = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/train_images_processed_no_pad512"

path_augmented_train_test_fold_0 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/augmented_train_test_no_pad_512_fold_0"
path_augmented_train_test_fold_1 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/augmented_train_test_no_pad_512_fold_1"
path_augmented_train_test_fold_2 = "/Volumes/HD-Enrico/Dataset/breast_cancer_RSNA/augmented_train_test_no_pad_512_fold_2"

n_applications = 20


os.makedirs(path_augmented_train_test_fold_0)
os.makedirs(path_augmented_train_test_fold_1)
os.makedirs(path_augmented_train_test_fold_2)


# Functions
transform = A.Compose([
    A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.CLAHE(p=0.7),
    A.RandomGamma(p=0.7),
    A.Flip(p=0.7),
    A.Transpose(p=0.7),
])


def augment_train_fold(path_dataset_in, path_dataset_out, path_csv_train, path_augmented_csv_train, n_applications,
                       fold):
    number_syntetic_images = 0
    train_df = pd.read_csv(path_csv_train)
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
                new_file_name = str(image_id) + '_' + str(fold) + '_' + str(i) + '.png'
                dst_file = os.path.join(new_dir, new_file_name)
                cv2.imwrite(dst_file, augmented_image)
                train_df = train_df.append({'patient_id': patient_id,
                                            'image_id': str(image_id) + '_' + str(fold) + '_' + str(i),
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
    print("save augmented csv")
    train_df.to_csv(path_augmented_csv_train, index=False)
    print("train_df info AFTER")
    print(train_df.info())


def copy_test_fold(path_dataset_in, path_dataset_out, path_csv_test):
    test_df = pd.read_csv(path_csv_test)
    print("train_df info BEFORE")
    print(test_df.info())
    # change image_id from int64 to string field
    test_df['image_id'] = test_df['image_id'].astype(str)

    # iter over the row of the csv
    for index, row in test_df.iterrows():
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

number_files = 0
for subdir, dirs, files in os.walk(path_images):
    for cl in dirs:
        path_cl = os.path.join(path_images, cl)
        CHECK_FOLDER = os.path.isdir(path_cl)
        if CHECK_FOLDER:
            number_files += len(os.listdir(path_cl))
print("number_files: ", number_files)


train_df_0 = pd.read_csv(path_train_df_0)
train_df_1 = pd.read_csv(path_train_df_1)
train_df_2 = pd.read_csv(path_train_df_2)

print("Fold 0")
count_target = train_df_0.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_df_0)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))
print("")

print("Fold 1")
count_target = train_df_1.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_df_1)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))
print("")

print("Fold 2")
count_target = train_df_2.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_df_2)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))



print("Augment train fold 0")
augment_train_fold(path_dataset_in=path_images,
                   path_dataset_out=path_augmented_train_test_fold_0,
                   path_csv_train=path_train_df_0,
                   path_augmented_csv_train=path_augmented_train_df_0,
                   n_applications=n_applications,
                   fold=0)
print("Copy test fold 0")
copy_test_fold(path_dataset_in=path_images,
               path_dataset_out=path_augmented_train_test_fold_0,
               path_csv_test=path_test_df_0)

print("")
print("Augment train fold 1")
augment_train_fold(path_dataset_in=path_images,
                   path_dataset_out=path_augmented_train_test_fold_1,
                   path_csv_train=path_train_df_1,
                   path_augmented_csv_train=path_augmented_train_df_1,
                   n_applications=n_applications,
                   fold=1)
print("Copy test fold 1")
copy_test_fold(path_dataset_in=path_images,
               path_dataset_out=path_augmented_train_test_fold_1,
               path_csv_test=path_test_df_1)

print("")
print("Augment train fold 2")
augment_train_fold(path_dataset_in=path_images,
                   path_dataset_out=path_augmented_train_test_fold_2,
                   path_csv_train=path_train_df_2,
                   path_augmented_csv_train=path_augmented_train_df_2,
                   n_applications=n_applications,
                   fold=2)
print("Copy test fold 2")
copy_test_fold(path_dataset_in=path_images,
               path_dataset_out=path_augmented_train_test_fold_2,
               path_csv_test=path_test_df_2)


train_augmented_df_0 = pd.read_csv(path_augmented_train_df_0)
train_augmented_df_1 = pd.read_csv(path_augmented_train_df_1)
train_augmented_df_2 = pd.read_csv(path_augmented_train_df_2)

print("Fold 0")
count_target = train_augmented_df_0.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_augmented_df_0)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))
print("")

print("Fold 1")
count_target = train_augmented_df_1.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_augmented_df_1)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))
print("")

print("Fold 2")
count_target = train_augmented_df_2.groupby('cancer').size()
percent_target=count_target.apply(lambda x: x/len(train_augmented_df_2)*100)
print('Target absolute distribution')
print(round(count_target,3))
print("")
print('Target relative distribution')
print(round(percent_target,2))
print("")



number_files = 0
for subdir, dirs, files in os.walk(path_augmented_train_test_fold_0):
    for cl in dirs:
        path_cl = os.path.join(path_images, cl)
        CHECK_FOLDER = os.path.isdir(path_cl)
        if CHECK_FOLDER:
            number_files += len(os.listdir(path_cl))
print("number files fold 0: ", number_files)

number_files = 0
for subdir, dirs, files in os.walk(path_augmented_train_test_fold_1):
    for cl in dirs:
        path_cl = os.path.join(path_images, cl)
        CHECK_FOLDER = os.path.isdir(path_cl)
        if CHECK_FOLDER:
            number_files += len(os.listdir(path_cl))
print("number files fold 1: ", number_files)

number_files = 0
for subdir, dirs, files in os.walk(path_augmented_train_test_fold_2):
    for cl in dirs:
        path_cl = os.path.join(path_images, cl)
        CHECK_FOLDER = os.path.isdir(path_cl)
        if CHECK_FOLDER:
            number_files += len(os.listdir(path_cl))
print("number files fold 2: ", number_files)