from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
import numpy as np


project_path = WindowsPath(__file__).parent

datasets_path = project_path / 'datasets'
deepfashion_path = datasets_path / 'DeepFashion'
fashion550k_path = datasets_path / 'Fashion550k'

models_path = project_path / 'models'
models_path.mkdir(parents=True, exist_ok=True)
log_path = project_path / 'log'
log_path.mkdir(parents=True, exist_ok=True)

def relative_path(origin, destination):
    from os.path import relpath
    return relpath(destination, start=origin)



DATA_FORMAT = 'channels_last'
IMG_SIZE = (224, 224)
IMG_SHAPE = (
    IMG_SIZE + (3,)
    if DATA_FORMAT == 'channels_last'
    else (3,) + IMG_SIZE
)
BATCH_SIZE = 64
RESCALE = 1./255
FILL_MODE = 'nearest'

validate_filenames = False

augment_previously = True
fill_aug_imgs = True



from sklearn.preprocessing import MultiLabelBinarizer

category_attribute_prediction_path = (deepfashion_path / 'Category and Attribute Prediction Benchmark').absolute().resolve()
images_path = (category_attribute_prediction_path / 'Img').absolute().resolve()
augmented_imgs_path = (category_attribute_prediction_path / 'Img' / 'aug_img').absolute().resolve()
annotations_path = (category_attribute_prediction_path / 'Anno').absolute().resolve()
evaluation_path = (category_attribute_prediction_path / 'Eval').absolute().resolve()

dataset_anno_path = annotations_path / 'dataset.csv'
dataset_path = (category_attribute_prediction_path / 'Img' / 'dataset').absolute().resolve()
dataset_path.mkdir(parents=True, exist_ok=True)

n_attributes = int(pd.read_csv(annotations_path / 'list_attr_cloth.txt', nrows=1, header=None)[0][0])
attributes = pd.read_csv(annotations_path / 'list_attr_cloth.txt', skiprows=1, delimiter=r"\s\s+", engine='python')
attributes = attributes.astype({
    'attribute_name': str,
    'attribute_type': int
})
attribute_types = {
    1: 'texture-related attributes',
    2: 'fabric-related attributes',
    3: 'shape-related attributes',
    4: 'part-related attributes',
    5: 'style-related attributes'
}

attr_binarizer = MultiLabelBinarizer()
attr_binarizer.fit([attributes['attribute_name']])
n_labels = attr_binarizer.classes_.size
print(n_labels, 'classes found:', attr_binarizer.classes_)


corrected_path = (annotations_path / 'attribute_imgs copy.txt').absolute().resolve()


from ast import literal_eval
from tensorflow.keras.preprocessing.image import save_img
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

n_attribute_imgs = int(pd.read_csv(annotations_path / 'list_attr_img.txt', nrows=1, header=None)[0][0])
column_names = pd.read_csv(annotations_path / 'list_attr_img.txt', skiprows=1, nrows=1, delim_whitespace=True, header=None).values[0]

previously_augmented = False
if augment_previously:
    attribute_aug_imgs_path = annotations_path / 'attribute_aug_imgs_.txt'
    try:
        print('Trying to read augmented image attributes', end='')
        attribute_aug_imgs = pd.read_csv(attribute_aug_imgs_path)
        attribute_aug_imgs[column_names[1]] = attribute_aug_imgs[column_names[1]].apply(lambda x: list(y for y in literal_eval(x)))

        attribute_aug_imgs = attribute_aug_imgs.astype({
            column_names[0]: str,
            column_names[1]: object
        })

        print(' - Done')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(' - File not found... augmenting images later')
        attribute_aug_imgs = pd.DataFrame(columns=column_names)
        attribute_aug_imgs.to_csv(attribute_aug_imgs_path, index=False)

    n_attribute_aug_imgs = len(attribute_aug_imgs.index)
    if fill_aug_imgs:
        previously_augmented = (n_attribute_aug_imgs == n_attribute_imgs)
    else:
        previously_augmented = True

if not augment_previously or not previously_augmented:
    attribute_imgs_path = annotations_path / 'attribute_imgs_.txt'
    try:
        print('Trying to read image attributes', end='')
        attribute_imgs = pd.read_csv(attribute_imgs_path)
        attribute_imgs[column_names[1]] = attribute_imgs[column_names[1]].apply(lambda x: list(y for y in literal_eval(x)))
        print(' - Done')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(' - File not found... reading in chunks from bigger file')
        first_row = pd.read_csv(annotations_path / 'list_attr_img.txt', skiprows=2, nrows=1, delim_whitespace=True, header=None).values[0]
        attribute_imgs = pd.DataFrame(columns=column_names)
        for i, chunk in enumerate(pd.read_csv(
                annotations_path / 'list_attr_img.txt',
                skiprows=2,
                delim_whitespace=True,
                header=None,
                chunksize=50000,
                converters={
                    col: lambda x: True if x == '1' else False if x == '-1' else x
                    for col in range(1, len(first_row))
                }
        )):
            print(f'Reading chunk #{i}', end='')

            chunk = list(chunk.itertuples())
            y_values = [
                list(
                    idx
                    for idx, value in enumerate(row[2:])
                    if value
                ) for row in chunk
            ]
            chunk = pd.DataFrame({
                column_names[0]: [row[1] for row in chunk],
                column_names[1]: [[attributes['attribute_name'][y_] for y_ in y_value] if y_value else ['none'] for y_value in y_values]
            }, index=[row[0] for row in chunk])

            attribute_imgs = attribute_imgs.append(chunk, verify_integrity=True, sort=False)

            print(f' - Done')
        try:
            print('Trying to write attribute images', end='')
            attribute_imgs.to_csv(attribute_imgs_path, index=False)
            print(' - Done')
        except PermissionError:
            print(' - Permission denied')

    attribute_imgs = attribute_imgs.astype({
        column_names[0]: str,
        column_names[1]: object
    })
    converters = {
        column_names[0]: lambda x: str(images_path / x),
    } 
    full_attribute_imgs = pd.DataFrame.from_dict({
        col: series.apply(converters[col])
        if col in converters
        else series
        for col, series in attribute_imgs.iteritems()
    })
    
    print(full_attribute_imgs)

expanded_attr_binarizer = MultiLabelBinarizer()
if not augment_previously or not previously_augmented:
    expanded_attr_binarizer.fit(attribute_imgs[column_names[1]])
else:
    expanded_attr_binarizer.fit(attribute_aug_imgs[column_names[1]])

print(Path(__file__))

if augment_previously and not previously_augmented:
    print('Augmenting images')

    image_data_generator = ImageDataGenerator(
        rescale=RESCALE,
        data_format=DATA_FORMAT,
        fill_mode=FILL_MODE,
    )
    augmented_imgs_path.mkdir(parents=True, exist_ok=True)

    dataframe_iterator = image_data_generator.flow_from_dataframe(
        full_attribute_imgs[n_attribute_aug_imgs:],
        x_col=column_names[0],
        y_col=column_names[1],
        color_mode='rgb',
        classes=list(expanded_attr_binarizer.classes_),
        class_mode='categorical',
        validate_filenames=validate_filenames,
        shuffle=False,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    counter = n_attribute_aug_imgs
    break_all = False
    for batch in dataframe_iterator:
        idx = 0
        for in_batch_idx, (X, y) in enumerate(zip(*batch)):
            y = np.expand_dims(y, axis=0)
            inv_y = expanded_attr_binarizer.inverse_transform(y)
            
            origin = Path(dataframe_iterator.filenames[idx])
            file_path = augmented_imgs_path / origin.parts[-2] / origin.parts[-1]
            
            print(f'Index = {idx}/{n_attribute_imgs} [#{in_batch_idx} from batch {dataframe_iterator.batch_index}]')
            print(f'Origin = {origin}')
            print(f'y = {inv_y} [path = {file_path}]')

            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                save_img(file_path, X)
            attribute_aug_imgs = attribute_aug_imgs.append({
                column_names[0]: str(file_path),
                column_names[1]: inv_y[0]
            }, ignore_index=True)

            idx += 1
            counter += 1
            if counter == n_attribute_imgs:
                break_all = True
                break

        attribute_aug_imgs.to_csv(attribute_aug_imgs_path, index=False)
        if break_all:
            break

    previously_augmented = True
    print('Done')
    
    
    
    
# from sklearn.model_selection import train_test_split

# # Setup ============================
# dataset_size = 10000
# # problem = 'category'
# problem = 'attribute'
# from_file = False # keep it False!
# # ==================================

# if from_file:
#     try:
#         dataset = pd.read_csv(dataset_anno_path)
#         from_file_success = True
#     except FileNotFoundError:
#         from_file_success = False

# if (not from_file) or (not from_file_success):
#     if problem == 'attribute':
#         if previously_augmented:
#             dataset = attribute_aug_imgs
#         else:
#             dataset = attribute_imgs
#     elif problem == 'category':
#         if previously_augmented:
#             dataset = category_aug_imgs
#         else:
#             dataset = category_imgs

#     dataset, _ = train_test_split(dataset, train_size=dataset_size)

# if from_file and not from_file_success:
#     dataset_iter = ImageDataGenerator(
#         data_format=DATA_FORMAT,
#         fill_mode=FILL_MODE,
#     ).flow_from_dataframe(
#         dataset,
#         x_col=column_names[0],
#         y_col=column_names[1],
#         color_mode='rgb',
#         classes=list(expanded_attr_binarizer.classes_),
#         class_mode='categorical',
#         validate_filenames=validate_filenames,
#         shuffle=False,
#         target_size=IMG_SIZE,
#         batch_size=BATCH_SIZE
#     )

#     counter = 0
#     break_all = False
#     for batch in dataset_iter:
#         print_header(f'Batch {dataset_iter.batch_index}')
#         idx = 0
#         for in_batch_idx, (X, y) in enumerate(zip(*batch)):
#             y = np.expand_dims(y, axis=0)
#             inv_y = expanded_attr_binarizer.inverse_transform(y)
            
#             origin = Path(dataset_iter.filenames[idx])
#             file_path = dataset_path / origin.parts[-2] / origin.parts[-1]
            
#             print(f'Index = {counter}/{len(dataset.index)} [#{in_batch_idx} from batch {dataset_iter.batch_index}]')
#             print(f'Origin = {origin}')
#             print(f'y = {inv_y} [path = {file_path}]')

#             file_path.parent.mkdir(parents=True, exist_ok=True)
#             if not file_path.exists():
#                 save_img(file_path, X)

#             idx += 1
#             counter += 1
#             if counter == len(dataset.index):
#                 break_all = True
#                 break

#         if break_all:
#             break

#     from pathlib import Path

#     dataset[column_names[0]] = dataset[column_names[0]].apply(
#         lambda file_path: relative_path(
#             images_path,
#             dataset_path / Path(file_path).parts[-2] / Path(file_path).parts[-1]
#         )
#     )
#     dataset.to_csv(dataset_anno_path, index=False)

# val_size = 0.2
# test_size = 0.2
# train_set, test_set = train_test_split(dataset, test_size=test_size)
# train_set, val_set = train_test_split(train_set, test_size=val_size/(1 - test_size))

# total = sum(data.shape[0] for data in (train_set, val_set, test_set))
# for name, data in (('Train set', train_set), ('Validation set', val_set), ('Test set', test_set)):
#     print(f'{name} shape: {data.shape} [{100*data.shape[0]/total:.0f}% of data]')

# dataset.info()
# dataset.head()
# train_set.info()
# train_set.head()
# val_set.info()
# val_set.head()
# test_set.info()
# test_set.head()