import argparse
import random
import os

from PIL import Image
from tqdm import tqdm


SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/billyzhaoyh/Desktop/StillGood/RawData', help="Directory with the ingredient dataset")
parser.add_argument('--output_dir', default='/Users/billyzhaoyh/Desktop/StillGood/Data', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    #Resize the image contained in `filename` and save it to the `output_dir`#
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    #name of all the folders that contain data 
    data_set = ['Garlic', 'Carrot', 'Lemon', 'Onion','Lime','Potato','Apple','Broccoli','ButternutSquash','Egg']

    for folder in data_set:
        # Define the data directories from which folder
        raw_data_dir = os.path.join(args.data_dir, folder)

        # Get the filenames in RawData repositry
        filenames = os.listdir(raw_data_dir)
        filenames = [os.path.join(raw_data_dir, f) for f in filenames if f.endswith('.jpg')]


        # Split the images in RawData into 90% train and 10% validation
        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        random.seed(230)
        filenames.sort()
        random.shuffle(filenames)

        split1 = int(0.8 * len(filenames))
        split2 = int(0.9*len(filenames))
        train_filenames = filenames[:split1]
        val_filenames = filenames[split1:split2]
        test_filenames=filenames[split2:]

        filenames = {'Train': train_filenames,
                     'Val': val_filenames,
                     'Test':test_filenames}

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        else:
            print("Warning: output dir {} already exists".format(args.output_dir))

        # Preprocess train, dev and test
        for split in ['Train', 'Val','Test']:
            output_dir_split = os.path.join(args.output_dir, '{}'.format(split),'{}'.format(folder))
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_dir_split))

            print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
            for filename in tqdm(filenames[split]):
                resize_and_save(filename, output_dir_split, size=SIZE)

        print("Done building dataset")