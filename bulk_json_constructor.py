import json
from os import listdir
from os.path import isfile, join

class CustomImageDataset():
    def __init__(self, root_dir):  # root dir of images, transform optional
        files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
        #for f in files:
        #    print(f)

    def combine(self):
        files = [f for f in listdir('processed_shoeprint_jsons') if isfile(join('processed_shoeprint_jsons', f))]
        output_file = open('combined_jsons.json', 'a')
        for f in files:
            with open('processed_shoeprint_jsons/' + f) as fi:
                d = json.load((fi))
                print(d)
                output_file.write(json.dumps(d))
        #with open('all_jsons.json', 'w') as outfile:
        #    json.dump()


c = CustomImageDataset('processed_shoeprint_jsons')
c.combine()