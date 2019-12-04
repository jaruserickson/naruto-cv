""" Read the output from generate_split.py, and output COCO friendly files. """
import csv
import shutil
import os

CLASSES = [
    'naruto_uzumaki', 'sasuke_uchiha', 'kakashi_hatake', 'gaara', 'itachi_uchiha',
    'deidara', 'minato_namikaze', 'shikamaru_nara', 'hinata_hyuuga', 'sakura_haruno',
    'sai', 'yamato', 'neji_hyuuga', 'jiraya', 'temari', 'rock_lee', 'kushina_uzumaki',
    'kisame_hoshigaki', 'killer_bee', 'might_guy', 'kiba_inuzuka', 'ino_yamanaka',
    'sasori', 'pain', 'konan', 'iruka_umino', 'shino_aburame']

def copy_images(src, dst):
    for item in os.listdir(src):
        if item.split('.')[-1] == 'png':
            shutil.copy2(os.path.join(src, item), dst)

def convert_to_coco(folder='images', out_folder='coco', data_split=0.8):
    """ Convert contents of folder to COCO format in outfolder"""
    for split in ['train', 'test']:
        out_path = os.path.join(out_folder, 'images')
        label_path = os.path.join(out_folder, 'labels')
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        if not os.path.isdir(label_path):
            os.mkdir(label_path)
        # Copy images
        copy_images(
            src=os.path.join(folder, split),
            dst=out_path)

        # Copy metadata
        with open(f'{folder}/{split}_labels.csv', 'r') as infile:
            reader = csv.reader(infile)
            for i, line in enumerate(reader):
                if i > 0:
                    filename, w, h, clss, xmin, ymin, xmax, ymax = line
                    w, h, xmin, ymin, xmax, ymax = map(int, [w, h, xmin, ymin, xmax, ymax])
                    xcenter = (xmax - ((xmax-xmin)/2)) / w
                    width = (xmax - xmin) / w
                    ycenter = (ymax - ((ymax-ymin)/2)) / h
                    height = (ymax - ymin) / h
                    newline = f'{CLASSES.index(clss)} {xcenter} {ycenter} {width} {height}'
                    ann_path = f"{label_path}/{filename.split('.png')[0]}.txt"
                    with open(ann_path, ('a+' if os.path.isfile(ann_path) else 'w')) as txtfile:
                        txtfile.write(newline + '\n')

    # Create namefile
    with open(f'{out_folder}/naruto.names', 'w') as outfile:
        outfile.writelines([l + '\n' for l in CLASSES])

    # Create splitfiles
    files = os.listdir(f'{out_folder}/images')
    files = [f'../../dataset/{out_folder}/images/' + l for l in files]
    split_index = int(round(len(CLASSES) * data_split))
    train_split = files[:split_index]
    test_split = files[split_index:]

    with open(f'{out_folder}/naruto_train.txt', 'w') as outfile:
        outfile.writelines([l + '\n' for l in train_split])
    with open(f'{out_folder}/naruto_test.txt', 'w') as outfile:
        outfile.writelines([l + '\n' for l in test_split])

    # Create data config
    with open(f'{out_folder}/naruto.data', 'w') as outfile:
        outfile.writelines([
            f'classes={len(CLASSES)}\n',
            f'train=../../dataset/{out_folder}/naruto_train.txt\n',
            f'valid=../../dataset/{out_folder}/naruto_test.txt\n',
            f'names=../../dataset/{out_folder}/naruto.names\n',
            'backup=backup/\n'
            'eval=coco\n'])

if __name__ == '__main__':
    convert_to_coco()
