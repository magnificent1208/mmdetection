import os

file_path_xml = 'annotations'
file_path_txt = 'test.txt'

pathDir = os.listdir(os.path.join(file_path, file_path_xml))

with open(os.path.join(file_path, file_path_txt), 'w') as f:
    for path in pathDir:
        name = path.split('.')[0]
        f.write(str(name) + '\n')
    
    f.close()