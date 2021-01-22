import os


def filetype_rename():

    path = './JPEGImages'
    filelist = os.listdir(path)  
    for files in filelist:   

        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
                continue
        filename = os.path.splitext(files)[0]     
        filetype = os.path.splitext(files)[1]
        if filetype == '.JPG':
            Newdir = os.path.join(path, filename + '.jpg')
            os.rename(Olddir, Newdir)   
    return True


def file_rename():
    
    img_path = './JPEGImages'
    xml_path = './Annotations'
    filelist = os.listdir(img_path)

    num = 4916
    for files in filelist:   

        Olddir = os.path.join(img_path, files)
        file_name, file_type = os.path.splitext(files)

        if file_name[:2] == '00':
            continue

        # if file_type != '.jpg':
        #     print('error')
        #     import pdb; pdb.set_trace()
        Newdir = os.path.join(img_path, encode_num(num) + '.jpg')

        old_xml_dir = os.path.join(xml_path, file_name + '.xml')
        new_xml_dir = os.path.join(xml_path, encode_num(num) + '.xml')
        os.rename(Olddir, Newdir)
        os.rename(old_xml_dir, new_xml_dir)
        num += 1
    return True


def encode_num(num, n=6):
    return str('0' * (n - len(str(num))) + str(num)) 


if __name__ == '__main__':
    file_rename()