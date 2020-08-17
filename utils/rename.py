import os


def file_rename():

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

if __name__ == '__main__':
    file_rename()