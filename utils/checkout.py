import os
import os.path as osp
import pdb 

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


base_dir = './Annotations'
xml_dir = os.listdir(base_dir)

CLASS = ['ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs']


def check_class_name(root, class_name, file_name):
    
    error_name = []
    error_flag = False

    for obj in root.findall('object'):
        # pdb.set_trace()
        if obj[0].text not in class_name:
            error_flag = True
            error_name.append(obj[0].text)
            print('find error in ' + file_name)
    
    return error_flag, error_name


def check_size(root, file_name):
    error_name = []
    error_flag = False

    for obj in root.findall('size'):
        # pdb.set_trace()
        if int(obj.find('depth').text) != 3:
            error_flag = True
            print('find error in ' + file_name)
    
    return error_flag, error_name


def rename_filename(tree, root, xml_name, img_dir = './JPEGImages', img_form='.jpg'):

    img_name = osp.join(img_dir, xml_name.split('.')[0] + img_form)    
    try:
        root.find('path').text = img_name
    except:
        ele = Element('path')
        ele.text = img_name
        root.append(ele)
    
    tree.write(osp.join('annotations', xml_name))
    
    return True




def main():
    for xml in xml_dir:
        xml_path = osp.join(base_dir, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # error, _ = check_class_name(root, CLASS, xml)
        error, _ = check_size(root, xml)
        rename_filename(tree, root, xml)

if __name__ == '__main__':
    main()