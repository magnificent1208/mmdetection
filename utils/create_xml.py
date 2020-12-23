from xml.dom.minidom import Document
import cv2 as cv


def create_xml(data):
    data_split = []
    cur_data = []
    for line in data[6:]:
        line, level = pre_process(line)

        if level == 2:
            if cur_data:
                data_split.append(cur_data)
            cur_data = [line]
        else:
            cur_data.append(line)

    for anno in data_split:
        img_name = anno[1][-1].split('/')[-1]
        # import pdb; pdb.set_trace()
        if anno[2][-1] == '0':
            save_xml(img_name)
            continue

        rboxes = []
        rbox = []
        for l in anno[3:]:
            if l[0] == 'label':
                if rbox:
                    rboxes.append(rbox)
                rbox = [l[1]]
            elif l[0] in ['x', 'y', 'height', 'width', 'rotation']:
                rbox.append(l[1])
        rboxes.append(rbox)
        save_xml(img_name, rboxes)
    
    return True


def save_xml(img_name, rboxes=None, root_dir='./Annotations/'):
    xml_name = img_name.split('.')[0] + '.xml'
    img_path = './JPEGImages/' + img_name
    img_cv = cv.imread(img_path)
    img_shape = img_cv.shape
        
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    filename = create_element('filename', img_name, doc)
    annotation.appendChild(filename)

    size = doc.createElement('size')
    width = create_element('width', str(img_shape[1]), doc)
    size.appendChild(width)
    height = create_element('height', str(img_shape[0]), doc)
    size.appendChild(height)
    depth = create_element('depth', str(img_shape[2]), doc)
    size.appendChild(depth)
    annotation.appendChild(size)

    if rboxes:
        for rbox in rboxes:
            obj = doc.createElement('object')
            label = create_element('label', rbox[0], doc)
            obj.appendChild(label)
            dif = create_element('difficult', '0', doc)
            obj.appendChild(dif)
            bndbox = doc.createElement('bndbox')
            x = create_element('x', rbox[1], doc)
            bndbox.appendChild(x)
            y = create_element('y', rbox[2], doc)
            bndbox.appendChild(y)
            w = create_element('w', rbox[3], doc)
            bndbox.appendChild(w)
            h = create_element('h', rbox[4], doc)
            bndbox.appendChild(h)
            r = create_element('r', rbox[5], doc)
            bndbox.appendChild(r)
            obj.appendChild(bndbox)
            annotation.appendChild(obj)
    
    with open(root_dir + xml_name, 'wb+') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    # print('save successful')

    return True


def create_element(name, value, doc):
    obj = doc.createElement(name)
    obj_text = doc.createTextNode(value)
    obj.appendChild(obj_text)
    return obj

    
def pre_process(line):
    num_space = 0
    for x in line:
        if x == ' ':
            num_space += 1
        else:
            break
    line = line.replace(' ', '').replace('\n', '').split(':')
    return line, num_space


if __name__ == '__main__':
    with open('joints.lbprj') as f:
        data = f.readlines()
    
    create_xml(data)
