with open('d:/project/data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 1523 (index 1522)
if len(lines) > 1522:
    lines[1522] = lines[1522].replace('"娌℃湁绗﹀悎鏉′欢鐨勮褰?, None', '"没有符合条件的记录", None')

with open('d:/project/data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new2.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed line 1523')
