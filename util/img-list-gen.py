# Created by wz on 17-3-23.
# encoding=utf-8
import os


def main():
    imgs = os.listdir('../pic')
    list = "../list.txt"
    f = open(list, 'w')
    count = 0
    for i in imgs:
        _, lables, _ = i.split('_')
        name = str(count) + '.png'
        os.rename('../pic/'+i, '../pic1/'+name)
        f.write(name+' '+lables+'\n')
        count+=1
    f.close()


if __name__ == '__main__':
    main()
