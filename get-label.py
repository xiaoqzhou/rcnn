import os
import sys
import math

Sum-epoch =25
def generate():
    listfile = './data/task3_4label_test.list'
    path1 = './Result/'
    path2 = './Label/'
    strmat = 'rnn_label.txt'
    strmat1 = 'rnn_label.txt'
    testindex = []
    with open(listfile,'r') as fin:
    	for line in fin:
    		line = line.strip('\n')
    		index = line.strip('\r').split('\t')
    		testindex.append(index[2])
    for i in range(Sum-epoch):
    	epoch = i
    	filename1 = path1 + str(epoch)+ strmat
    	filename2 = path2
    	label = []
    	filename2 = filename2 + str(epoch+1) + strmat1
    	ofile = open(filename2,'w') 
    	with open(filename1,'r') as fin:
    		for line in fin:
    			vec = line.strip('\n').split()
    			v = float(vec[0])
    			if v==0:
    				label.append('Good')
    			elif v==1:
    				label.append('Potential')
    			elif v==2:
    				label.append('Bad')	
    			elif v==3:
    				label.append('Dialogue')
    	for j in range(len(testindex)):
    		ofile.write(testindex[j]+'\t'+label[j]+'\n')
    	ofile.close()
if __name__ == '__main__':
    generate()

