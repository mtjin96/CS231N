import matplotlib.pyplot as plt

with open('log/raw_image_2500.log') as f:
	content = f.readlines()
	loss = []
	corr = []
	CCC = []
	for line in content:
		if 'Evaluation' in line:
			loss.append(float(line.split(':')[3].split('\t')[0].strip()))
			# corr.append(line.split(':')[4].split('\t')[0].strip())
			CCC.append(float(line.split(':')[5].split('\t')[0].strip()))

loss = loss[:600]
CCC = CCC[:600]
epoch = []
for i in range(len(loss)):
	epoch.append(i)

plt.plot(epoch, CCC)
# plt.ylim([0.03,0.12])
plt.ylim([-0.02, 0.18])
plt.xlabel('Epoch')
plt.ylabel('CCC')
plt.show()
			
# with open('logs/multi_mse.log') as f:
#     content = f.readlines()
#     loss2 = []
#     corr2 = []
#     CCC2 = []
#     for line in content:
#     	if 'Evaluation' in line:
# 	    	loss2.append(line.split(':')[3].split('\t')[0].strip())
# 	    	corr2.append(line.split(':')[4].split('\t')[0].strip())
# 	    	CCC2.append(line.split(':')[5].split('\t')[0].strip())


# with open('logs/AR1.log') as f:
#     content = f.readlines()
#     loss3 = []
#     corr3 = []
#     CCC3 = []
#     for line in content:
#     	if 'Evaluation' in line:
# 	    	loss3.append(line.split(':')[3].split('\t')[0].strip())
# 	    	corr3.append(line.split(':')[4].split('\t')[0].strip())
# 	    	CCC3.append(line.split(':')[5].split('\t')[0].strip())


# with open('logs/TF1.log') as f:
#     content = f.readlines()
#     loss4 = []
#     corr4 = []
#     CCC4 = []
#     for line in content:
#     	if 'Evaluation' in line:
# 	    	loss4.append(line.split(':')[3].split('\t')[0].strip())
# 	    	corr4.append(line.split(':')[4].split('\t')[0].strip())
# 	    	CCC4.append(line.split(':')[5].split('\t')[0].strip())


# with open('loss_ccc_mse.csv','wb') as file:
# 	for i in range(len(loss)):
# 		file.write(str(i+1) + ', ' + loss[i] + ', ' +  loss2[i])
# 		file.write('\n')


# with open('w10o5corr.csv','wb') as file:
#     for i in range(len(corr)):
#         file.write(str(i+1) + ', ' + corr[i] + ', ' +  corr2[i] + ', ' +  corr3[i] + ', ' +  corr4[i])
#         file.write('\n')
 

# with open('w10o5CCC.csv','wb') as file:
#     for i in range(len(CCC)):
#         file.write(str(i+1) + ', ' + CCC[i] + ', ' +  CCC2[i] + ', ' +  CCC3[i] + ', ' +  CCC4[i])
#         file.write('\n')