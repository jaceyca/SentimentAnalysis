import numpy as np
correct = 0
mismatchfg = 0
mismatchfh = 0
mismatchgh = 0
mismatchfgh = 0
diff_indicesfg = []
diff_indicesgh = []
diff_indicesfh = []


with open("submissions/togPred.txt") as f:
    with open("submissions/votingsubmission.txt") as g:
        for i in range(10002):
            line1 = f.readline()
            line2 = g.readline()
            if line1 == line2:
                correct += 1
            else:
                mismatchfgh += 1

print (correct, mismatchfgh)
        # with open("submissions/qewitt.txt") as h:
        #     for i in range(10002):
        #         line1 = f.readline()
        #         line2 = g.readline()
        #         line3 = h.readline()
        #         if line1 == line2 and line2 == line3 and line1 == line3:
        #             correct += 1
                
        #         if line1 != line2 and line2 != line3 and line1 != line3:
        #             mismatchfgh += 1
                
        #         if line1 != line2:
        #             mismatchfg += 1
        #             diff_indicesfg.append(i)
                
        #         if line2 != line3:
        #             mismatchgh += 1
        #             diff_indicesgh.append(i)
                
        #         if line1 != line3:
        #             mismatchfh += 1
        #             diff_indicesfh.append(i)

# print("correct:", correct, "mismatchfgh:", mismatchfgh, 'mismatchfg:', mismatchfg,
#     "mismatchgh:", mismatchgh, "mismatchfh:", mismatchfh)
# print(diff_indicesfg, diff_indicesgh, diff_indicesfh)
'''
voting = np.loadtxt("submissions/votingsubmission.txt", skiprows=1, delimiter=',', usecols=(1)).astype(int)
svctxt = np.loadtxt("submissions/SVCgamma1C2.txt", skiprows=1, delimiter=',', usecols=(1)).astype(int)
neuralnet = np.loadtxt("submissions/qewitt.txt", skiprows=1, delimiter=',', usecols=(1)).astype(int)

# seq_predictions, ridgePred, log
# consideration of 3 scores
togPred = []
for index, element in enumerate(voting):
    if (2.0 * element + svctxt[index] + neuralnet[index]) > 1.33333333333333:
        togPred.append(1)
    else: 
        togPred.append(0)

# print (togPred, len(togPred))
with open("submissions\\%s" % "togPred2.txt", "w") as f:
    f.write("Id,Prediction\n")
    for Id, prediction in enumerate(togPred, 1):
        string = str(Id) + ',' + str(prediction) + '\n'
        f.write(string)



# 8: 9453 547, 9409 593, 9517 485
# 9: 9548 452, 9506 496
# 10: 9495 505, 9534 468, 9367 635, 9568 434

# 9470 532
# correct2 = 0
# mismatch2 = 0
# diff_indices2 = []

# neuralnet = np.loadtxt("submissions/qewitt.txt", skiprows=1, delimiter=',', usecols=(1)).astype(int)
# print (neuralnet, len(neuralnet))

# for i in range(len(diff_indices)):
#     if diff_indices[i] != 


'''