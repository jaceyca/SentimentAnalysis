correct = 0
mismatch = 0

with open("submissions/NeuralNetworkSubmission.txt") as f:
    with open("submissions/SVCgamma1C2.txt") as g:
        for i in range(10002):
            line1 = f.readline()
            line2 = g.readline()
            if line1 == line2:
                correct += 1
            else:
                mismatch += 1

print("correct:", correct, "mismatch:", mismatch)

# 8: 9453 547, 9409 593, 9517 485
# 9: 9548 452, 9506 496
# 10: 9495 505, 9534 468, 9367 635, 9568 434

# 9470 532