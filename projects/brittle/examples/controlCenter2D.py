import subprocess

#TEST CONTROL CENTER
test1 = [1]

sigmaF = 140 * 10**2
cf = 2000 / 10**5 / 2

for i in range(len(test1)):
    if test1[i]:
        runCommand = 'python3 circleCrusher.py ' + str(sigmaF) + ' ' + str(cf)
        subprocess.call([runCommand], shell=True)