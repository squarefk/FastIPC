import subprocess

#TEST CONTROL CENTER
test1 = [1]

GfList = [1, 10, 100]
sigmaFList = [1, 10, 100]
dMinList = [0.25, 0.5, 0.75]

#target output1: "../output/mode1Fracture/brittle.ply"
outputBase = "../output/mode1Fracture2D/Gf"

for i in range(len(test1)):
    if test1[i]:
        for Gf in GfList:
            for sigmaF in sigmaFList:
                for dMin in dMinList:
                    outputPath = outputBase + str(Gf) + "_sigmaF" + str(sigmaF) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                    outputPath2 = outputPath + "/brittle_i.ply"
                    
                    mkdirCommand = 'mkdir ' + outputPath
                    subprocess.call([mkdirCommand], shell=True)

                    outputPath += "/brittle_p.ply"
                    
                    runCommand = 'python3 mode1Fracture2D.py ' + str(Gf) + ' ' + str(sigmaF) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                    subprocess.call([runCommand], shell=True)