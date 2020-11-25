import subprocess

#TEST CONTROL CENTER
demoSetA = [0, 1, 0, 0] #notchedMode1, circleCrusher, ringDrop, bulletShoot

#TEST CONTROL SUBSTATION
notchedMode1Tests = [1]
circleCrusherTests = [1]
ringDropTests = [1]
bulletShootTests = [1]

if demoSetA[0]:
    for test in notchedMode1Tests:
        
        if test:

            GfList = [1, 10, 100]
            sigmaFList = [1, 10, 100]
            dMinList = [0.25, 0.5, 0.75]
            outputBase = "../output/mode1Fracture2D_Wedging/Gf"

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

if demoSetA[1]:
    for test in circleCrusherTests:
        
        if test:

            #best setup so far is Gf = 0.01, sigmaF = 40, dMin = 0.25
            GfList = [0.005, 0.01, 0.2]
            sigmaFList = [30, 40, 50]
            dMinList = [0.1, 0.25, 0.5]
            outputBase = "../output/circleCrusher2D_Wedging/Gf"

            for Gf in GfList:
                for sigmaF in sigmaFList:
                    for dMin in dMinList:
                        outputPath = outputBase + str(Gf) + "_sigmaF" + str(sigmaF) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                        outputPath2 = outputPath + "/brittle_i.ply"
                        
                        mkdirCommand = 'mkdir ' + outputPath
                        subprocess.call([mkdirCommand], shell=True)

                        outputPath += "/brittle_p.ply"
                        
                        runCommand = 'python3 circleCrusher.py ' + str(Gf) + ' ' + str(sigmaF) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                        subprocess.call([runCommand], shell=True)

if demoSetA[2]:
    for test in ringDropTests:
        
        if test:

            GfList = [1]
            sigmaFList = [1]
            dMinList = [0.25]
            outputBase = "../output/ringDrop2D_Wedging/Gf"

            for Gf in GfList:
                for sigmaF in sigmaFList:
                    for dMin in dMinList:
                        outputPath = outputBase + str(Gf) + "_sigmaF" + str(sigmaF) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                        outputPath2 = outputPath + "/brittle_i.ply"
                        
                        mkdirCommand = 'mkdir ' + outputPath
                        subprocess.call([mkdirCommand], shell=True)

                        outputPath += "/brittle_p.ply"
                        
                        runCommand = 'python3 ringDrop2D.py ' + str(Gf) + ' ' + str(sigmaF) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                        subprocess.call([runCommand], shell=True)


if demoSetA[3]:
    for test in bulletShootTests:
        
        if test:

            GfList = [1]
            sigmaFList = [1]
            dMinList = [0.25]
            outputBase = "../output/bulletShoot2D_Wedging/Gf"

            for Gf in GfList:
                for sigmaF in sigmaFList:
                    for dMin in dMinList:
                        outputPath = outputBase + str(Gf) + "_sigmaF" + str(sigmaF) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                        outputPath2 = outputPath + "/brittle_i.ply"
                        
                        mkdirCommand = 'mkdir ' + outputPath
                        subprocess.call([mkdirCommand], shell=True)

                        outputPath += "/brittle_p.ply"
                        
                        runCommand = 'python3 bulletShoot2D.py ' + str(Gf) + ' ' + str(sigmaF) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                        subprocess.call([runCommand], shell=True)