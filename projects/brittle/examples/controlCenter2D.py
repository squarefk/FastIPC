import subprocess

#TEST CONTROL CENTER
demoSetA = [1, 0, 0, 0] #notchedMode1, circleCrusher, ringDrop, bulletShoot

#TEST CONTROL SUBSTATION
notchedMode1Tests = [0,1]
circleCrusherTests = [1]
ringDropTests = [1]
bulletShootTests = [1]

#Command Line Parameter Order:
# python3 script.py percentStretch Gf dMin outputPath1 outputPath2
# python3 script.py percentStretch eta zeta dMin outputPath1 outputPath2
if demoSetA[0]:        
    if notchedMode1Tests[0]:

        GfList = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        percentList = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        dMinList = [0.1, 0.25, 0.4]
        outputBase = "../output/mode1Fracture2D_Wedging/p"

        for i in range(len(GfList)):
            Gf = GfList[i]
            p = percentList[i]
            for dMin in dMinList:
                outputPath = outputBase + str(p) + "_Gf" + str(Gf) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                outputPath2 = outputPath + "/brittle_i.ply"
                
                mkdirCommand = 'mkdir ' + outputPath
                subprocess.call([mkdirCommand], shell=True)

                outputPath += "/brittle_p.ply"
                
                runCommand = 'python3 mode1Fracture2D.py ' + str(p) + ' ' + str(Gf) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                subprocess.call([runCommand], shell=True)

    if notchedMode1Tests[1]:

        p = 1e-2
        etaList = [1e-5]
        zetaList = [1e4, 1e5]
        dMin = 0.25
        outputBase = "../output/mode1Fracture2D_Wedging/p"

        for eta in etaList:
            for zeta in zetaList:
                outputPath = outputBase + str(p) + "_eta" + str(eta)+ "_zeta" + str(zeta) + "_dMin" + str(int(dMin*100))
                outputPath2 = outputPath + "/brittle_i.ply"
                
                mkdirCommand = 'mkdir ' + outputPath
                subprocess.call([mkdirCommand], shell=True)

                outputPath += "/brittle_p.ply"
                
                runCommand = 'python3 mode1Fracture2D.py ' + str(p) + ' ' + str(eta) + ' ' + str(zeta) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                subprocess.call([runCommand], shell=True)

# if demoSetA[1]:
#     for test in circleCrusherTests:
        
#         if test:

#             #best setup so far is Gf = 0.01, sigmaF = 50, dMin = 0.25
#             GfList = [0.01]
#             sigmaFList = [40, 50, 60]
#             #dMinList = [0.05, 0.15, 0.25]
#             dMinList = [0.3, 0.4, 0.5]
#             outputBase = "../output/circleCrusher2D_Wedging/Gf"

#             for Gf in GfList:
#                 for sigmaF in sigmaFList:
#                     for dMin in dMinList:
#                         outputPath = outputBase + str(Gf) + "_sigmaF" + str(sigmaF) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
#                         outputPath2 = outputPath + "/brittle_i.ply"
                        
#                         mkdirCommand = 'mkdir ' + outputPath
#                         subprocess.call([mkdirCommand], shell=True)

#                         outputPath += "/brittle_p.ply"
                        
#                         runCommand = 'python3 circleCrusher.py ' + str(Gf) + ' ' + str(sigmaF) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
#                         subprocess.call([runCommand], shell=True)

if demoSetA[2]:
    for test in ringDropTests:
        
        if test:

            p = 1e-5
            dMin = 0.25
            minGf = 2.48e-7
            maxGf = 3.04e-7
            diff = maxGf - minGf
            iters = 10
            deltaGf = diff / float(iters)

            # Gf = 1e-3
            # dMinList = 0.25
            # minPercent = 2.47502325e-3
            # maxPercent = 2.4750233e-3
            # diff = maxPercent - minPercent
            # iters = 100
            # deltaP = diff / float(iters)
            
            outputBase = "../output/ringDrop2D_Wedging/p"

            for i in range(iters):
                Gf = minGf + (deltaGf * i)
                outputPath = outputBase + str(p) + "_Gf" + str(Gf) + "_dMin" + str(dMin) #+ "/brittle_p.ply"
                outputPath2 = outputPath + "/brittle_i.ply"
                
                mkdirCommand = 'mkdir ' + outputPath
                subprocess.call([mkdirCommand], shell=True)

                outputPath += "/brittle_p.ply"
                
                runCommand = 'python3 simpleRingDrop2D.py ' + str(p) + ' ' + str(Gf) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                subprocess.call([runCommand], shell=True)


if demoSetA[3]:
    for test in bulletShootTests:
        
        if test:

            GfList = [5e-4]
            percentList = [1e-3]
            dMinList = [0.25, 0.4]
            outputBase = "../output/bulletShoot2D_Wedging/p"

            for Gf in GfList:
                for p in percentList:
                    for dMin in dMinList:
                        outputPath = outputBase + str(p) + "_Gf" + str(Gf) + "_dMin" + str(int(dMin*100)) #+ "/brittle_p.ply"
                        outputPath2 = outputPath + "/brittle_i.ply"
                        
                        mkdirCommand = 'mkdir ' + outputPath
                        subprocess.call([mkdirCommand], shell=True)

                        outputPath += "/brittle_p.ply"
                        
                        runCommand = 'python3 dirichletBulletShoot2D.py ' + str(p) + ' ' + str(Gf) + ' ' + str(dMin) + ' ' + outputPath + ' ' + outputPath2
                        subprocess.call([runCommand], shell=True)