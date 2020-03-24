

# basic imports
import numpy as np


'''
Reads data from an experimental design file.

f: the file to be parsed
'''


def readExperimentalDesignFile():

    # open file for reading
    # note: the experimental design file is ALWAYS in the experiment's 'base' folder
    experimentalDesignFile=open('./experimentalDesign.txt','r')
    
    # parse file contents
    for l in experimentalDesignFile:
        
        # look for environment definitions:
        if 'environmentName' in l:
            l=l.split('=')
            environmentName=l[1].rstrip('\n')
            print('*** environment name: %s ***' % environmentName)
        
        # look for topology definitions:
        if 'topologyType' in l:
            l=l.split('=')
            topologyType=l[1].rstrip('\n')
            print('topology type: %s' % topologyType)
            
            
        if 'topologyGoalNodes' in l:
            goalNodesList=[]
            l=l.split('=')
            topologyGoalNodeStr=l[1].rstrip('\n')
            topologyGoalNodes=topologyGoalNodeStr.split(',')
            for n in range(len(topologyGoalNodes)):
                goalNodesList+=[int(topologyGoalNodes[n])]
            topologyGoalNodes=goalNodesList
            print('topology goal nodes: ', topologyGoalNodes)
            
        
        if 'topologyStartNodes' in l:
            startNodesList=[]
            l=l.split('=')
            topologyStartNodesStr=l[1].rstrip('\n')
            topologyStartNodes=topologyStartNodesStr.split(',')
            for n in range(len(topologyStartNodes)):
                startNodesList+=[int(topologyStartNodes[n])]
            topologyStartNodes=startNodesList
            print('topology start nodes: ',topologyStartNodes)
            
            
        if 'topologyGridSize' in l:
            l=l.split('=')
            topologyGridSize=float(l[1].rstrip('\n'))
            print('topology grid size: %f' % topologyGridSize)
        
        
        
        if 'topologyCliqueSize' in l:
            l=l.split('=')
            topologyCliqueSize=int(l[1].rstrip('\n'))
            print('topology clique size: %d' % topologyCliqueSize)
        
        
        # look for 'startFearAcquisition'
        if 'startFearAcquisition' in l:
            l=l.split('=')
            startFearAcquisition=int(l[1].rstrip('\n'))
            print('startFearAcquisition: %d' % startFearAcquisition)
        
        # look for 'startFearExtinction'
        if 'startFearExtinction' in l:
            l=l.split('=')
            startFearExtinction=int(l[1].rstrip('\n'))
            print('startFearExtinction: %d' % startFearExtinction)
            
        # look for startReturnOfFearTest
        if 'startReturnOfFearTest' in l:
            l=l.split('=')
            startReturnOfFearTest=int(l[1].rstrip('\n'))
            print('startReturnOfFearTest: %d' % startReturnOfFearTest)
            
        # look for experimentDuration
        if 'experimentDuration' in l:
            l=l.split('=')
            experimentDuration=int(l[1].rstrip('\n'))
            print('experimentDuration: %d' % experimentDuration)
        
        
        # look for nrOfRuns:
        if 'nrOfRuns' in l:
            l=l.split('=')
            nrOfRuns=int(l[1].rstrip('\n'))
            print('nrOfRuns: %d' % nrOfRuns)
        
        
    # create a dictionary containing the retrieved variables
    ret=dict()
    
    if 'environmentName' in locals():
        ret['environmentName']=environmentName
    
    if 'topologyType' in locals():
        ret['topologyType']=topologyType
    
    if 'topologyGoalNodes' in locals():
        ret['topologyGoalNodes']=topologyGoalNodes
    
    if 'topologyStartNodes' in locals():
        ret['topologyStartNodes']=topologyStartNodes
    
    if 'topologyGridSize' in locals():
        ret['topologyGridSize']=topologyGridSize
    
    if 'topologyGoalNode' in locals():
        ret['topologyGoalNode']=topologyGoalNode
    
    if 'topologyCliqueSize' in locals():
        ret['topologyCliqueSize']=topologyCliqueSize
    
    if 'startFearAcquisition' in locals():
        ret['startFearAcquisition']=startFearAcquisition
    
    if 'startFearExtinction' in locals():
        ret['startFearExtinction']=startFearExtinction
    
    if 'startReturnOfFearTest' in locals():
        ret['startReturnOfFearTest']=startReturnOfFearTest
    
    if 'experimentDuration' in locals():
        ret['experimentDuration']=experimentDuration
    
    if 'nrOfRuns' in locals():
        ret['nrOfRuns']=nrOfRuns
    
    # return the dictionary
    return ret
    
