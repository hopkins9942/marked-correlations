#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:29:28 2021

@author: matthew

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime
import os
from scipy.stats import kstest
#import time

#%% defs

h = 0.7
delta_rp = 0.005/h # Mpc
max_rp = 0.1/h
max_pi = 40/h # Mpc # skibba was 40/h, should possibly do same for final run. DAis and peebles 83 used 2500kms^s/H0 = 36Mpc, similar
rng = np.random.default_rng() # random number generator for scrambling marks, keeping constant seed while developing
savepath = ""
markList = ["count", "Merger", "inclusive merger", "Quenching", "SFR", "Stellar mass", "g-r colour ratio", "u-r colour ratio", "g-i colour ratio", "u spectral luminosity", "g spectral luminosity", "r spectral luminosity", "i spectral luminosity"]
Nmarks = len(markList)

# globalindices = np.zeros((100000,3)) # first value is bin, remainder are indices of pair constituents
# globalcount = 0


def rp(distSum, raDiff, dec1, dec2):
    """
    Gives perpendicular difference between two points, as defined in Peebles?
    """
    arg = np.sin((dec1-dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(raDiff/2)**2
    return (distSum)*np.sqrt(arg/(1-arg))

class Node:
    
    leafMinPoints = 10
    leafMinPerpSize = max_rp/10
    leafMinParaSize = max_pi/10
    #TODO check these considering size of search area -10, size/10 works well though
    
    def __init__(self, nodeIndices, idnum):
        # bound = np.array([[dmin,dmax],[decmin,decmax],[ramin,ramax]])
        #NOTE now done entirely with indices
        self.nodeIndices = nodeIndices
        self.idnum = idnum
        self.bounds = np.array([[min(distArray[nodeIndices]), max(distArray[nodeIndices])],
                                [min( decArray[nodeIndices]), max( decArray[nodeIndices])],
                                [min(  raArray[nodeIndices]), max(  raArray[nodeIndices])]])
        dist_split = ((self.bounds[0,1]**3 + self.bounds[0,0]**3)/2)**(1/3)
        dec_split = np.arcsin((np.sin(self.bounds[1,1]) + np.sin(self.bounds[1,0]))/2)
        ra_split = (self.bounds[2,1] + self.bounds[2,0])/2
        self.splitArray = np.array([dist_split, dec_split, ra_split])
        
        dist_size = (self.bounds[0,1]-self.bounds[0,0])
        dec_size = dist_split*(self.bounds[1,1]-self.bounds[1,0])
        ra_size = dist_split*np.cos(dec_split)*(self.bounds[2,1] - self.bounds[2,0])
        self.sizeArray = np.array([dist_size, dec_size, ra_size])
        
        self.numGalaxies = np.count_nonzero(self.nodeIndices)
        if (np.max(self.sizeArray[1:])<=self.leafMinPerpSize)or(self.sizeArray[0]<=self.leafMinParaSize)or(self.numGalaxies<=self.leafMinPoints):
            self.isLeaf=True
        else:
            self.isLeaf=False
            
    def leftChild(self):
        if self.isLeaf:
            return None
        else:
            maxSizeIndex = np.argmax(self.sizeArray)
            if maxSizeIndex==0:
                # ie dist dimention longest
                leftIndices = np.logical_and(distArray < self.splitArray[0], self.nodeIndices)
                # here leftIndices gives boolean array with True at indices where corresponding galaxies are in range of left child ie in bounds of current node and correct half.
            elif maxSizeIndex==1:
                # ie dec dimention longest
                leftIndices =  np.logical_and(decArray < self.splitArray[1], self.nodeIndices)
            elif maxSizeIndex==2:
                # ie ra dimention longest
                leftIndices =  np.logical_and(raArray < self.splitArray[2], self.nodeIndices)
            return Node(leftIndices, 2*self.idnum+1)
        
    def rightChild(self):
        if self.isLeaf:
            return None
        else:
            maxSizeIndex = np.argmax(self.sizeArray)
            if maxSizeIndex==0:
                # ie dist dimention longest
                rightIndices = np.logical_and(distArray >= self.splitArray[0], self.nodeIndices)
            elif maxSizeIndex==1:
                # ie dec dimention longest
                rightIndices =  np.logical_and(decArray >= self.splitArray[1], self.nodeIndices)
            elif maxSizeIndex==2:
                # ie ra dimention longest
                rightIndices =  np.logical_and(raArray >= self.splitArray[2], self.nodeIndices)                
            return Node(rightIndices, 2*self.idnum+2)
        

def minrp(node1,node2):
    "gives minimum rp distance between node1 and node2 (tested)"
    # rp minimised by choosing lowest radial distance faces of nodes, then finding minimum angle between faces
    decDiff =              0.5*(abs(node1.bounds[1,1]+node1.bounds[1,0]-node2.bounds[1,1]-node2.bounds[1,0])
                                 - (node1.bounds[1,1]-node1.bounds[1,0]+node2.bounds[1,1]-node2.bounds[1,0]))

    raDiff  = min(          0.5*abs(node1.bounds[2,1]+node1.bounds[2,0]-node2.bounds[2,1]-node2.bounds[2,0]),
                  2*np.pi - 0.5*abs(node1.bounds[2,1]+node1.bounds[2,0]-node2.bounds[2,1]-node2.bounds[2,0])
                  )        -0.5  * (node1.bounds[2,1]-node1.bounds[2,0]+node2.bounds[2,1]-node2.bounds[2,0])
    
    if decDiff<=0:
        if raDiff<=0:
            #overlap in both,
            return 0
        else:
            # ie overlap in dec, min distis given by dist between points at nearest radial distance, with most extreme dec in dec overlap
            absdec = max(abs(np.sort([node1.bounds[1,0], node1.bounds[1,1], node2.bounds[1,0], node2.bounds[1,1]])[1:3]))
            #sort means middle two ([1:3]) give ends of overlap, max(abs()) gives most extreme. this gives absolute value of dec of nearest points, but this changes nothing in rp
            return rp(node1.bounds[0,0]+node2.bounds[0,0],raDiff, absdec, absdec)
    else:
        decToUse = np.sort([node1.bounds[1,0], node1.bounds[1,1], node2.bounds[1,0], node2.bounds[1,1]])[1:3]
        #definitly no overlap in dec means middle 2 decs are nearesr edges, order irrrelevant
        if raDiff<=0:
            #overlap in ra only
            
            return rp(node1.bounds[0,0]+node2.bounds[0,0], 0, decToUse[0], decToUse[1])
        else:
            # no overlaps
            return rp(node1.bounds[0,0]+node2.bounds[0,0], raDiff, decToUse[0], decToUse[1])
        
def maxrp(node1,node2):
    "gives max rp distance between node1 and node2"
    # rp minimised by choosing lowest radial distance faces of nodes, then finding minimum angle between faces
    cornerDists = [rp(node1.bounds[0,1]+node2.bounds[0,1], node1.bounds[2,0]-node2.bounds[2,1], node1.bounds[1,0], node2.bounds[1,1]),
                   rp(node1.bounds[0,1]+node2.bounds[0,1], node1.bounds[2,0]-node2.bounds[2,1], node1.bounds[1,1], node2.bounds[1,0]),
                   rp(node1.bounds[0,1]+node2.bounds[0,1], node1.bounds[2,1]-node2.bounds[2,0], node1.bounds[1,0], node2.bounds[1,1]),
                   rp(node1.bounds[0,1]+node2.bounds[0,1], node1.bounds[2,1]-node2.bounds[2,0], node1.bounds[1,1], node2.bounds[1,0]),
                   ]
    return max(cornerDists)

def minpidiff(node1, node2):
    return 0.5*(abs(node1.bounds[0,1]+node1.bounds[0,0]-node2.bounds[0,1]-node2.bounds[0,0])
                                 - (node1.bounds[0,1]-node1.bounds[0,0]+node2.bounds[0,1]-node2.bounds[0,0]))
def maxpidiff(node1,node2):
    return max(abs(node1.bounds[0,0]-node2.bounds[0,1]), abs(node1.bounds[0,1]-node2.bounds[0,0]))

                
def SlowBinCount(node1,node2, binEdges, marks2cross):
    """
    Calculates weighted pair sums (not normalised) in bins between two nodes. Works for nodes equal or different, at any seperation.
    Used after DualTreeBinCount has trimmed nodes outside of range for max speed.
    V7 update now sums directly over node indices, calculating pairs sums, squared sums, and given crosses. Plus scrambled version:
        i and j index galaxies, k indexes bin, m1 and m2 index marks with cross product index m = Tm1+m2
    """
    global globalcount
    
    count = np.zeros((2*Nmarks+len(marks2cross), len(binEdges)-1, 2))
    if node1.idnum!=node2.idnum:
        for i in np.arange(Ngalaxies)[node1.nodeIndices]:
            for j in np.arange(Ngalaxies)[node2.nodeIndices]:
                if (abs(distArray[i]-distArray[j])<max_pi):
                    temp_rp = rp(distArray[i]+distArray[j], raArray[i]-raArray[j], decArray[i], decArray[j])
                    for k in range(len(binEdges)-1):
                        if (temp_rp<binEdges[k+1]):
                            # if (i not in globalindices):
                            #     globalindices[globalcount,:] = [k,i]
                            #     globalcount+=1
                            # if (j not in globalindices):
                            #     globalindices[globalcount,:] = [k,j]
                            #     globalcount+=1
                            
                            # if (k==0)or(k==9)or(k==19):
                            #     globalindices[globalcount,:] = [k,i,j]
                            #     globalcount+=1
                            
                            for l in range(0,Nmarks):
                                count[l,k,0] += markArray[i,l]*markArray[j,l]
                                count[l,k,1] += (markArray[scrambleIndices[i,l,:],l]*markArray[scrambleIndices[j,l,:],l]).mean()
                                count[Nmarks+l,k,0] += (markArray[i,l]*markArray[j,l])**2
                                count[Nmarks+l,k,1] += ((markArray[scrambleIndices[i,l,:],l]*markArray[scrambleIndices[j,l,:],l])**2).mean()
                            for l in range(len(marks2cross)):
                                count[(2*Nmarks + l), k, 0] += (markArray[i,marks2cross[l][0]]
                                                                *markArray[j,marks2cross[l][0]]
                                                                *markArray[i,marks2cross[l][1]]
                                                                *markArray[j,marks2cross[l][1]])
                                count[(2*Nmarks + l), k, 1] += (markArray[scrambleIndices[i,marks2cross[l][0],:],marks2cross[l][0]]
                                                                *markArray[scrambleIndices[j,marks2cross[l][0],:],marks2cross[l][0]]
                                                                *markArray[scrambleIndices[i,marks2cross[l][1],:],marks2cross[l][1]]
                                                                *markArray[scrambleIndices[j,marks2cross[l][1],:],marks2cross[l][1]]).mean()
                            
                            break
    else:
        for i in np.arange(Ngalaxies)[node1.nodeIndices][1:]:
            for j in np.arange(i)[node1.nodeIndices[:i]]:
                if (abs(distArray[i]-distArray[j])<max_pi): 
                    temp_rp = rp(distArray[i]+distArray[j], raArray[i]-raArray[j], decArray[i], decArray[j])
                    for k in range(len(binEdges)-1):
                        if (temp_rp<binEdges[k+1]):
                            # if (i not in globalindices):
                            #     globalindices[globalcount,:] = [k,i]
                            #     globalcount+=1
                            # if (j not in globalindices):
                            #     globalindices[globalcount,:] = [k,j]
                            #     globalcount+=1
                                
                            # if (k==0)or(k==9)or(k==8)or(k==10)or(k==19):
                            #     globalindices[globalcount,:] = [k,i,j]
                            #     globalcount+=1
                            
                            for l in range(0,Nmarks):
                                count[l,k,0] += markArray[i,l]*markArray[j,l]
                                count[l,k,1] += (markArray[scrambleIndices[i,l,:],l]*markArray[scrambleIndices[j,l,:],l]).mean()
                                count[Nmarks+l,k,0] += (markArray[i,l]*markArray[j,l])**2
                                count[Nmarks+l,k,1] += ((markArray[scrambleIndices[i,l,:],l]*markArray[scrambleIndices[j,l,:],l])**2).mean()
                            for l in range(len(marks2cross)):
                                count[(2*Nmarks + l), k, 0] += (markArray[i,marks2cross[l][0]]
                                                                *markArray[j,marks2cross[l][0]]
                                                                *markArray[i,marks2cross[l][1]]
                                                                *markArray[j,marks2cross[l][1]])
                                count[(2*Nmarks + l), k, 1] += (markArray[scrambleIndices[i,marks2cross[l][0],:],marks2cross[l][0]]
                                                                *markArray[scrambleIndices[j,marks2cross[l][0],:],marks2cross[l][0]]
                                                                *markArray[scrambleIndices[i,marks2cross[l][1],:],marks2cross[l][1]]
                                                                *markArray[scrambleIndices[j,marks2cross[l][1],:],marks2cross[l][1]]).mean()
                            
                            break
    return count
                
def DualTreeBinCount(node1,node2, binEdges, marks2cross):
    """
    Similar to Moore 2000, with weighted pair counts for pairs with rp in each bin and seperation in pi less than pimax
    Returns array of sums of weighted pair counts with entry for each mark plus two, like markArray plus cross correlations
    Splits and prunes nodes, then feeds nearby nodes into SlowBinCount to calculate sums
    """
    if (node1.idnum==node2.idnum)and(2**8-1<=node1.idnum<2**9-1):
        print(node1.idnum-2**8+1, datetime.datetime.now())#print(np.log2(1+node1.idnum), datetime.datetime.now())
    if (minrp(node1,node2)>binEdges[-1])or(minpidiff(node1,node2)>max_pi):
        return 0 # nodes seoarated by more than max binEdge, no points match
    else:
        if (maxrp(node1,node2)<binEdges[-1])and(maxpidiff(node1,node2)<max_pi):
            # nodes completely inside bin range
            return SlowBinCount(node1,node2, binEdges, marks2cross)
            
        else: # nodes not seperated at extremes by more or less than s or max_pi - split so subnodes outside can be rejected and nodes inside can be counted more quickly
            if not node1.isLeaf:
                if not node2.isLeaf:
                    if node1.numGalaxies>node2.numGalaxies:
                        return (  DualTreeBinCount(node1.leftChild(), node2,  binEdges, marks2cross)
                                + DualTreeBinCount(node1.rightChild(), node2, binEdges, marks2cross))
                    elif node1.idnum!=node2.idnum:
                        return (  DualTreeBinCount(node1, node2.leftChild(),  binEdges, marks2cross)
                                + DualTreeBinCount(node1, node2.rightChild(), binEdges, marks2cross))
                    else: # node1 is same node as node2
                        return (  DualTreeBinCount(node1.leftChild(),  node1.leftChild(), binEdges, marks2cross)
                                + DualTreeBinCount(node1.leftChild(),  node1.rightChild(), binEdges, marks2cross)
                                + DualTreeBinCount(node1.rightChild(), node1.rightChild(), binEdges, marks2cross))
                
                else: # 1 not leaf, 2 is leaf
                    return (  DualTreeBinCount(node1.leftChild(),  node2, binEdges, marks2cross)
                            + DualTreeBinCount(node1.rightChild(), node2, binEdges, marks2cross))
            
            else: # 1 is leaf
                if not node2.isLeaf:
                    return (  DualTreeBinCount(node1, node2.leftChild(),  binEdges, marks2cross)
                            + DualTreeBinCount(node1, node2.rightChild(), binEdges, marks2cross))
                else:
                    return SlowBinCount(node1, node2, binEdges, marks2cross)
    
def findChain(idnum):
    """Gets chain of children IN downwards ORDER"""
    depth = int(np.floor(np.log2(1+idnum)))
    idList = [None]*(depth+1)
    childList = [None]*(depth+1) # 0 means right child, 1 means left
    tempid = idnum
    for i in range(depth+1):
        idList[depth-i] = tempid
        childList[depth-i] = tempid%2
        if tempid%2==0:
            tempid=(tempid-2)/2
        else:
            tempid=(tempid-1)/2
    return((idList, childList))

def getNode(idnum):
    childList = findChain(idnum)[1]
    n = Node(np.full(Ngalaxies,True),0) # root
    for i in range(len(childList)-1):
        if childList[1+i]%2==0:#right
            n=n.rightChild()
        else:
            n=n.leftChild()
    return n

#%%data
with fits.open("/home/matthew/MPhys Project/Data/MatchedTable2.fits",
               memmap=True) as hdul:
    nsa_table = hdul[1].data
    hdr = hdul[1].header

Lnu_ref = 4.34538e20#ergs^-1Hz^-1
unfiltered_raArray   = np.zeros(hdr["NAXIS2"])
unfiltered_decArray  = np.zeros(hdr["NAXIS2"])
unfiltered_distArray = np.zeros(hdr["NAXIS2"])
unfiltered_markArray = np.zeros((hdr["NAXIS2"], Nmarks))
for lineNum in range(hdr["NAXIS2"]):
    if lineNum%10000==0:
        print(lineNum)
    temp_SFR = (1.4e-28)*Lnu_ref*10**((nsa_table[lineNum]['ELPETRO_ABSMAG'][1]+5*np.log10(h))/(-2.5))
    temp_SM = nsa_table[lineNum]['ELPETRO_MASS']/(h**2)
    
    unfiltered_raArray[lineNum]   = nsa_table[lineNum]['RA_1']*np.pi/180
    unfiltered_decArray[lineNum]  = nsa_table[lineNum]['DEC_1']*np.pi/180
    unfiltered_distArray[lineNum]  = (3e3 * nsa_table[lineNum]["ZDIST"])/h
    unfiltered_markArray[lineNum,:] = np.array([1,
                            nsa_table[lineNum]["merging_merger_fraction\n"],
                            nsa_table[lineNum]["merging_merger_fraction\n"]
                            + nsa_table[lineNum]["merging_major-disturbance_fraction"]
                            + nsa_table[lineNum]["merging_minor-disturbance_fraction"],
                            (temp_SM**0.76)/(temp_SFR*10**(7.64)),
                            temp_SFR,
                            temp_SM,
                            10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][4]-nsa_table[lineNum]["ELPETRO_ABSMAG"][3])/(-2.5)),
                            10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][4]-nsa_table[lineNum]["ELPETRO_ABSMAG"][2])/(-2.5)),
                            10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][5]-nsa_table[lineNum]["ELPETRO_ABSMAG"][3])/(-2.5)),
                            Lnu_ref*10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][2]+5*np.log10(h))/(-2.5)),
                            Lnu_ref*10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][3]+5*np.log10(h))/(-2.5)),
                            Lnu_ref*10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][4]+5*np.log10(h))/(-2.5)),
                            Lnu_ref*10**((nsa_table[lineNum]["ELPETRO_ABSMAG"][5]+5*np.log10(h))/(-2.5))])


#%%filtering
#filtering: note must be run immediately after data gathering - mayb not now I have unfiltered_
filterIndices = nsa_table[:]["ELPETRO_NMGY"][:,4]>=10**(9-17.77/2.5) # flux limit r<17.77
filterIndices = np.logical_and(filterIndices, np.isfinite(unfiltered_markArray[:,1])) # filters artefacts, about 540
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,3]<10**2)#q
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,3]>1e-1)#q
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,4]<1e1)#SFR
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,4]>1e-2)#SFR
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,5]<1e11)#SM 
#filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,5]>1e9)#SM #TODO this needs adjusting for vol lim changes
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,6]<10**0.5)#g-r
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,6]>10**0)#g-r
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,7]<10**1.5)#u-r
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,7]>1e0)#u-r
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,8]<10**0.6)#g-i 
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,8]>10**0)#g-i
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,9]<1e30)#ulum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,9]>1e26)#ulum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,10]<1e30)#glum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,10]>1e26)#glum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,11]<1e31)#rlum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,11]>1e26)#rlum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,12]<1e31)#ilum
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,12]>1e26)#ilum

# gives 230000/240000 galaxies, expected to only filter stripe 82? galaxies plus weird detections (old data gave 220000)

#Volume limit
volLimMr = -18 #actual magnitude assuming value of h
d_VL = 10**((-7.23-volLimMr)/5) #actual dist assuming value of h
Lr_VL = Lnu_ref*10**(-0.4*volLimMr)#actual lum assuming value of h
filterIndices = np.logical_and(filterIndices, np.logical_and(unfiltered_distArray<d_VL, unfiltered_markArray[:,11]>Lr_VL))
if savepath!="": np.save(os.path.join(savepath,"filterIndices"),filterIndices)
raArray   = unfiltered_raArray[filterIndices]
decArray  = unfiltered_decArray[filterIndices]
distArray = unfiltered_distArray[filterIndices]
markArray = unfiltered_markArray[filterIndices,:]
Ngalaxies = len(distArray)

#%%mass matched sample - note doesn't work yet if marks are logged

#sorting by mass
massSortIndices = np.argsort(markArray[:,5])
raArray = raArray[massSortIndices]
decArray = decArray[massSortIndices]
distArray = distArray[massSortIndices]
markArray = markArray[massSortIndices,:]

logMass_range = 0.5
inSample = np.full(Ngalaxies, False)
isMerger = markArray[:,1]>0.4
Nmergers = np.count_nonzero(isMerger)
#sampleIndices = np.array([], dtype="int")
massDiscrepancy = np.array([])
for i in np.arange(Ngalaxies)[isMerger]:#indices of rows of mergers
    if (i%1000==0): print(i)
    merger_logMass = np.log10(markArray[i,5])
    #closest eligible above
    j_above = i+1
    found_above = False
    while (j_above < Ngalaxies) and (np.log10(markArray[j_above,5]) < merger_logMass + logMass_range):
        if (not isMerger[j_above]) and (not inSample[j_above]):
            found_above = True
            break # now j_above equals index of closest mass galaxy above
        else:
            j_above += 1
    
    #closest eligible below
    j_below = i-1
    found_below = False
    while (j_below >= 0) and (np.log10(markArray[j_below,5]) >= merger_logMass - logMass_range):
        if (not isMerger[j_below]) and (not inSample[j_below]):
            found_below = True
            break # now j_below equals index of closest mass galaxy below
        else:
            j_below -= 1
            
    if found_above and found_below:
        if (merger_logMass-np.log10(markArray[j_below,5]) > np.log10(markArray[j_above,5])-merger_logMass):
            # mass above is closest
            inSample[j_above] = True
            #sampleIndices = np.append(sampleIndices, j_above)
            massDiscrepancy = np.append(massDiscrepancy, np.log10(markArray[j_above,5])-merger_logMass)
        else:
            #mass below is closer
            inSample[j_below] = True
            #sampleIndices = np.append(sampleIndices, j_below)
            massDiscrepancy = np.append(massDiscrepancy, np.log10(markArray[j_below,5])-merger_logMass)
    elif found_above and (not found_below):
        inSample[j_above] = True
        #sampleIndices = np.append(sampleIndices, j_above)
        massDiscrepancy = np.append(massDiscrepancy, np.log10(markArray[j_above,5])-merger_logMass)
    elif (not found_above) and (found_below):
        inSample[j_below] = True
        #sampleIndices = np.append(sampleIndices, j_below)
        massDiscrepancy = np.append(massDiscrepancy, np.log10(markArray[j_below,5])-merger_logMass)
    else:
        print(f"no match found for {i}")
print("Matching done")
meanDiscrepancy = massDiscrepancy.mean()
stdDiscrepancy = np.sqrt(np.square(massDiscrepancy).mean())
print(f"mean = {meanDiscrepancy},\n std = {stdDiscrepancy}")


#%%plots1

cmap = mpl.cm.get_cmap("viridis")
norm = mpl.colors.Normalize()
fig, ax = plt.subplots()
hist, xedges, yedges, quadmesh = ax.hist2d(raArray*180/np.pi, decArray*180/np.pi, bins=50, norm = norm)
quadmesh.set_rasterized(True) # this rasterises plot so pdf effects don't make grid pattern, while keeping axes and labels as vectors
ax.set_title("RA-DEC distribution")
ax.set_xlabel("RA/degrees")
ax.set_ylabel("DEC/degrees")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cb.set_label("Number of galaxies per bin")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

fig, ax = plt.subplots()
ax.hist(distArray*h, bins = 50, edgecolor='black', linewidth=0.2)
ax.set_title("Distance Distribution")
ax.set_ylabel("Frequency")
ax.set_xlabel("distance / Mpc*h^-1")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

cmap = mpl.cm.get_cmap("viridis")
hist_bins = 50
hist_range = [[6,12],[-4,2]]
norm = mpl.colors.Normalize()#LogNorm()
fig, ax = plt.subplots()
all_hist, xedges, yedges, quadmesh = ax.hist2d(np.log10(markArray[:,5]), np.log10(markArray[:,4]), bins=hist_bins, range = hist_range, norm = norm)
ax.plot(np.linspace(hist_range[0][0],hist_range[0][1]), 0.76*np.linspace(hist_range[0][0],hist_range[0][1]) -7.64, ':', color='red')
quadmesh.set_rasterized(True)
ax.set_title("SFR,SM Distribution")
ax.set_xlabel("log10(Stellar mass / solar masses)")
ax.set_ylabel("log10(SFR / solar masses per year)")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cb.set_label("Number of galaxies per bin")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

cmap = mpl.cm.get_cmap("viridis")
hist_bins = 50
hist_range = [[6,12],[-4,2]]
norm = mpl.colors.Normalize()#LogNorm()
fig, ax = plt.subplots()
all_hist, xedges, yedges, quadmesh = ax.hist2d(np.log10(markArray[markArray[:,1]>0.5,5]), np.log10(markArray[markArray[:,1]>0.5,4]), bins=hist_bins, range = hist_range, norm = norm)
ax.plot(np.linspace(hist_range[0][0],hist_range[0][1]), 0.76*np.linspace(hist_range[0][0],hist_range[0][1]) -7.64, ':', color='red')
quadmesh.set_rasterized(True)
ax.set_title("Merger SFR,SM Distribution")
ax.set_xlabel("log10(Stellar mass / solar masses)")
ax.set_ylabel("log10(SFR / solar masses per year)")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cb.set_label("Number of galaxies per bin")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

cmap = mpl.cm.get_cmap("viridis")
hist_bins = 50
hist_range = [[6,12],[-4,2]]
norm = mpl.colors.Normalize()#LogNorm()
fig, ax = plt.subplots()
all_hist, xedges, yedges, quadmesh = ax.hist2d(np.log10(markArray[inSample,5]), np.log10(markArray[inSample,4]), bins=hist_bins, range = hist_range, norm = norm)
ax.plot(np.linspace(hist_range[0][0],hist_range[0][1]), 0.76*np.linspace(hist_range[0][0],hist_range[0][1]) -7.64, ':', color='red')
quadmesh.set_rasterized(True)
ax.set_title("Mass Matched Sample SFR,SM Distribution")
ax.set_xlabel("log10(Stellar mass / solar masses)")
ax.set_ylabel("log10(SFR / solar masses per year)")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cb.set_label("Number of galaxies per bin")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

#CDFS
fig,ax = plt.subplots(figsize=[10,5])
ax.plot(np.log10(markArray[:,5]), np.array(range(1,Ngalaxies+1))/Ngalaxies, label="all galaxies")
ax.plot(np.log10(markArray[inSample,5]), np.array(range(1,Nmergers+1))/Nmergers, label="mass-matched sample")
ax.plot(np.log10(markArray[isMerger,5]), np.array(range(1,Nmergers+1))/Nmergers, label="mergers")
ax.set_xlim([8,12])
ax.legend()
ax.set_title("CDFs in Stellar Mass")
ax.set_xlabel("log10(Stellar mass / solar masses)")
ax.set_ylabel("CDF")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")
print("mass KS p-values:")
print(f"all-merger: {kstest(np.log10(markArray[isMerger,5]), np.log10(markArray[:,5]))[1]}")
print(f"mms-merger: {kstest(np.log10(markArray[isMerger,5]), np.log10(markArray[inSample,5]))[1]}")
print(f"mms-all: {kstest(np.log10(markArray[:,5]), np.log10(markArray[inSample,5]))[1]}")

fig,ax = plt.subplots(figsize=[10,5])
ax.plot(np.sort(np.log10(markArray[:,4])), np.array(range(1,Ngalaxies+1))/Ngalaxies, label="all galaxies")
ax.plot(np.sort(np.log10(markArray[inSample,4])), np.array(range(1,Nmergers+1))/Nmergers, label="mass-matched sample")
ax.plot(np.sort(np.log10(markArray[isMerger,4])), np.array(range(1,Nmergers+1))/Nmergers, label="mergers")
ax.set_xlim([-3,1])
ax.legend()
ax.set_title("CDFs in SFR")
ax.set_xlabel("log10(SFR / solar masses per year)")
ax.set_ylabel("CDF")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")
print("SFR KS p-values:")
print(f"all-merger: {kstest(np.log10(markArray[isMerger,4]), np.log10(markArray[:,4]))[1]}")
print(f"mms-merger: {kstest(np.log10(markArray[isMerger,4]), np.log10(markArray[inSample,4]))[1]}")
print(f"mms-all: {kstest(np.log10(markArray[:,4]), np.log10(markArray[inSample,4]))[1]}")

#%%mark distribution
for l in range(1,Nmarks):
    fig, ax = plt.subplots()
    ax.hist(markArray[:,l], bins = 100, edgecolor='black', linewidth=0.2)
    ax.set_title(f"{markList[l]} Mark Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Mark Value")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")
    
    fig, ax = plt.subplots()
    ax.hist(np.log10(markArray[:,l])[np.isfinite(np.log10(markArray[:,l]))], bins = 100, edgecolor='black', linewidth=0.2)
    ax.set_title(f"log {markList[l]} Mark Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Mark Value")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

#%% Tree Search
print("Tree Algorithming:")
print(datetime.datetime.now())


Nscrambles = 10
scrambleIndices = (np.ones((Nmarks,Nscrambles))[np.newaxis,:,:]*np.arange(Ngalaxies)[:,np.newaxis,np.newaxis]).astype(int)
for i in range(Nscrambles):
    for j in range(Nmarks):
        rng.shuffle(scrambleIndices[:,j,i])

rootNode = Node(np.full(Ngalaxies,True),0)


pairBinEdges      = np.array(range(int(max_rp//delta_rp)+1))*delta_rp 
pairBinWidths     =  pairBinEdges[1:]  - pairBinEdges[:-1]
pairBinMidpoints  = (pairBinEdges[:-1] + pairBinEdges[1:] )/2

mean_markArray = markArray[rootNode.nodeIndices,:].mean(0)
mean_squareMarkArray = (markArray[rootNode.nodeIndices,:]**2).mean(0)

marks2cross = ((1,3),(1,4),(3,4),(1,5),(3,5))

pairBinTotals = DualTreeBinCount(rootNode,rootNode,pairBinEdges,marks2cross)

print("Finished finding pairs")
print(datetime.datetime.now())

#%%plots2

if savepath!="": np.save(os.path.join(savepath,"pairBinTotals"),pairBinTotals)

fig, ax = plt.subplots()
ax.bar(pairBinEdges[0:-1]*h, pairBinTotals[0,:,1], width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
ax.set_title("pairs in each annulus")
ax.set_ylabel("pairs")
ax.set_xlabel("pair separation / Mpc*h^-1")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

norm_pairBinCounts = pairBinTotals[0,:,0]/(np.pi*2*max_pi*h**3*(pairBinEdges[1:]**2 - pairBinEdges[:-1]**2))

fig, ax = plt.subplots()
ax.bar(pairBinEdges[0:-1]*h, norm_pairBinCounts, width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
ax.set_title("pairs per area of each annulus")
ax.set_ylabel("pairs per Mpc^3*h^-3")
ax.set_xlabel("pair separation / Mpc*h^-1")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

#error bars, estimating cf by last value alone should work well enough for error bar, could do with curve fit if bothered
errorBars = np.zeros((Nmarks,len(pairBinEdges)-1,2))
errorBars[:,:,0] = (pairBinTotals[:Nmarks,:,0]/((mean_markArray[:,np.newaxis]**2)*pairBinTotals[0,:,0]))*np.sqrt(
    (pairBinTotals[Nmarks:2*Nmarks,:,0]/pairBinTotals[:Nmarks,:,0]**2) - 1/pairBinTotals[0,:,0]
    + (mean_squareMarkArray[:,np.newaxis]*pairBinTotals[0,:,0]/pairBinTotals[:Nmarks,:,0] - 1)*2*(norm_pairBinCounts/norm_pairBinCounts[-1])/Ngalaxies)

errorBars[:,:,1] = np.sqrt(1/Nscrambles)*(pairBinTotals[:Nmarks,:,1]/((mean_markArray[:,np.newaxis]**2)*pairBinTotals[0,:,1]))*np.sqrt(
    (pairBinTotals[Nmarks:2*Nmarks,:,1]/pairBinTotals[:Nmarks,:,1]**2) - 1/pairBinTotals[0,:,1]
    + (mean_squareMarkArray[:,np.newaxis]*pairBinTotals[0,:,1]/pairBinTotals[:Nmarks,:,1] - 1)*2*(norm_pairBinCounts/norm_pairBinCounts[-1])/Ngalaxies)

for l in range(Nmarks):
    #MCFS
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[:-1]*h, pairBinTotals[l,:,0]/(pairBinTotals[0,:,0]*mean_markArray[l]**2), yerr=errorBars[l,:,0], width=pairBinWidths[:]*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title(markList[l] + " MCF")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")
    
    #scrambled MCFs
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[0:-1]*h, pairBinTotals[l,:,1]/(pairBinTotals[0,:,1]*mean_markArray[l]**2), yerr=errorBars[l,:,1], width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title("Scrambled " + markList[l] + " MCF")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

correlCoef = np.zeros((len(marks2cross),len(pairBinEdges)-1,2))
for l in range(len(marks2cross)):
    correlCoef[l,:,0] = ((pairBinTotals[0,:,0]*pairBinTotals[2*Nmarks+l,:,0] - (pairBinTotals[marks2cross[l][0],:,0])*(pairBinTotals[marks2cross[l][1],:,0]))
                         /np.sqrt((pairBinTotals[0,:,0]*pairBinTotals[Nmarks+marks2cross[l][0],:,0] - pairBinTotals[marks2cross[l][0],:,0]**2)
                                 *(pairBinTotals[0,:,0]*pairBinTotals[Nmarks+marks2cross[l][1],:,0] - pairBinTotals[marks2cross[l][1],:,0]**2)))
    
    correlCoef[l,:,1] = ((pairBinTotals[0,:,1]*pairBinTotals[2*Nmarks+l,:,1] - (pairBinTotals[marks2cross[l][0],:,1])*(pairBinTotals[marks2cross[l][1],:,1]))
                         /np.sqrt((pairBinTotals[0,:,1]*pairBinTotals[Nmarks+marks2cross[l][0],:,1] - pairBinTotals[marks2cross[l][0],:,1]**2)
                                 *(pairBinTotals[0,:,1]*pairBinTotals[Nmarks+marks2cross[l][1],:,1] - pairBinTotals[marks2cross[l][1],:,1]**2)))
    
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[0:-1]*h, correlCoef[l,:,0], yerr=(1-correlCoef[l,:,0]**2)/np.sqrt(pairBinTotals[0,:,0]), width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title(markList[marks2cross[l][0]] + " "+ markList[marks2cross[l][1]] +" Correlation")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")
    
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[0:-1]*h, correlCoef[l,:,1], yerr=(1-correlCoef[l,:,1]**2)/np.sqrt(Nscrambles*pairBinTotals[0,:,1]), width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title("Scrambled "+markList[marks2cross[l][0]] +" "+ markList[marks2cross[l][1]] +" Correlation")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")


fig, ax = plt.subplots()
ax.errorbar(pairBinMidpoints*h, pairBinTotals[9 ,:,0]/(pairBinTotals[0,:,0]*mean_markArray[9 ]**2), fmt='b', yerr=errorBars[9 ,:,0], label="Lu")
ax.errorbar(pairBinMidpoints*h, pairBinTotals[10,:,0]/(pairBinTotals[0,:,0]*mean_markArray[10]**2), fmt='g', yerr=errorBars[10,:,0], label="Lg")
ax.errorbar(pairBinMidpoints*h, pairBinTotals[11,:,0]/(pairBinTotals[0,:,0]*mean_markArray[11]**2), fmt='r', yerr=errorBars[11,:,0], label="Lr")
ax.set_title("Luminosity MCFs")
ax.set_xlabel("pair separation / Mpc*h^-1")
ax.legend()
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")

fig, ax = plt.subplots()
ax.errorbar(pairBinMidpoints*h, pairBinTotals[6,:,0]/(pairBinTotals[0,:,0]*mean_markArray[6]**2), fmt='b', yerr=errorBars[6,:,0], label="Lr/Lu")
ax.errorbar(pairBinMidpoints*h, pairBinTotals[7,:,0]/(pairBinTotals[0,:,0]*mean_markArray[7]**2), fmt='g', yerr=errorBars[7,:,0], label="Lr/Lg")
ax.set_title("Colour MCFs")
ax.set_xlabel("pair separation / Mpc*h^-1")
ax.legend()
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+".pdf")




