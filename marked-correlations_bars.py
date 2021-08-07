#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:29:28 2021

@author: hopkins9942

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import datetime
import os
#from scipy.stats import kstest
#import time

#import warnings # used for catch_warnings context manager which can turn warnings into exceptions for debugging

#%% defs1

h = 0.7
delta_rp = 0.2/h # Mpc
max_rp = 10/h
max_pi = 40/h # Mpc # skibba was 40/h, should possibly do same for final run. DAis and peebles 83 used 2500kms^s/H0 = 36Mpc, similar
rng = np.random.default_rng() # random number generator for scrambling marks, keeping constant seed while developing
datapath = r"D:\Data\marked-correlations"
savepath = "D:\\Data\\marked-correlations\\results\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(savepath)
markList = ["count", "Strong bar", "Weak bar", "Total bar", "Quenching", "SFR", "Stellar mass", "r spectral luminosity"]
Nmarks = len(markList)
imageExtension = ".png"

#%% defs2
# split so savepath can remain unchanged

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
    #global globalcount
    
    count = np.zeros((2*Nmarks+len(marks2cross), len(binEdges)-1, 2))
    if node1.idnum!=node2.idnum:
        for i in np.arange(Ngalaxies)[node1.nodeIndices]:
            for j in np.arange(Ngalaxies)[node2.nodeIndices]:
                if (abs(distArray[i]-distArray[j])<max_pi):
                    temp_rp = rp(distArray[i]+distArray[j], raArray[i]-raArray[j], decArray[i], decArray[j])
                    for k in range(len(binEdges)-1):
                        if (temp_rp<binEdges[k+1]):
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

with fits.open(os.path.join(datapath,"MatchedTable.fits"),
                memmap=True) as hdul:
    matchedTable = hdul[1].data
    hdr = hdul[1].header

Lnu_ref = 4.34538e20#ergs^-1Hz^-1
unfiltered_raArray   = np.zeros(hdr["NAXIS2"])
unfiltered_decArray  = np.zeros(hdr["NAXIS2"])
unfiltered_distArray = np.zeros(hdr["NAXIS2"])
unfiltered_markArray = np.zeros((hdr["NAXIS2"], Nmarks))
for lineNum in range(hdr["NAXIS2"]):
    if lineNum%10000==0:
        print(lineNum)
    if ((not np.isfinite(matchedTable[lineNum]["bar_strong_fraction"]))
        or (not np.isfinite(matchedTable[lineNum]["ELPETRO_ABSMAG"][0]))
        or (matchedTable[lineNum]["disk-edge-on_no"] < 0.5*matchedTable[lineNum]["smooth-or-featured_total-votes"])):
        # checks entry has defined bar vote fractions (ie has at least 1 vote for spiral)
        # and checks entry has measured magnitudes (removes around 30 objects)
        # and checks at least half of volunteers classified object as non-edge-on spiral
        continue
        # leaves entry as all zeros, which is then filtered out
    else:
        temp_SFR = (1.4e-28)*Lnu_ref*10**((matchedTable[lineNum]['ELPETRO_ABSMAG'][1]+5*np.log10(h))/(-2.5))
        temp_SM = matchedTable[lineNum]['ELPETRO_MASS']/(h**2)
        
        unfiltered_raArray[lineNum]   = matchedTable[lineNum]['RA_1']*np.pi/180
        unfiltered_decArray[lineNum]  = matchedTable[lineNum]['DEC_1']*np.pi/180
        unfiltered_distArray[lineNum]  = (3e3 * matchedTable[lineNum]["ZDIST"])/h
        unfiltered_markArray[lineNum,:] = np.array([1,
                                matchedTable[lineNum]["bar_strong"]/matchedTable[lineNum]["bar_total-votes"],
                                matchedTable[lineNum]["bar_weak"]/matchedTable[lineNum]["bar_total-votes"],
                                (matchedTable[lineNum]["bar_strong"]
                                 + matchedTable[lineNum]["bar_weak"])/matchedTable[lineNum]["bar_total-votes"],
                                # matchedTable[lineNum]["bar_strong"]/matchedTable[lineNum]["smooth-or-featured_total-votes"],
                                # matchedTable[lineNum]["bar_weak"]/matchedTable[lineNum]["smooth-or-featured_total-votes"],
                                # (matchedTable[lineNum]["bar_strong"]
                                #  + matchedTable[lineNum]["bar_weak"])/matchedTable[lineNum]["smooth-or-featured_total-votes"],
                                (temp_SM**0.76)/(temp_SFR*10**(7.64)),
                                temp_SFR,
                                temp_SM,
                                Lnu_ref*10**((matchedTable[lineNum]["ELPETRO_ABSMAG"][4]+5*np.log10(h))/(-2.5)),])


#%%filtering

filterIndices = matchedTable[:]["ELPETRO_NMGY"][:,4]>=10**(9-17.77/2.5) # flux limit r<17.77
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:, markList.index("count")]!=0)
# explicitly removes invalid entries (objects without measured magnitudes and definite ellipticals)

filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,markList.index("Quenching")]<1e3)#q
filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,markList.index("Quenching")]>1e-2)#q
# filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,4]<1e1)#SFR
# filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,4]>1e-2)#SFR
# filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,5]<1e11)#SM 
# filterIndices = np.logical_and(filterIndices, unfiltered_markArray[:,5]>1e9)#SM #TODO this needs adjusting for vol lim changes

#Volume limit
volLimMr = -18 #actual magnitude assuming value of h
d_VL = 10**((-7.23-volLimMr)/5) #actual dist assuming value of h
Lr_VL = Lnu_ref*10**(-0.4*volLimMr)#actual lum assuming value of h
filterIndices = np.logical_and(filterIndices, np.logical_and(unfiltered_distArray<d_VL, unfiltered_markArray[:,markList.index("r spectral luminosity")]>Lr_VL))
if savepath!="": np.save(os.path.join(savepath,"filterIndices"),filterIndices)

raArray   = unfiltered_raArray[filterIndices]
decArray  = unfiltered_decArray[filterIndices]
distArray = unfiltered_distArray[filterIndices]
markArray = unfiltered_markArray[filterIndices,:]
Ngalaxies = len(distArray)

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
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

fig, ax = plt.subplots()
ax.hist(distArray*h, bins = 50, edgecolor='black', linewidth=0.2)
ax.set_title("Distance Distribution")
ax.set_ylabel("Frequency")
ax.set_xlabel("distance / Mpc*h^-1")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

cmap = mpl.cm.get_cmap("viridis")
hist_bins = 50
hist_range = [[6,12],[-4,2]]
norm = mpl.colors.Normalize()#LogNorm()
fig, ax = plt.subplots()
all_hist, xedges, yedges, quadmesh = ax.hist2d(np.log10(markArray[:,markList.index("Stellar mass")]), np.log10(markArray[:,markList.index("SFR")]), bins=hist_bins, range = hist_range, norm = norm)
ax.plot(np.linspace(hist_range[0][0],hist_range[0][1]), 0.76*np.linspace(hist_range[0][0],hist_range[0][1]) -7.64, ':', color='red')
quadmesh.set_rasterized(True)
ax.set_title("SFR,SM Distribution")
ax.set_xlabel("log10(Stellar mass / solar masses)")
ax.set_ylabel("log10(SFR / solar masses per year)")
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
cb.set_label("Number of galaxies per bin")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

for l in range(1,Nmarks):
    fig, ax = plt.subplots()
    ax.hist(markArray[:,l], bins = 100, edgecolor='black', linewidth=0.2)
    ax.set_title(f"{markList[l]} Mark Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Mark Value")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)
    
    fig, ax = plt.subplots()
    ax.hist(np.log10(markArray[:,l])[np.isfinite(np.log10(markArray[:,l]))], bins = 100, edgecolor='black', linewidth=0.2)
    ax.set_title(f"log {markList[l]} Mark Distribution")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Mark Value")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

#%% Tree Search
print("Tree Algorithming:")
print(datetime.datetime.now())


Nscrambles = 10
scrambleIndices = (np.ones((Nmarks,Nscrambles))[np.newaxis,:,:]*np.arange(Ngalaxies)[:,np.newaxis,np.newaxis]).astype(int)
for i in range(Nscrambles):
    for j in range(Nmarks):
        rng.shuffle(scrambleIndices[:,j,i])
# scrambleIndices[k,j,i] is the index at which the value of the jth mark
# of the kth galaxy in the ith scrambling is found in markArray.
# Alternatively, markArray[scrambleIndices[k,j,i], j] is value of jth mark 
# of kth galaxy in ith scramble.
#Note: since marks are scrambled independantly, scrambled functions should show NO CORRELATIONS

rootNode = Node(np.full(Ngalaxies,True),0)


pairBinEdges      = np.array(range(int(max_rp//delta_rp)+1))*delta_rp 
pairBinWidths     =  pairBinEdges[1:]  - pairBinEdges[:-1]
pairBinMidpoints  = (pairBinEdges[:-1] + pairBinEdges[1:] )/2

mean_markArray = markArray[rootNode.nodeIndices,:].mean(0)
mean_squareMarkArray = (markArray[rootNode.nodeIndices,:]**2).mean(0)

marks2cross = ((markList.index("Strong bar"),markList.index("Weak bar")),
               (markList.index("Total bar"),markList.index("Strong bar")),
               (markList.index("Total bar"),markList.index("Weak bar")),
               (markList.index("Total bar"),markList.index("Quenching")),
               (markList.index("Total bar"),markList.index("Stellar mass")),
               (markList.index("Weak bar"),markList.index("Quenching")),
               (markList.index("Weak bar"),markList.index("Stellar mass")),
               (markList.index("Strong bar"),markList.index("Quenching")),
               (markList.index("Strong bar"),markList.index("Stellar mass")),)
               # (markList.index("Strong bar (weighted)"),markList.index("Weak bar (weighted)")),
               # (markList.index("Total bar (weighted)"),markList.index("Strong bar (weighted)")),
               # (markList.index("Total bar (weighted)"),markList.index("Weak bar")),
               # (markList.index("Total bar (weighted)"),markList.index("Quenching")),
               # (markList.index("Total bar (weighted)"),markList.index("Stellar mass")),
               # (markList.index("Weak bar (weighted)"),markList.index("Quenching")),
               # (markList.index("Weak bar (weighted)"),markList.index("Stellar mass")),
               # (markList.index("Strong bar (weighted)"),markList.index("Quenching")),
               # (markList.index("Strong bar (weighted)"),markList.index("Stellar mass")))

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
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

norm_pairBinCounts = pairBinTotals[0,:,0]/(np.pi*2*max_pi*h**3*(pairBinEdges[1:]**2 - pairBinEdges[:-1]**2))

fig, ax = plt.subplots()
ax.bar(pairBinEdges[0:-1]*h, norm_pairBinCounts, width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
ax.set_title("pairs per area of each annulus")
ax.set_ylabel("pairs per Mpc^3*h^-3")
ax.set_xlabel("pair separation / Mpc*h^-1")
if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

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
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)
    
    #scrambled MCFs
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[0:-1]*h, pairBinTotals[l,:,1]/(pairBinTotals[0,:,1]*mean_markArray[l]**2), yerr=errorBars[l,:,1], width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title("Scrambled " + markList[l] + " MCF")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)

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
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)
    
    fig, ax = plt.subplots()
    ax.bar(pairBinEdges[0:-1]*h, correlCoef[l,:,1], yerr=(1-correlCoef[l,:,1]**2)/np.sqrt(Nscrambles*pairBinTotals[0,:,1]), width=pairBinWidths*h, align="edge", edgecolor='black', linewidth=0.2)
    ax.set_title("Scrambled "+markList[marks2cross[l][0]] +" "+ markList[marks2cross[l][1]] +" Correlation")
    ax.set_xlabel("pair separation / Mpc*h^-1")
    if savepath!="": fig.savefig(os.path.join(savepath,ax.get_title())+imageExtension)




