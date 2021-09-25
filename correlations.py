# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:37:15 2021

@author: matth

numpy arrays are passed by reference

functions defined in a module can be redefined by simple redefinition:
    
    def newFunct():
        ...
        
    modulename.funct = newFunct

other functions in module which use module.funct will use updated function
new functions can also be added to module

class methods can be added/changed after class definition:
    
    class A:
        a=3
        
    A.newMethod = newFunct
    
    newMethod can then be used in all instances of A, even ones defined before newFunct
        
Users could modify functions defined in this file to suit their geometry

USage: 
    user imports file, sets up data and position arrays
    modifies geometry functions and parameters
    runs correlationCalculator(data, position, ...)
    
module parameters should be quantifies with default values: #TODO
cC arguments should be very changeable: #TODO

data is (Npoints,Nmarks) array,
positions is (Npoints,Ndimentions) array    

schematic of module:
    Node class defines how to split node into children, and leaf definition
    
    (min/max)(pi/rp)diff functions give distances between nodes, #TODO needs to be generalised
    
    slowBinCount calculates contributions from cells not entirely within or without
    
    some convenience functions
    
    DualBinTreeCount calculates sums, using (min/max)(pi/rp)diff functions
    
#TODO
replace (min/max)(pi/rp)diff functions with more general function which tells if two nodes
are within, without or neither, which can be redefined by user
preferable: make user just need to customise one point-to-point distance function

How it should all work:
    User supplies coordinates of points, and array of properties, and separation function.
    
    
    Need way of splitting nodes into children.
    could just split one axis in half - which axis? how to take geometry into account?
    eg in my code splitting in r direction probably did little good
    splitting should take user separation function into account only!
    
    when two nodes considered, the min and max seperations are measured and compared 
    to maxsep. if max<maxsep, splitting does no good and slowbincount begins
    if min>maxsep, there can be no pairs so 0 is returned
    if neither of above, node containing largest number of points is split.
    How Should It Be Split? Should be by disecting one coordinate axis
    Bear in mind that often the two nodes will be identical
    Should be by disecting one coordinate axis - anything fancier will take considerable time
    which coordinate? Goal is to create one child with a much larger min and a much smaller max
    Could measure "width" of node in each coordinate by user-separation of midpoints of each pair of faces
    this would produce splits in r if extent of cell exceeded rmax
    
    Plan for now: put in global maximiser as default,
    check timing vs local, if slow just try local with sufficiently square cells
    later put in geometry options - if global minimiser not too slow then not needed
    sort splitting as described above
    consider how to implement diffpimax and others like it
    
    Plan as of 19/09, after break:
        local minimiser with roughly square cells will probably be fine
        as long as it doesn't get stuck on perpendicular walls (use relevant method) DONE
        Measure width of cells with user separation between midpoints of faces, DONE
        then split along largest width direction by half by coordinate DONE
        I think this implemented will be end!
    
    
    With regard to lists and tuples, following convention of 
    https://stackoverflow.com/questions/626759/whats-the-difference-between-lists-and-tuples
    noting especially that although they CAN be used similarly (python is for
    consenting adults) there is a difference in how they SHOULD be used to
    match the conventions of the community and keep understandable code.
    
    Thoughts 25/09/21 while considering random scrambling.
    Most conventional way of doing this would be to define a function to be
    used, rather than changing functions and variables inside module from outside.
    this partially solves seperation issue - have cartesian, polar options and
    option to define own function. Passing big arrays is fine, it occurs by reference.
    DOING THIS NOW
    Plan:
        Big function called either "DualTree<something>"
        - not generic name in case I add other algorithms
        Takes input of marks, positions, seperation function option/def, marks2cross
        Additional options (possibly as arguments with default values or editable
                            with crrl.minLeafPoints=10 etc):
            leaf parameters
        
    Should function just output sums, or output exactly what can be plotted?
    Means are easily calculable, so user could change easily
    Output MCF as more useable?
    Output sums because crosscorrelations may be less simple?
    
    Solution: have DualTreeBinnedCount to perform algorithm basically,
    Then have a calling function which sets up nodes, calls DTBC on double base node,
    then divides by means to produce graphables
    
    PROBLEM: WHEN TESTING MIN AND MAX SEPERATION,
    FORGOT TO TEST WHEN CALLED ON SAME NODE
    
    To do - test maxsep on same node
    - write from top down calling function, then DualTreeBinnedCount, then slow bin count, etc
"""

import numpy as np
import datetime
from scipy import optimize

def DualTreeBinnedCount():
    """
    Calculates ...
    Inputs ...
    
    Uses modified version of DualTreeCount algorithm in Moore et al. 2001.
    Reason for modification is in motivating case typical size of leaf cells 
    was larger than bin sizes, meaning negligibly few cells would lie entirely
    within s_lo and s_hi of each other. Solution was to use dual tree approach
    to efficiently exclude groups of points with minimum seperation greater
    than maximum seperation investigated, then perform slow per-pair counts 
    on what remained.
    Resulting algorithm is thus:
        Measure maximum and minumum distances between nodes.
        If min exceeds maximum seperation investigated, no pairs therfore return zero
        ElseIf max is less than maximum seperation investigated, every point in n1 pairs with every point in n2
        so perform slow bin count
        ElseIf both nodes are leafs then slow bin count
        Elseif nodes are distict, split largest aliong longest axis and return 
        sum of DTBC on (child1, other node) and (child2, other node) 
        else (if same node, non-leaf) split along longest axis and return
        DTBC(child1,child1) + DTBC(child1,child2) + DTBC(child2,child2)
    """

def separation(coords1, coords2):
    """
    Customisable, currently set to Peebles rp for testing
    coords is (dist,dec,ra)
    #TODO change to euclidian default #TODO add in inf when out of pi range
    #TODO can't use inf as for minimiser function must be increasing, maybe use step and increase
    """
    # distSum = coords1[0]+coords2[0]
    # dec1 = coords1[1]
    # dec2 = coords2[1]
    # raDiff = coords1[2]-coords2[2]#order irrelevant as only in sin**2
    
    # arg = np.sin((dec1-dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(raDiff/2)**2
    # sep = (distSum)*np.sqrt(arg/(1-arg))
    
    temp=0
    for i in range(coords1.size):
        temp+=(coords1[i]-coords2[i])**2
    sep = np.sqrt(temp)
    
    return sep


class Node:
    
    leafMinSize = 0 #TODO tune these values - especially min points (minsize unknowable)
    leafMinPoints = 10
    
    def __init__(self, nodeIndices, idnum):
        self.nodeIndices = nodeIndices
        self.idnum = idnum
        
        self.bounds = tuple((min(coordinateArray[nodeIndices, i]), max(coordinateArray[nodeIndices, i])) for i in range(coordinateArray.shape[1]))
        
        self.midpoints = tuple((self.bounds[i][0]+self.bounds[i][1])/2 for i in range(coordinateArray.shape[1]))
        
        self.widths = tuple(separation(
                                     tuple(self.bounds[j][0] if j==i else self.midpoints[j]
                                      for j in range(coordinateArray.shape[1])),
                                     tuple(self.bounds[j][1] if j==i else self.midpoints[j]
                                      for j in range(coordinateArray.shape[1]))
                                     ) for i in range(coordinateArray.shape[1]))
        #self.widths[i] is separation between points on midpoints of faces corresponding to min and max in coordinate number i

        self.numPoints = np.count_nonzero(self.nodeIndices)
        
    @property
    def isLeaf(self): # done as property in case leafMinPoints etc is changed
        if (np.max(self.widths)<=self.leafMinSize)or(self.numPoints<=self.leafMinPoints):
            return True
        else:
            return False
            
    def leftChild(self):
        if self.isLeaf:
            return None
        else:                
            maxSizeIndex = np.argmax(self.widths)
            leftIndices = np.logical_and(coordinateArray[:, maxSizeIndex] < self.midpoints[maxSizeIndex], self.nodeIndices)
            return Node(leftIndices, 2*self.idnum+1)
        
    def rightChild(self):
        if self.isLeaf:
            return None
        else:
            maxSizeIndex = np.argmax(self.widths)
            rightIndices = np.logical_and(coordinateArray[:, maxSizeIndex] >= self.midpoints[maxSizeIndex], self.nodeIndices)
            return Node(rightIndices, 2*self.idnum+2)
        

class _testnode:
    """used to test min and max functions"""
    def __init__(self,bounds):
        self.bounds = bounds
        dist_split = ((self.bounds[0][1]**3 + self.bounds[0][0]**3)/2)**(1/3)
        dec_split = np.arcsin((np.sin(self.bounds[1][1]) + np.sin(self.bounds[1][0]))/2)
        ra_split = (self.bounds[2][1] + self.bounds[2][0])/2
        self.midpoints = (dist_split, dec_split, ra_split)


def _minmaxsep_local(node1, node2, mode):
    """Returns coordinates of min/max (mode dependant) separations between nodes
    as len=2n np array, with first n entries being coordinates in node1 and
    second n entries being coordinates in node2
    
    #TODO : cases occur where global minimum not found eg cartesian sep with n1 = (0,1)(0,1)(0,1), n2 = (2,3)(0,1)(0,1)
    default minimize just goes to local minimum, which results in wrong minimum or "perpendicular wall" error
    perpendicular wall error could be fixed by using random starting point, but this increases chance of wrong minimum found
    cannot assume minimum will be on corner, imagine euclidean distance with cells in polar coordinates
    """
    bounds = node1.bounds + node2.bounds # concatenates bounds into single tuple
    if mode==0: # finding minimum separation
        func = lambda coords:    separation(coords[:coords.size//2], coords[coords.size//2:])
    elif mode==1: # finding maximum separation
        func = lambda coords: -1*separation(coords[:coords.size//2], coords[coords.size//2:])
    res = optimize.minimize(func, np.asarray(node1.midpoints+node2.midpoints),
                            method = 'Nelder-Mead',
                            bounds=bounds)
    return res#.x #TODO decide what to output
#Methods that work: NM,P
#that dont: L-BFGS-B, TNC, SLSQP, trust-

def _minmaxsep_global(node1, node2, mode):
    """uses global minimiser
    .
    Testing with shgo
    
    unused
    """
    bounds = node1.bounds + node2.bounds # concatenates bounds into single tuple
    if mode==0: # finding minimum separation
        func = lambda coords:    separation(coords[:coords.size//2], coords[coords.size//2:])
    elif mode==1: # finding maximum separation
        func = lambda coords: -1*separation(coords[:coords.size//2], coords[coords.size//2:])
    res = optimize.shgo(func, bounds, sampling_method='halton')
    return res#.x #TODO decide what to output
#Methods that work: NM,P
#that dont: L-BFGS-B, TNC, SLSQP, trust-constr
    
def minSep(node1, node2):
    """Finds maximum separation between nodes"""
    coords = _minmaxsep_local(node1, node2, 0).x
    return separation(coords[:coords.size//2], coords[coords.size//2:])
    
def maxSep(node1, node2):
    """Finds minimum separation between nodes"""
    coords = _minmaxsep_local(node1, node2, 1).x
    return separation(coords[:coords.size//2], coords[coords.size//2:])
    
                
def SlowBinCount(node1,node2, binEdges, marks2cross):
    """
    Calculates weighted pair sums (not normalised) in bins between two nodes. Works for nodes equal or different, at any separation.
    Used after DualTreeBinCount has trimmed nodes outside of range for max speed.
    V7 update now sums directly over node indices, calculating pairs sums, squared sums, and given crosses. Plus scrambled version:
        i and j index galaxies, k indexes bin, m1 and m2 index marks with cross product index m = Tm1+m2
    """
    #global globalcount
    
    count = np.zeros((2*markArray.shape[1]+len(marks2cross), len(binEdges)-1, 2))
    if node1.idnum!=node2.idnum:
        for i in np.arange(coordinateArray.shape[0])[node1.nodeIndices]:
            for j in np.arange(coordinateArray.shape[0])[node2.nodeIndices]:
                
                
                temp_sep = separation(coordinateArray[i,:],coordinateArray[j,:])
                for k in range(len(binEdges)-1):
                    if (temp_sep<binEdges[k+1]):
                        for l in range(0,markArray.shape[1]):
                            count[l,k,0] += markArray[i,l]*markArray[j,l]
                            #TODO got to here, need tot work out hw to scramble. Think I need scrambleindices as global
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
                
def DualTreeBinCount(node1,node2, binEdges, marks2cross):
    """
    Similar to Moore 2000, with weighted pair counts for pairs with rp in each bin and separation in pi less than pimax
    Returns array of sums of weighted pair counts with entry for each mark plus two, like markArray plus cross correlations
    Splits and prunes nodes, then feeds nearby nodes into SlowBinCount to calculate sums
    """
    
    # PUT RNG HERE
    
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
    
def correlationCalculator(data, binEdges, marks2cross, max_pi, max_rp):
    rootNode = Node(np.full(data.shape[0],True),0)
    return DualTreeBinCount(rootNode, rootNode, binEdges, marks2cross)
    