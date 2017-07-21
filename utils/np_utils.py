import numpy as np

def pad_rgb_2d_lobsided(arrToPad, padVal, position=None): #position need be one of top bottom left right
    r, c, rgb = arrToPad.shape

    if ((position=='T') or (position=='B')):
        r+=padVal
    if ((position=='L') or (position=='R')):
        c+=padVal

    outArr = np.full(shape=(r,c,rgb), fill_value=np.nan, dtype=arrToPad.dtype)

    if position=='T':
        outArr[padVal : r, 0 : c, :] = arrToPad

    if position=='B':
        outArr[0 : (r-padVal), 0 : c, :] = arrToPad

    if position=='L':
        outArr[0 : r, padVal : c, :] = arrToPad

    if position=='R':
        outArr[0 : r, 0 : (c-padVal), :] = arrToPad

    return outArr



def pad_2d_edges(arrToPad, padVal):

    m,n = arrToPad.shape
    paddedArr = np.full(shape=(m+padVal*2, n+padVal*2), fill_value=np.nan, dtype=arrToPad.dtype)
    paddedArr[padVal : (m+padVal), padVal : (n+padVal)] = arrToPad

    return paddedArr



def unpad_2d_edges(arrToUnPad, padVal):
    m,n = arrToUnPad.shape

    unPaddedArr = arrToUnPad[padVal : (m+padVal), padVal : (n+padVal)]
                       
    return unPaddedArr



def unpad_2d_edges_after_resX(arrToUnPad, addToEndPadL=0, overWriteResX=0):
    global padL
    global resX

    if overWriteResX == 0:
        overWriteResX = resX-3
    else:
        overWriteResX = resX-3

    # begZ = (padL+1+addToEndPadL) * ((2)**(overWriteResX-2))  #Left and top is where nans begin. should this be padL * 2?
    # endZ = (padL+addToEndPadL) * ((2)**(overWriteResX-2))+1#  The bigger EndZ, the smaller the array 

    begZ = ((padL+1)*2) * ((2)**(overWriteResX))#Left and top is where nans begin. should this be padL * 2?
    endZ = (padL*2) * ((2)**(overWriteResX))#  The bigger EndZ, the smaller the array



    unPaddedArr = np.copy(arrToUnPad[(begZ):(arrToUnPad.shape[0] - endZ), (begZ) : (arrToUnPad.shape[1] - endZ)])
                       
    return unPaddedArr



def pad_3d_edges(anInArr, padding=None):
    global animationPaddingMultiplier


    if padding==None:
        pv = (int)(math.ceil(anInArr.shape[1] * animationPaddingMultiplier))
    else:
        pv = padding

    h = anInArr.shape[1]
    w = anInArr.shape[2]

    newH = h + 2 * pv
    newW =  w + 2 * pv

    paddedWaveArr = np.full(shape=(anInArr.shape[0], newH, newW, 4), fill_value=1.0, dtype="float32")

    paddedWaveArr[:, pv : (pv + h), pv : (pv + w), :] = anInArr

    return paddedWaveArr, pv



def pad_rgb_2d_edges(arrToPad, padVal):
    m, n, rgb = arrToPad.shape
    paddedArr = np.full(shape=(m+padVal*2, n+padVal*2, rgb), fill_value=np.nan, dtype=arrToPad.dtype)
    paddedArr[padVal : (m+padVal), padVal : (n+padVal), :] = arrToPad

    return paddedArr    



def insert_alt_rows(arrToAddRows, val):
    newNrows = arrToAddRows.shape[0]*2
    arrWithNewRows = np.full(shape=(newNrows, arrToAddRows.shape[1]), fill_value=val, dtype=arrToAddRows.dtype)
    arrWithNewRows[1::2,] = arrToAddRows
    # print(str(arrToAddRows.shape) + " _ new rows shape _ " + str(arrWithNewRows.shape)
    return arrWithNewRows



def insert_alt_cols(arrToAddCols, val):
    newNcols = arrToAddCols.shape[1]*2
    arrWithNewCols = np.full(shape=(arrToAddCols.shape[0], newNcols), fill_value=val, dtype=arrToAddCols.dtype)
    arrWithNewCols[:,1::2] = arrToAddCols
    # print(str(arrToAddCols.shape) + " _ new cols shape _ " + str(arrWithNewCols.shape)
    return arrWithNewCols



def insert_alt_rows_and_cols(arrToAddRowsAndCols, val=np.nan): #Handles 2d and 3d
    if (len(arrToAddRowsAndCols.shape)==2): #if 2D

        altRowsArr = insert_alt_rows(np.copy(arrToAddRowsAndCols), val)
        altColsAndRowsArr = insert_alt_cols(altRowsArr, val)

    else: #assume 3D
        newNrows = arrToAddRowsAndCols.shape[0]*2
        newNcols = arrToAddRowsAndCols.shape[1]*2

        altColsAndRowsArr3D = np.full(shape=(newNrows,newNcols,arrToAddRowsAndCols.shape[2]), fill_value=np.nan, dtype=arrToAddRowsAndCols.dtype)
        for x in range(arrToAddRowsAndCols.shape[2]):
            altRowsArr = insert_alt_rows(arrToAddRowsAndCols[:,:,x], val)
            altColsAndRowsArr3D[:,:,x] = insert_alt_cols(altRowsArr, val)  

        altColsAndRowsArr = altColsAndRowsArr3D

    return altColsAndRowsArr



## ======== Normalisation functions 

def normalise_element_wise(elementToNorm_, scalingArrMin_, toNormArrMin_, scalingArrRange_, toNormArrRange_):

    scalingX = np.divide((elementToNorm_ - toNormArrMin_), toNormArrRange_)
    normedElement = scalingX * scalingArrRange_ + scalingArrMin_

    return normedElement



def normalise_between(arrToNormalise, arrToScaleTo, onlyNonZeros=False):
    if onlyNonZeros:
        onlyZeroLocs = np.where(arrToNormalise==0)
        print("shape(nonZeroLocs) ", shape(onlyZeroLocs))
    scalingArrMin = np.nanmin(arrToScaleTo)
    scalingArrMax = np.nanmax(arrToScaleTo)
    scalingArrRange = scalingArrMax - scalingArrMin

    toNormArrMin = np.nanmin(arrToNormalise)
    toNormArrMax = np.nanmax(arrToNormalise)
    toNormArrRange = toNormArrMax - toNormArrMin

    func = np.vectorize(normalise_element_wise, otypes=[np.float])

    normedArr = func(arrToNormalise, scalingArrMin, toNormArrMin, scalingArrRange, toNormArrRange)   

    if onlyNonZeros:
        normedArr[onlyZeroLocs] = 0    
    return normedArr



# ========== Conversion between nans, zeros, ones, for 2d array
# Decided masks add a layer of unnecessary complexity
def make_nans_zero_ATs_one(inArr):
    outArr = np.copy(inArr)
    outArr[~(np.isnan(inArr))] = 1
    outArr[np.isnan(inArr)] = 0
    return outArr    

def make_nans_zero(inArr):
    outArr = np.copy(inArr)
    outArr[np.isnan(inArr)] = 0
    return outArr

def make_nans_ones(inArr):
    outArr = np.copy(inArr)    
    outArr[np.isnan(inArr)] = 1
    return outArr    

def make_zeros_nans(inArr):
    outArr = np.copy(inArr)        
    outArr[np.where(inArr==0)] = np.nan
    return outArr

def make_ones_nans(inArr):
    outArr = np.copy(inArr)        
    outArr[np.where(inArr==1)] = np.nan
    return outArr   

def make_non_nans_one(inArr):
    outArr = np.copy(inArr)        
    outArr[get_non_nan_coords(inArr)] = 1
    return outArr   

def invert_ones_zeros(arrToInvert):
    outArr = np.copy(arrToInvert)
    outArr[np.where(arrToInvert==1)] = 0
    outArr[np.where(arrToInvert==0)] = 1
    return outArr

 

def get_non_nans(inArr):
    outArr = np.copy(inArr)
    outArr = outArr[get_non_nan_coords(inArr)]
    return outArr

def get_non_nan_coords(inArr):
    outArr = np.copy(inArr)
    nonNanCoords = np.where((make_nans_zero_ATs_one(inArr))==1)
    return nonNanCoords   

def get_nan_coords(inArr):
    outArr = np.copy(inArr)
    nanCoords = np.where((make_nans_zero_ATs_one(inArr))==0)
    return nanCoords



## =========== creation of nan sandwiches
def create_nan_sandwiches(nanSandwichSize):
    sandwichArr = np.full(shape=(1, nanSandwichSize), fill_value=np.nan)
    invSandwichArr = np.copy(sandwichArr)
    #Make bread (outer layers) = 1
    sandwichArr[0,0] = 1
    sandwichArr[0,nanSandwichSize-1] = 1

    horzArr = sandwichArr

    vertArr = horzArr.reshape((-1,1))

    lrDiagArr = np.diagflat(horzArr)

    lrDiagArr[np.where(lrDiagArr==0)] = np.NAN
    lrDiagArr=np.array(lrDiagArr)

    rlDiagArr = np.rot90(lrDiagArr)

    #invert sandwiches
    invSandwichArr[0,1:(nanSandwichSize-1)] = 1

    invHorzArr = invSandwichArr

    invVertArr = invHorzArr.reshape((-1,1))
    
    invLrDiagArr = np.diagflat(invHorzArr)

    invLrDiagArr[np.where(invLrDiagArr==0)] = np.NAN
    invLrDiagArr = np.array(invLrDiagArr)  

    invRlDiagArr = np.rot90(invLrDiagArr)

    return [lrDiagArr, rlDiagArr,horzArr, vertArr], [invLrDiagArr, invRlDiagArr, invHorzArr, invVertArr]  



## =========== numpy union intersection ========
def replace_2d_arr_with_union_of(intersection12, union1, union2, replaceArr):
    tmpArr = np.copy(union1)
    union1[get_non_nan_coords(intersection12)] = np.nan

    tmpArr[:,:] = union1+union2
    # tmpArr[:,:] = make_zeros_nans(make_nans_zero(union2) + (make_nans_zero(union1) - make_nans_zero(intersection12)))
    return tmpArr



def replace_2d_arr_with_intersection_of(intersect1, intersection2, replaceArr):
    tmpArr = np.copy(replaceArr)
    tmpArr[:,:] = np.nan
    tmpArr[np.where(intersect1==intersection2)] = replaceArr[np.where(intersect1==intersection2)]
    
    return tmpArr



def replace_2d_arr_with_difference_of(this1, minusThis1):
    tmpArr = np.copy(this1)
    tmpArr[:,:] = np.nan
    tmpArr[np.where(this1!=minusThis1)] = this1[np.where(this1!=minusThis1)]

    return np.copy(tmpArr)



def union_of(intersection12, union1, union2):

    tmpArr = np.copy(union1)
    union1[get_non_nan_coords(intersection12)] = np.nan
    tmpArr[:,:] = union1 + union2

    # tmpArr[:,:] = make_zeros_nans(make_nans_zero(union2) + (make_nans_zero(union1) - make_nans_zero(intersection12)))

    return tmpArr



def intersection_of(intersect1, intersect2):

    tmpArr = np.copy(intersect1)
    tmpArr[:,:] = np.nan
    tmpArr[np.where(intersect1==intersect2)] = intersect1[np.where(intersect1==intersect2)]

    return tmpArr    
