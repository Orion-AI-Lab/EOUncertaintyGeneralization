import numpy as np
from scipy.stats import rankdata

def calc_cpa(response, predictor):
    """
    Calculate CPA coefficient.

    CPA attains values between zero and one. Weighted probability of concordance. 

    Parameters
    ----------
    response : 1D array_like, 1-D array containing observation (response). Need to have the same length in the ``axis`` dimension as predictor.
    predictor : 1D array_like, 1-D array containing predictions for observation.
       
    Returns
    -------
    correlation : float
        	  CPA coefficient 
    """    
    response = np.asarray(response)
    if response.ndim > 1:
        raise ValueError("CPA only handles 1-D arrays of responses")

    predictor = np.asarray(predictor)
	
    if predictor.ndim > 1:
        ValueError("CPA only handles 1-D arrays of forecasts")   
  
    	# check for nans
    if np.isnan(np.sum(response)) == True:
        ValueError("response contains nan values")
		
    if np.isnan(np.sum(predictor)) == True:
        ValueError("forecast contains nan values")
	
    #responseOrder = np.argsort(response)
    #responseSort = response[responseOrder] 
    #forecastSort = predictor[responseOrder]                
    forecastRank = rankdata(predictor, method='average')
    responseRank = rankdata(response, method='average')
    responseClass = rankdata(response, method='dense')
    
    return((np.cov(responseClass,forecastRank)[0][1]/np.cov(responseClass,responseRank)[0][1]+1)/2) 
	
 