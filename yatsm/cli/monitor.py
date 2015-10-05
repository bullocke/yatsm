def check_mask():
  #Check to see if pixels overlap with mask of monitoring location
  
def update_status():
  #Save new entry on status - is it in the middle of a possible change? 
  
def update_probability():
  #Save the current probability that the pixel is undergoing a chance - this can be sensor specific
  
def check_refit():
  #Check if enough data exists to refit the model - may or may not be necessary
  
def run_pixel():
  #Run YATSM on pixel if is within the mask (if there is a mask). This may reference yatsm.py, or may need unique functions
  
def determine_prob():
  '''Use various weighted inputs to determine the probability of change in near real time. This inputs include:
     -Magnitude of change
     -Type of change (as decided in inputs)
     -View angle for MODIS
     -Consecutive days
     -Cloud and shadow masks (similar to type of change)'''
