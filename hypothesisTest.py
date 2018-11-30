from scipy.stats import t
import math
def hypothesisTesting(mean1,std1,mean2,std2,n,alpha=0.95):
   X = ((mean1-mean2)*math.sqrt(n))/math.sqrt(std1**2+std2**2)
   v = (std1**2 + std2**2)**2*(n-1)/(std1**4+std2**4)
   xAlpha = t.ppf(1-alpha,v)
   
   betterAlg = 0; 

   if(X > xAlpha):
     betterAlg = 1

   return betterAlg
   
