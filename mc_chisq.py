import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as sps
import twopiece.scale as tps
import statistics as stat
import matplotlib.pyplot as plt

def paramFit(array):
    
    #this should calculate mean and median
    #if mean-median difference is <1%, use normal pdf. if >1% use skewed.
    #if normal, present mean + std, if skewed present 4-value summary
    
    xx = np.linspace(min(array),max(array),1000)
    
    mean = stat.mean(array)
    median = stat.median(array)
    std = stat.stdev(array)
    skew = sps.skew(array)
    
    #find indices for ±1std
    flip = 0
    for i in range(len(xx)):
                
        if (flip == 0):
            if (xx[i] > mean - std):
                a = i
                flip = 1
        if (flip == 1):
            if (xx[i] > mean + std):
                b = i
                break
    
    if (abs(skew) < 0.1):
        fit_type = "normal"
        fit = sps.norm.fit(array)
        pdf = sps.norm.pdf(xx, *fit)
        
        return [fit_type, xx, pdf, array, mean, std, a, b, skew]
        
    else:
        fit_type = "skewed"
        q1 = np.quantile(array,0.25)
        q3 = np.quantile(array,0.75)
        fit = sps.skewnorm.fit(array)
        pdf = sps.skewnorm.pdf(xx, *fit)
        
        #find indices for q1 and q3
        flip = 0
        for i in range(len(xx)):
                    
            if (flip == 0):
                if (xx[i] > q1):
                    c = i
                    flip = 1
            if (flip == 1):
                if (xx[i] > q3):
                    d = i
                    break
        
        return [fit_type, xx, pdf, array, mean, std, a, b, median, q1, q3, c, d, skew]


def plotHist(data,design,info,min_chisq):
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in ([ax.title,ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() +ax.get_yticklabels()):
        item.set_fontsize(25)    
    ax.grid(False)
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])

    #histogram plot design
    if (data[0] == "normal"):
        ax.plot([data[4]-data[5],data[4]-data[5]],[0,data[2][data[6]]],'k--',linewidth=2) #-1std
        ax.plot([data[4]+data[5],data[4]+data[5]],[0,data[2][data[7]]],'k--',linewidth=2) #+1std
        ax.plot(data[1],data[2],design[4],linewidth=2) #fit line
        ax.fill_between(data[1][data[6]:data[7]],data[2][data[6]:data[7]],facecolor=design[7],alpha=0.5,
                        label=r'± 1$\sigma$ (± %.2f)' %(data[5]))
        n, bins, patches = ax.hist(data[3],bins='sqrt',density=True,facecolor=design[2],alpha=0.3,edgecolor=design[3]) #histogram
        ax.plot([data[4],data[4]],[0,max(n)+20],design[5],linewidth=2.5,
                label='mean = %.2f' %(data[4])) #mean
    
    else:
        
        ax.plot([data[4]-data[5],data[4]-data[5]],[0,data[2][data[6]]],'k--',linewidth=2) #-1std
        ax.plot([data[4]+data[5],data[4]+data[5]],[0,data[2][data[7]]],'k--',linewidth=2) #+1std
        ax.fill_between(data[1][data[6]:data[7]],data[2][data[6]:data[7]],facecolor=design[7],alpha=0.25,
                        label=r'± 1$\sigma$ (± %.2f)' %(data[5]))
        
        ax.plot([data[9],data[9]],[0,data[2][data[11]]],'k--',linewidth=2) #Q1
        ax.plot([data[10],data[10]],[0,data[2][data[12]]],'k--',linewidth=2) #Q3
        ax.plot(data[1],data[2],design[4],linewidth=2) #fit line
        ax.fill_between(data[1][data[11]:data[12]],data[2][data[11]:data[12]],facecolor=design[8],alpha=0.5,
                        label='Q1 = %.2f \nQ2 = %.2f \nIQR = %.2f' %(data[9],data[10],data[10]-data[9])) #inter quartile range
        n, bins, patches = ax.hist(data[3],bins='sqrt',density=True,facecolor=design[2],alpha=0.3,edgecolor=design[3])
        
        ax.plot([data[4],data[4]],[0,max(n)*1.05],design[5],linewidth=2.5,
                label='Mean = %.2f' %(data[4])) #mean
        ax.plot([data[8],data[8]],[0,max(n)*1.05],design[6],linewidth=2.5,
                label='Median = %.2f' %(data[8])) #median
    
    if (min_chisq != 0):
        ax.plot([min_chisq,min_chisq],[0,max(n)*1.05], 'k-', linewidth=2.5,
                label=r'$\chi^2$ min = %.2f' %(min_chisq))
        

    #ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.9, 1),
              fancybox=True, shadow=True, fontsize=20)
     
    
    if (design[0] != [0,0]):      
        #dynamic plot limits
        ax.set_xlim(design[0][0],design[0][1])
    else:
        #dynamic plot limits
        ax.set_xlim(data[4]-3*data[5],data[4]+3*data[5])
    
    if (design[1] != [0,0]):
        ax.set_ylim(design[1][0],design[1][1])
    else:
        #dynamic plot limits
        ax.set_ylim(0,max(n)*1.1)

    plt.savefig(info['savefig'], bbox_inches='tight')
    plt.show()
    
    return

def plotData(data,shape,design,info):
    
    #design = hist_face_color, fit_line
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for item in ([ax.title,ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() +ax.get_yticklabels()):
        item.set_fontsize(25)
    ax.grid(False)
    
    #change axes here
    ax.set_xlabel(info['xlabel'])
    ax.set_ylabel(info['ylabel'])
    
    x = data[0]
    y = data[1] 
    z = data[2]
    
    #builds histogram
    for i in  range(len(y)):
        if (i == 0):
            histData = int(y[0]) * [x[0]]
        else:
            histData += int(y[i]) * [x[i]]
    
    bins = x.copy()
    bins.append(z)
    for i in range(len(bins)):
        bins[i] -= 0.5 * (bins[1] - bins[0])
    
    n, b, patches =  ax.hist(histData,bins=x,density=False,facecolor=design[2],alpha=0.5,edgecolor=design[3])
    
    xx = np.linspace(x[0],x[len(x)-1],1000)
    if (shape[0] == "double gaussian"):
        A_alpha = data[3]
        mu_alpha = data[4]
        A_beta = data[5]
        mu_beta = data[6]
        sig1 = shape[1]
        sig2 = shape[2]
        X = np.empty([3,len(xx)])
        X[1][0] = sig1
        X[2][0] = sig2
        for i in range(len(xx)):
            X[0][i] = xx[i]
        
        yy_sum = f_double_gaussian(X,A_alpha,mu_alpha,A_beta,mu_beta)
        yy_alpha = f_gaussian(xx,A_alpha,mu_alpha,sig1)
        yy_beta = f_gaussian(xx,A_beta,mu_beta,sig2)
        
        #plot curves
        ax.plot(xx,yy_sum,design[4])
        ax.plot(xx,yy_alpha,design[5])
        ax.plot(xx,yy_beta,design[6])
            
    elif (shape[0] == "gaussian"):
        A = data[3]
        mu = data[4] 
        sig = data[5]
    
        yy = f_gaussian(xx,A,mu,sig)
        
        #plot curves
        ax.plot(xx,yy,design[4])
    
    
    if (design[0] != [0,0]):      
        #dynamic plot limits
        ax.set_xlim(design[0][0],design[0][1])
   
    
    if (design[1] != [0,0]):
        ax.set_ylim(design[1][0],design[1][1])
    

    #change savefile here
    plt.savefig(info['savefig'], bbox_inches='tight')
    plt.show()
    
    return



def fit(x, y, yerr, trials, shape):
    
    """
    if (shape[0] == "linear"):
        linear(x, y, yerr, trials)
    elif (shape[0] == "exponential"):
        exponential(x, y, yerr, trials)
    """

    if (shape[0] == "double gaussian"):
        data = double_gaussian(x, y, yerr, trials, shape)
    elif (shape[0] == "gaussian"):
        data = gaussian(x, y, yerr, trials, shape)
    
    return data


#Gaussian curve function
def f_gaussian(x, A, mu, sig):
    return A * np.exp(-(x - mu)**2/(2*sig**2)) 

#Double Gaussian
def f_double_gaussian(X, A_alpha, mu_alpha, A_beta, mu_beta):
    x = X[0]
    sig1 = X[1][0]
    sig2 = X[2][0]
    return A_alpha * np.exp(-(x - mu_alpha)**2/(2*sig1**2)) + A_beta * np.exp(-(x - mu_beta)**2/(2*sig2**2)) 


#chisq calculation
def calc_chisq(shape, x, y, params):
    
    if (shape[0] == "double gaussian"):
        A_alpha, mu_alpha, A_beta, mu_beta = params
        X = np.empty([3,len(x)])
        X[1][0] = shape[1]
        X[2][0] = shape[2]
        for i in range(len(x)):
            X[0][i] = x[i]
        r = f_double_gaussian(X,A_alpha,mu_alpha,A_beta,mu_beta)
        
    elif (shape[0] == "gaussian"):
        A, mu, sig = params
        r = f_gaussian(x,A,mu,sig)
    
    chisq = 0
    for i in range(len(r)):
        chisq += (y[i]-r[i])**2
    
    chisq_red = chisq/(len(x)-len(params))
        
    return chisq,chisq_red


def gaussian(x, y, yerr, trials, shape):
    
    p0 = shape[3]
    
    y_values = []

    # generate distribution of points
    for i in range(len(y)):
        
        if (len(yerr) == 2): #if len = 2, [0] = sigma1, [1] = sigma2
            dist = tps.tpnorm(loc=y[i], sigma1=yerr[0][i], sigma2=yerr[1][i])
        else:
            dist = tps.tpnorm(loc=y[i], sigma1=yerr[i], sigma2=yerr[i])
        sample = dist.random_sample(size = trials)
        y_values.append(sample)
    
    
    yi = np.empty(len(y))
    A = []
    mu = []
    sig = []
    
    #fit distributions of points
    for i in range(trials):
        
        for j in range(len(y)):
            yi[j] = y_values[j][i]
                
        popt,pcov = curve_fit(f_gaussian,x,yi,p0=p0)
        A.append(popt[0])
        mu.append(popt[1])
        sig.append(popt[2])
             
    A = paramFit(A)
    mu = paramFit(mu)
    sig = paramFit(sig)
    
    return A, mu, sig


def double_gaussian(x, y, yerr, trials, shape):
    
    sig1 = shape[1]
    sig2 = shape[2]
    p0 = shape[3]
    
    #must be able to pass the variables sig1 and sig2 to the function that needs fitting
    
    X = np.empty([3,len(x)])
    X[1][0] = sig1
    X[2][0] = sig2
    for i in range(len(x)):
        X[0][i] = x[i]
    
    y_values = []

    # generate distribution of points
    for i in range(len(y)):
        
        if (len(yerr) == 2): #if len = 2, [0] = sigma1, [1] = sigma2
            dist = tps.tpnorm(loc=y[i], sigma1=yerr[0][i], sigma2=yerr[1][i])
        else:
            dist = tps.tpnorm(loc=y[i], sigma1=yerr[i], sigma2=yerr[i])
        sample = dist.random_sample(size = trials)
        y_values.append(sample)
    
    
    yi = np.empty(len(y))
    A_alpha = []
    mu_alpha = []
    A_beta = []
    mu_beta = []
      
    #fit distributions of points
    for i in range(trials):
        
        for j in range(len(y)):
            yi[j] = y_values[j][i]
                
        popt,pcov = curve_fit(f_double_gaussian,X,yi,p0=p0)
        A_alpha.append(popt[0])
        mu_alpha.append(popt[1])
        A_beta.append(popt[2])
        mu_beta.append(popt[3])
    
    mu_diff = []
    pr_alpha = []
    pr_beta = []
    for i in  range(len(mu_alpha)):
        mu_diff.append(mu_beta[i]-mu_alpha[i])
        area_alpha = A_alpha[i]*np.sqrt(2*np.pi*sig1**2)
        area_beta = A_beta[i]*np.sqrt(2*np.pi*sig2**2)
        
        pr_alpha.append((area_alpha/(area_alpha+area_beta)))
        pr_beta.append((area_beta/(area_alpha+area_beta)))
    
    A_alpha = paramFit(A_alpha)
    mu_alpha = paramFit(mu_alpha)
    A_beta = paramFit(A_beta)
    mu_beta = paramFit(mu_beta)
    mu_diff = paramFit(mu_diff)
    pr_alpha = paramFit(pr_alpha)
    pr_beta = paramFit(pr_beta)
    
    return A_alpha, mu_alpha, A_beta, mu_beta, mu_diff, pr_alpha, pr_beta
   
 
"""
def linear(x, y, yerr, trials):
    return

def exponential(x, y, yerr, trials):
    return
"""