
"""
Author: Rebecca Thomson
IIT #A20548618

Oct. 10, 2024

Project Description:  To preform a Regularized Linear Discriminant Analysis on classification (as described in 
section 4.3.1 of Elements of Statistical Learning, 2nd Edition) from first principals.

This program takes in the clean array of attributes and a dependent classifier variable. 
    -Data with NA or other missing values will not be accepted (an error check exists)
The dependent classifier variable:
    -Can be 2 or more classes
    -Do not need to be ballanced
    -Each class should be identified by a unique number
    -Class identifying numbers need not be sequential.
The attributes:
    -All attributes should be numerical and sequential or one-hot encoded.
    -The weighted covariance matrix (sigma-hat) of the attributes must be invertable.

The program will automatically seperate the classes for all calculations.
The program will then calculate a weighted sigma hat covarience matrix.
The program will then used this sigma hat to find a regularized LDA with Sigma(alpha, gamma)
Unfortunately, the program must be given data that produces an invertable sigma-hat.



"""
'''libraries'''
#numpy
import numpy as np
#pandas
import pandas as pd
#system, for exit if certain errors.
import sys
from sklearn.model_selection import train_test_split
#for test

class regLDA():
    def __init__(self):
        pass
        #self.xs=xs
        #self.ys=ys   

        
    def update_sigma_step_a(self, sigma_given, cov_given, alpha):            
        update_sigma=[]
        inv_alpha=1-alpha       
        #must have iteration to add array 
        for j in range(len(self.sigma_hat)):
                
            s_temp_row=[]
            for k in range(len(self.sigma_hat[j])):
                s_temp=(cov_given[j][k]*alpha)+(sigma_given[j][k]*inv_alpha)
                s_temp_row.append(s_temp)
                #Build new sigma hat
            update_sigma.append(s_temp_row)
            
            #this will result in a sigma_hat(alpha)
        return update_sigma
    def update_sigma_step_g(self, sigma_given, gamma): 
        update_sigma=[]
        inv_gamma=1-gamma
        #sigma_squared=cov_given[K][K]
        for j in range(0,len(self.sigma_hat)):    
            s_temp_row=[]
            for k in range(0, len(self.sigma_hat)):
                if(k==j):
                    s_temp=sigma_given[j][k]*gamma+inv_gamma*sigma_given[j][k]
                    s_temp_row.append(s_temp)
                else:
                    s_temp_row.append(sigma_given[j][k]*gamma)
                #Build new sigma hat
            update_sigma.append(s_temp_row)
        return update_sigma

    def log_loss(self,sigma_used,K_class,x_list):
      #This produces a proportional probability function for a given class with a given sigma and using the data's mu and pi.
        mu=pandas.DataFrame(self.mu_list[K_class]).iloc[:, 0]
        x=pandas.DataFrame(x_list).iloc[:, 0]
        
        pi=self.pi_list[K_class]
        try:
            sigma_inv=numpy.linalg.inv(numpy.asarray(sigma_used))
            
        except:
            print("An error occured inverting the sigma-hat matrix.")
        #Original formula is Eq. 4.10 of TEXT.
        term_three= numpy.log(pi)
        #print(term_three)
        term_two = -1/2*(numpy.matmul(mu.transpose(),numpy.matmul(sigma_inv,mu)))
        #print(term_two)
        term_one= (numpy.matmul(x.transpose(),numpy.matmul(sigma_inv,mu)))
        #print(term_one)
        term_for_this_class=term_one+term_two+term_three                  
        return term_for_this_class
    def predict_single(self,temp_s,x):
        #Because we don't care the exact probability of the winning catagory, only which catagory won, we return the index of 
        #with the highest proportional probability.
        temp_guess=[]
        for i in range(0,len(self.classes_given)):
            value=self.log_loss(temp_s,i,x)
            temp_guess.append(value)
        #print(temp_guess)
        winner=self.classes_given[temp_guess.index(max(temp_guess))]
        return winner    
    def predict(self,X_list):
        
        #checking that input is list.
        if isinstance(X_list,pandas.core.frame.DataFrame):
            X_list=X_list.iloc[:,:].values.tolist()
        if (isinstance(X_list,list)!=True):
            print("Wrong input format.  Not List or DataFrame.  Try again.")
            print(type(X_list))
            tempstatus=1
            sys.exit(tempstatus)
        #creates predicted  list for 'outside' use
        predict_list=[]
        #print(type(self.sigma_hat))
        for i in range(0,len(X_list)):
            each_pred=self.predict_single(self.sigma_hat,X_list[i])
            predict_list.append(each_pred)
        return predict_list
    def tunning_predict(self,testxs,testys):
        #To determine the best gamma, we must tune the parameter.
        #make test, train sets    
        best_gamma_list=[]
        temp_gamma=1.0
        modified_sigma=self.sigma_hat
        temp_gamma=self.gamma
        Error_rate=0 
        for i in range(0,100):
            Error_rate=0   
            #train with given gamma
            temp_sigma=self.update_sigma_step_g(modified_sigma,temp_gamma)
            #test with given gamma
            predict_list=[]
            for i in range(0,len(testxs)):
                each_pred=self.predict_single(temp_sigma,testxs[i])
                predict_list.append(each_pred)
            #compare test predicted to given. +1 to error rate for each error
            for k in range(0, len(testxs)):
                if (testys[k]!=predict_list[k]):
                    Error_rate+=1
                
            best_gamma_list.append([temp_gamma,Error_rate]) #add gamma and error rate to list
            temp_gamma+=-0.01
        #now, to find best gamma
        gamma_df = pandas.DataFrame(best_gamma_list, columns = ["temp_gamma", "Errors"])  #turn into dataframe      
        winner_gamma=gamma_df.loc[gamma_df["Errors"].min(), 'temp_gamma']#.iloc[0]   #extract min error's gamma, this will pick the first instance of this rate.   
        #set the gamma,sigma for the whole model
        self.gamma=winner_gamma
        #print(winner_gamma)
        temp_sigma=self.sigma_hat
        self.sigma_hat=self.update_sigma_step_g(temp_sigma,winner_gamma)      
    def fit(self,fullys,fullxs):
        #checking that input is list.
        if isinstance(fullxs,pandas.core.frame.DataFrame):
            fullxs=fullxs.iloc[:,:].values.tolist()
        if isinstance(fullys,pandas.core.frame.DataFrame):
            fullys=fullys.iloc[:,:].values.tolist()
        if (isinstance(fullxs,list)!=True):
            print("Wrong input format on X.  Not List or DataFrame.  Try again.")
            print(type(fullxs))
            tempstatus=1
            sys.exit(tempstatus)
        if (isinstance(fullys,list)!=True):
            print("Wrong input format on Y.  Not List or DataFrame.  Try again.")
            print(type(fullys))
            tempstatus=1
            sys.exit(tempstatus)
            #fullys=[int(item[0]) for item in fullys]
        #setting given variables into test and training set for future tuning.
        
        # split the dataset
        xs, testXS, ys, testYS = train_test_split(fullxs, fullys, test_size=0.20, random_state=0)
        #print(xs)
        #print(len(xs))
        
        self.totalN=len(ys) #total no. of training data points (#rows)
        self.total_i_per_class=[]
        self.classes_given=list(set(ys)) #list of possible classes
        self.classes_given_no=len(self.classes_given) #no. of classes
        self.attributes_no=len(xs[0]) # no attributes (#columns in xs)
        self.X_by_class=[] #Empty list that will have each class' xs values listed seperately
        self.mu_list=[]
        self.cov_list=[] 
        self.pi_list=[]
        self.gamma = 1.0
        row_template = [0] * self.attributes_no
        self.sigma_hat= [row_template[:] for _ in range(self.attributes_no)]
        self.sigma_hat_unmodified= [row_template[:] for _ in range(self.attributes_no)]
        #Check input data for N/A errors
        tempErrorFound=False
        if (pandas.isna(ys).any() or pandas.isnull(ys).any()):
            print("NA found in response variable.  check data.  Analysis stopped")
            tempstatus=1
            sys.exit(tempstatus)
        for i in range (0, self.totalN):
            if (pandas.isna(xs[i]).any()or pandas.isnull(xs[i]).any()):
                tempErrorFound=True
                print("NA found in dependant variables.  check data.  Analysis stopped")
                tempstatus=1
                sys.exit(tempstatus)
        if (self.totalN != len(xs)):
            print("Dependant and response variables not egual length.  check data.  Analysis stopped")
            tempstatus=1
            sys.exit(tempstatus)
        #Error check, there must be more points than diffent classes.  # classes < N
        if (self.classes_given_no==self.totalN):
            print("Data Points are not separatable into classes.  check data.  Stop analysis.")
            tempstatus=1
            sys.exit(tempstatus)
        
        #Caluculate initial variable values 
        for i in range(0,self.classes_given_no): #Itterate through classes to divide by class.
            temp_list=[]
            #for j in range(0,self.total_i):
            #this creates a filtered list of [row][col]
            for j in range(0,len(ys)):
                if (ys[j]==self.classes_given[i]):
                    temp_list.append(xs[j])
                
            #temp_list=xs[numpy.where(ys==self.classes_given[i])[0],:].tolist() #Itterate through xs to divide by class.
            #this creates the filtered master list of [class][row][col] , and determines pi for each class
            self.X_by_class.append(temp_list) #stick list into master list.
           
            self.total_i_per_class.append(len(temp_list)) # Count Nk per class for weighted sigma
            temp_pi=len(temp_list)/self.totalN
            self.pi_list.append(temp_pi) #Pi for each class
            
            #Check that each class individually has enough data points
            if (self.attributes_no>len(temp_list)):
                print("Not enough data points for this class.  Stop analysis.  Class at stop:")
                print(self.classes_given[i])
                tempstatus=1
                sys.exit(tempstatus)
            #this appends the mu_list for each mean, creates the cov. for each class.
            
            #for j in range(0,self.classes_given_no): #Itterate through classes    
            temp_mu=[]
            #for j in range(0, self.attributes_no):#TODO 
            temp_mu= numpy.mean(temp_list, axis=0)#TODO Check that axis does not shift
            #print(temp_mu)
            self.mu_list.append(temp_mu)
            #now that info is seperated by class, calculate other values.  
        
            #this creates each cov to the cov_list for each class:
            df=np.array(temp_list)
            cov_temp= numpy.cov(df.transpose())
            #(1.0/(len(temp_list)-self.classes_given_no))* numpy.matmul((temp_list-self.mu_list[i]).T,(temp_list-self.mu_list[i])) #TODO check formula
            #Keeping cov for each class
            self.cov_list.append(cov_temp)
            temp_sigma=[]
            for j in range(len(cov_temp)):
                #take each cov value *pi to make this classes' portion of sigma hat
                s_temp_row=[]
                for k in range(len(cov_temp[j])):
                    temp=cov_temp[j][k]*temp_pi
                    s_temp=temp+self.sigma_hat[j][k]
                    
                    s_temp_row.append(s_temp)
                #Build new sigma hat
                temp_sigma.append(s_temp_row)
            
             #this will result in a weighted sigma hat 
            self.sigma_hat=temp_sigma
            self.sigma_hat_unmodified=temp_sigma
        # tune the gamma
        self.tunning_predict(testXS,testYS)
    def Single_confusion_matrix(self,T_pos,T_neg,F_pos,F_neg,class_name):
        text1=class_name+"vs. All Others.                Actual".rjust(40)
        text2="Pos".rjust(30)
        text3="Neg".rjust(20)
        text_title=text2+text3
        text_line="- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - "
        text4=format("Pred. Pos",'<10')
        text5=format("Pred. Neg",'<10')
        textTP=format(T_pos,'>20')
        textFP=format(F_pos,'>20')
        textFN=format(F_neg,'>20')
        textTN=format(T_neg,'>20')
        p_text=text4 + textTP + textFP 
        n_text=text5 + textFN + textTN
        results=text_line +'\n' +text1 +'\n' +text_line +'\n' + text_title +'\n' +text_line +'\n' + p_text +'\n' + text_line +'\n' +n_text +'\n' +text_line +'\n' 

        print(results) 
        
    def confusion_matrix(self,Y_predict,Y_actual):
        for i in range(0,self.classes_given_no): #Itterate through classes to create indvidual confusion matrixes per class)
            T_pos=0
            T_neg=0
            F_pos=0
            F_neg=0
            for j in range(0,len(Y_predict)):
                if (Y_predict[j]==Y_actual[j]):
                    if (Y_predict[j]==self.classes_given[i]):
                        T_pos+=1
                    else:
                        T_neg+=1
                else:
                    if (Y_predicted[j]==self.classes_given[i]):
                        F_pos+=1
                    else:
                        F_neg+=1
            
            self.Single_confusion_matrix(T_pos,T_neg,F_pos,F_neg,str(self.classes_given[i]))
                