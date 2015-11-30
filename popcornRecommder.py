
# coding: utf-8

# In[201]:

import numpy as np
from scipy.optimize import minimize
np.set_printoptions(threshold='nan')


# ## prepare data set
#     - this bloc is for preparing the data matrix has_rating[i,j] 1 if a paticular user has rated a particular movie, 0 otherwise. Rating[i,j] real is rating that given by particular users **value exist if and only if coresponding has_rating[i,j]==1
#     - number of users can be optain by len(names) and number of movies can be optain by len(movies)

# In[202]:

import StringIO
import csv
with open('resource/resource_sheet.csv') as f:
    s = StringIO.StringIO(f.read()) 
    reader = csv.reader(s, delimiter=',')
    raw = [row for row in reader]

def get_names(raw):
    return raw[0][1:]

def get_movies(raw):
    movies = []
    for i in range(1, len(raw)):
        movies.append(raw[i][0])
    return movies

def get_rating(raw):
    
    names = get_names(raw)
    movies = get_movies(raw)
    name_offset = 1
    movies_offset = 1
    
    n_names = len(names)
    n_movies = len(movies)

    has_rating = np.zeros((n_names, n_movies))
    rating = np.zeros((n_names, n_movies))
    
    for ir in range(n_movies):
        
        this_row = raw[movies_offset + ir]

        for ip in range(n_names):
            
            this_cell =  this_row[name_offset + ip]
            hr = this_cell.isdigit()
            has_rating[ip, ir] = 1 if hr else 0
            if hr:
                rating[ip, ir] = int(this_cell)
                
    return names, movies, has_rating, rating

names, movies, has_rating, rating = get_rating(raw)

#print "name", names
#print "movies", movies
#print "has_rating", has_rating.T
#print "rating", rating.T


# ## Implementation of Detail Mean Normalization
#     - the seection below here is implemented to handle the case of newly added user that has never rated any movie before 
#     - notice that in our data set we have Eve who never rate any movie so we can think of her as newly added user

# In[203]:

# init the average matrix
average = np.zeros((len(movies),1))

# find average value for each row 
for i in range (0,len(movies)):
    sum_each_row = 0
    count_num = 0
    
    for j in range (0,len(names)):
        
        if has_rating.T[i][j] == 1:
            sum_each_row = sum_each_row + rating.T[i][j]
            count_num = count_num+1
            
    average[i][0] = sum_each_row/count_num
    
#print average


# # running data after Implementation of Detail Mean Normalization
#     - on the section below here we re-run our main again with the new rating matrix  

# In[204]:

#print "original rating",rating.T

# re-init value for rating matrix
for i in range (0,len(movies)):
    for j in range (0,len(names)):
        if has_rating.T[i][j] == 1:
            rating.T[i][j] = rating.T[i][j] - average[i]
#print "rating after DMN",rating.T 


# ## Getting the best weight
#     - start Linear Regression class below here

# In[205]:

class LinearRegression: 
    
     # init all the data so it can easily accessed by all methods
    def __init__(self,has_rating,real_rating,fea,lamda,names,movies,theta,x):
        
        #has_rating and real_rating here are already transposed
        self.has_rating = has_rating
        self.real_rating = real_rating
        self.fea=fea
        self.lamda=lamda
        self.names = names
        self.movies = movies
        self.num_names = len(names)
        self.num_movies = len(movies) 
        self.theta = theta
        self.x = x
    
    #this function is use to find the gradient value for x 
    def sum_x(self,i,k,lamda):
        
        sum_x_value = 0 
        
        for j in range (0,self.num_names):
            if self.has_rating[i][j] == 1:
                sum_x_value = sum_x_value+(np.dot(self.theta[j].T,self.x[i]) - self.real_rating[i][j])*self.theta[j][k]         
        
        return sum_x_value
    
    #this function is use to find the gradient value for theta
    def sum_theta(self,j,k,lamda):
        
        sum_theta_value = 0
        
        for i in range (0,self.num_movies):
            if self.has_rating[i][j] == 1:
                sum_theta_value = sum_theta_value+(np.dot(self.theta[j].T,self.x[i]) - self.real_rating[i][j])*self.x[i][k]
        
        return sum_theta_value
                
                
                
                
    #this function will seek for the x and thata that can meet the local minimum of the function
    def gradient_descent(self,alpha,lamda):
        
        # init keep_array for both x and theta
        keep_x = np.zeros((self.num_movies,self.fea))
        keep_theta = np.zeros((self.num_names,self.fea))
        keep_x = self.x
        keep_theta = self.theta
        
        #for-loop here is used for allowing the convergent of gradient value
        for times in range(0,10000):
           
            #re-assign value to self.x and self.theta
            self.x = keep_x
            self.theta = keep_theta
            
            #set the value of x
            for i in range (0,self.num_movies):
                for k in range (0,self.fea):
                    
                    keep_x[i][k] = self.x[i][k] - alpha*(self.sum_x(i,k,lamda)+lamda*self.x[i][k])
                    
            #set the value of theta
            for j in range (0,self.num_names):
                for k in range (0,self.fea):
                    
                    keep_theta[j][k] = self.theta[j][k] - alpha*(self.sum_theta(j,k,lamda)+lamda*self.theta[j][k])
            
            #every 10000 time we reduce the step size
            if times == 9000:
                alpha = alpha/10
            
        return self.x,self.theta


# ## Calling linear regression class

# In[206]:

#set the number of feature 
number_feature = 5

#init small guessing value for both theta and x
guess_theta = np.random.uniform(0.01,1,[len(names),number_feature])
guess_x = np.random.uniform(0.01,1,[len(movies),number_feature])

#get object from linear regression class
linear_obj = LinearRegression(has_rating.T,rating.T,number_feature,0.01,names,movies,guess_theta,guess_x)

#get return from gradient descent function (minimize x and minimize theta)
result_x, result_theta = linear_obj.gradient_descent(0.05,0.01)

#show them
#print "x", result_x
#print "theta", result_theta


# ## Output array

# In[207]:

#out array is the final guessing value from minimize x and theta
out = np.zeros((len(movies),len(names)))

for i in range(0,len(movies)):
            for j in range(0,len(names)):
                    out[i][j] = float("{0:.2f}".format(float((np.dot(result_theta[j].T,result_x[i])))+float((average[i]))))
#print out


# ## Display as Table
#     -Code below here is use for display the output as table so it is easier to compare the outcome to the real rating

# In[208]:

from IPython.display import HTML

class TableCell:
    
    def __init__(self, text, tc=None, color=None):
        self.text = text
        self.tc = tc
        self.color = color
    
    def to_html(self):
        return '<td>%s</td>'%self.text
    
def maketable(rating, has_rating, guess, restaurants, names,average):
    n_rests = len(restaurants)
    n_names = len(names)
    tab = np.empty((n_rests+1, n_names+1),dtype='object')
    
    for irest in range(n_rests):
        tab[irest+1,0] = restaurants[irest]

    for iname in range(n_names):
        tab[0,iname+1] = names[iname]

    for irest in range(n_rests):
        
        for iname in range(n_names):
            
            if not has_rating[irest, iname]:
                tab[irest+1, iname+1] = TableCell('<span style="color:red">%3.2f</span>'%(guess[irest, iname]))
            else:
                tab[irest+1, iname+1] = TableCell('<span style="color:blue">%3.2f</span><span style="color:red">(%3.2f)</span>'%(rating[irest, iname]+average[irest], guess[irest, iname]))
    
    #now convert tab array to nice html table
    nrow, ncol = tab.shape
    t = []
    t.append('<table>')
    for irow in range(nrow):
        t.append('<tr>')
        for icol in range(ncol):
            cell = tab[irow,icol]
            if cell is not None:
                if isinstance(cell,TableCell):
                    t.append(tab[irow, icol].to_html())
                else:
                    t.append('<td>')
                    t.append(tab[irow, icol])
                    t.append('</td>')
            else:
                t.append('<td></td>')
        t.append('</tr>')  
    t.append('</table>')
    return '\n'.join(t)


# In[209]:

#HTML(maketable(rating.T, has_rating.T, out, movies, names,average))


# ## Getting recommen list 
#     -recommend movie for particular user
# 

# In[210]:

import operator
import sys
def recommend(user):
    if user in names:
        recommend_list = {}
        sorted_list = {}
        for i in range (0,len(movies)):
            if has_rating.T[i][names.index(user)]==0:
                recommend_list[str(movies[i])] = out[i][names.index(user)]
                #recommend_list.append(out[i][names.index(user)])
        sorted_list = sorted(recommend_list.items(), key=operator.itemgetter(1),reverse=True)
        print sorted_list
        #wright output to file
        out_file = open('output/output.txt','w')
        for output_name,output_rating in sorted_list:
            out_file.write("%s,%s\n" % (str(output_name),str(int(output_rating))))  
    else: 
        print "database does not contain this user"
        #wright error to file
        out_file = open('output/output.txt','w')
        out_file.write("database does not contain this user")
        
#user_input = "Tae"
user_input=str(sys.argv[1])
print user_input
recommend(user_input)


# In[ ]:



