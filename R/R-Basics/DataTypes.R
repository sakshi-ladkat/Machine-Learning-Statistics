#Basic Data Structures
my_num=3.4                                                            #numeric
my_int=300                                                            #integer
my_logic = FALSE                                                      #Bbolean
REGIONS =factor(c("North America","South America","Europe","Asia"))   #Factor


#vectors
?vector                                                     #helps to know more about vectors
my_vector=c(24,15,16,17)                                    #initializing vector
my_vector                                                   #Printing vector
str(my_vector)                                              #checking data type of a vector
?summary                                                    #helps to know more about vectors        
my_vector[2]                                                #Access 2nd element in the vector


#Matrices
?matrix                                                   #helps to know more about vectors        
my_matrix =matrix(c(1,2,3,4,5,6,7,8,9),
                  nrow=3,ncol=3,byrow=TRUE)               #creates matrix with  data points row-wise 
my_matrix
summary(my_matrix)                                        #Prints matrix
my_matrix =matrix(c(1,2,3,4,5,6,7,8,9),nrow=3,ncol=3)    #creates matrix with  data points column-wise 
my_matrix[7]                                             #Access 7 th element by going column-wise                             
my_matrix[c(1,3)]                                        #creates matrix with data 1,3
my_matrix[,c(2,3)]                                       #prints with column 2,3
my_matrix[,2:3]                                          #prints matrix with column 2,3
my_matrix[1:2,]                                          #prints matrix with rows 1,2



my_big_matrix = matrix(c(seq(from=-98,to=100,by=2)),     #Creates big matrix
                       nrow=10,ncol=10)
my_big_matrix
?apply
apply(my_big_matrix,2,mean)                              #Calculate particular function over each row or column
apply(my_big_matrix,1,sum)                               #row=1 , column=2
apply(my_big_matrix,1,function(x)mean(x[x>0]))           #Apply Function Mean

#List

my_list=list("look",2,TRUE)                             #Creates List
my_list                                                 #prints list
n=c(2,3,5)                                              
s=c("aa","bb","cc","dd","ee")
b=c(TRUE,FALSE,TRUE,FALSE,FALSE)
x=list(n,s,b,3)

my_list[2]
x[3] 
x
v=list(bob=c(2,3,5),john=c("aa","bb"))
v
v$bob
v$john













