
op <- par(mfrow=c(3,5))
n<-c(150,300,450,600,750,900,1200,1500,1700,2000,2300,2500,2700,3100,3500)
 for(i in n) 
 {
   x<- rnorm(n) 
   hist(x,probability=TRUE,main=paste("n=",i)) 
   curve(dnorm(x,mean=mean(x),sd=sd(x)),col="red",add=TRUE)
 }


op <- par(mfrow=c(4,5))
for(i in 1:20) 
{
  x<- rnorm(100) 
  plot(x) 
  curve(dnorm(x,mean=mean(x),sd=sd(x)),col="red",add=TRUE)
}