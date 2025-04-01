data<-read.csv("C:/Users/Admin/Desktop/ML Mini Project/Fish.csv")

#Give path to save pdf on Machine 
pdf("C:\\Users\\Admin\\Desktop\\Mini-Project\\plots.pdf")

#head(data)
str(data)

#Checking Type of Data
is.matrix(data)
is.list(data)
is.array(data)
is.data.frame(data)
dim(data)

#Checking for missing values
is.na(data)

#Checking Numerical Summary of Data 
summary(data)


#histogram
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for(i in n)
{
  hist(data_n[,i],main=colnames(data)[i],xlab = colnames(data)[[i]],col="lightblue")
}
par(op)



#Histogram with Normal Distribution Curve
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for (i in n) {
  mean_i <- mean(data_n[, i])
  sd_i <- sd(data_n[, i])
  
  hist(data_n[,i], 
       main = colnames(data)[i], 
       xlab = colnames(data)[i], 
       col = "lightblue",freq=FALSE)
  
  x <- seq(min(data_n[,i]), max(data_n[,i]), length = length(data_n[,i]))
  y <- dnorm(x, mean = mean_i, sd = sd_i)
  lines(x, y, col = "blue", lwd = 2,lty=2)
}
par(op)


#Density plot
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for (i in n) {
  col_name <- colnames(data)[i]
  
  plot(density(data_n[, i]), main = col_name, ylab = colnames(data)[2], xlab = col_name,col="black")
}
par(op)

#Boxplot
n <- c(3:7)
op <- par(mfrow = c(1, 5))
for(i in n) 
{ 
  boxplot(data[,i],main=colnames(data)[i],col="salmon")
}
par(op)

#Boxplot
par(mfrow = c(1, 1))
boxplot(data_n[,3:7],col="salmon")


#Boxplot of Length 
n <- c(3:5)
op <- par(mfrow = c(1, 3))
for(i in n) 
{ 
  boxplot(data[,i],main=colnames(data)[i],col="salmon")
}
par(op)

#Boxplot Bivariant numerical vs catgorical
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for(i in n) 
{ 
  boxplot(data[,i]~data[,1] ,main=colnames(data)[i],col="salmon",xlab=colnames(data)[i],ylab=colnames(data)[1])
}
par(op)

#qqplot and qqline

n <- c(3:7)
op <- par(mfrow = c(2, 3))
for(i in n) 
{ 
  qqplot(data[,i],data[,2],main = paste("Q-Q Plot ",colnames(data)[i], "vs", colnames(data)[2]))
  qqline(data[,i],col="red")
}
par(op)

#qqnorm and qqline
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for(i in n) 
{ 
  qqnorm(data[,i],main = paste("Q-Q norm",colnames(data)[i]))
  qqline(data[,i],col="red")
}
par(op)

#pairplot

#convertion and to count no. of species 
category_factor <- as.factor(data[, 1])  
num_categories <- length(levels(category_factor)) 
#Manual colors
color_palette <-c("red", "blue", "green", "purple", "orange", "brown", "pink")
category_colors <- color_palette[as.numeric(category_factor)]  
#pairplot
pairs(data[, 3:7],  
      main = "Pairplot of Variables",  
      pch = 19,  
      col = category_colors,   
      cex = 0.7,  
      upper.panel = NULL)
#label for color and species
legend("topright", legend = levels(category_factor),  
       col = color_palette, pch = 19, title = "Category")


# Compute correlation matrix
cor_matrix <- cor(data[, 3:7], use = "complete.obs")
cor_matrix[lower.tri(cor_matrix)] <- NA 
cor_matrix

# Create custom color palette
col_palette <- colorRampPalette(c("navyblue", "white", "firebrick3"))(256)
heatmap(cor_matrix,
        col = col_palette,
        main = "Correlation Heatmap",
        symm = TRUE,
        Rowv = NA, Colv = NA, # Disable clustering
        revC = TRUE, # Proper diagonal alignment
        labRow = colnames(cor_matrix), # Show row labels
        labCol = colnames(cor_matrix)
)


# Correlation between All predictors
cor(data[, 3:7], data[, 3:7])

# Correlation between response and predictor varible
cor(data[, 2], data[, 3:7])


#Correlation with Confidence interval and p values 
numeric_data <- data[, 2:7]
cor_results <- data.frame(Var1 = character(), Var2 = character(),
                          Correlation = numeric(), P_value = numeric(),
                          CI_Lower = numeric(), CI_Upper = numeric(),
                          stringsAsFactors = FALSE)

for (i in 1:(ncol(numeric_data) - 1)) {
  for (j in (i + 1):ncol(numeric_data)) {
    
    # Perform correlation test
    test <- cor.test(numeric_data[, i], numeric_data[, j], method = "pearson")
cor_results <- rbind(cor_results, data.frame(
  Var1 = colnames(numeric_data)[i],
  Var2 = colnames(numeric_data)[j],
  Correlation = round(test$estimate, 3),  
  P_value = format(test$p.value, scientific = TRUE, digits = 5),  
  CI_Lower = round(test$conf.int[1], 3), 
  CI_Upper = round(test$conf.int[2], 3)  
))
  }
}
print(cor_results)


# Scatterplot
n <- c(3:7)

op <- par(mfrow = c(2, 3))
for (i in n) {
  col_name <- colnames(data)[i]
  
  plot(data[, i], data[, 2], main = paste("scatterplot",col_name, " vs Weight"), ylab = "Weight (gm)", xlab = paste(col_name, "(cm)"))
}
par(op)

# Fitting Linear Regression To Each Predictor Different
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for (i in n)
{
  plot(data[, 2] ~ data[, i], xlab = colnames(data)[i], ylab = colnames(data)[2],main="Linear fit")
  lf <- lm(data[, 2] ~ data[, i])
  summary(lf)
  abline(coef(lf), col = "red")
}
par(op)


# Linear Regression Residuals
n <- c(3:7)
op <- par(mfrow = c(2, 3))
for (i in n) {
  plot(data[, i], resid(lf),
       main = paste("Residuals vs", colnames(data)[i]),
       xlab = colnames(data)[i], ylab = "Residuals"
  )
  
  abline(coef(lf), col = "red")
}
par(op)


# Fitting Full Model
n <- c(3:7)
Targetvariable <- data[, 2]
model_results <- data.frame(
  Predictor_1 = character(),
  Predictor_2 = character(),
  Predictor_3 = character(),
  R_squared = numeric(),
  AIC = numeric(),
  BIC = numeric(),
  stringsAsFactors = FALSE
)
for (i in n) {
  for (j in n) {
    for (k in n) {
      if (i < j & j < k) {  
        pred1 <- colnames(data)[i]
        pred2 <- colnames(data)[j]
        pred3 <- colnames(data)[k]
        
        formula <- as.formula(paste("Targetvariable ~", pred1, "+", pred2, "+", pred3))
        model <- lm(formula, data = data)
        
        r_squared <- summary(model)$r.squared
        aic_value <- AIC(model)
        bic_value <- BIC(model)
        
        model_results <- rbind(model_results, data.frame(
          Predictor_1 = pred1,
          Predictor_2 = pred2,
          Predictor_3 = pred3,
          R_squared = round(r_squared, 4),
          AIC = round(aic_value, 2),
          BIC = round(bic_value, 2)
        ))
        
        # Plot residuals vs fitted values
        #plot(model, which = 1, main = paste("Model:", pred1, "+", pred2, "+", pred3))
        #title(xlab = "Fitted Values", ylab = "Residuals")  # Add axis labels separately
      }
    }
  }
}
print(model_results)




#From Above obeservation we get best model with following features 
pred_1<-data[,3]
pred_2<-data[,5]
pred_3<-data[,6]

#Fitting Linear Model
lf<-lm( Targetvariable ~ (pred_1+pred_2+pred_3))
summary(lf)
predicted_values <- predict(lf, newdata = data)
plot(data$Weight, predicted_values,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "linear Regression",
     col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red", lwd = 2)

#Fitting logarithmic Model
lf<-lm( Targetvariable ~ log(pred_1)+log(pred_2)+log(pred_3))
summary(lf)
predicted_values <- predict(lf, newdata = data)
plot(data$Weight, predicted_values,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "Logarithmic Regression",
     col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red", lwd = 2)

#Fitting Polynomial Model
lf<-lm( Targetvariable ~ poly(pred_1)+poly(pred_2)+poly(pred_3))
summary(lf)
predicted_values <- predict(lf, newdata = data)
plot(data$Weight, predicted_values,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "Polynomial Regression",
     col = "blue", pch = 16)
abline(a = 0, b = 1, col = "red", lwd = 2)

dev.off()