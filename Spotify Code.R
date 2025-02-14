# DATA PREPARATION
# ------------------------------------------------------------------------------

# Loading data obtained from Kaggle
Spotify_Youtube <- read.csv("Spotify_Youtube.csv")
str(Spotify_Youtube)

# Loading packages
library(glmnet)
library(caret)
library(caretEnsemble)
library(elasticnet)
library(psych)
library(car)
library(corrplot)

# Sub setting the columns we want
spotify <- Spotify_Youtube[ ,c(6, 8:18, 28)]

# Getting rid of all missing values
spotify <- spotify[!is.na(spotify$Stream), ]
spotify <- spotify[!is.na(spotify$Duration_ms), ]

# Converting album type into a factor
spotify$Album_type <- as.factor(spotify$Album_type)

# LINEAR REGRESSION
# ------------------------------------------------------------------------------

# Running a linear regression with number of streams as DV and all IV's
linear_regression <- lm(Stream ~ Album_type + Danceability + Energy + Key
                        + Loudness + Speechiness + Acousticness + Instrumentalness +
                          Liveness + Valence + Tempo + Duration_ms, data = spotify, singular.ok = TRUE)
summary(linear_regression)
# Nearly all coefficients are significant
 
# Checking multicollinearity assumption
vif_values <- car::vif(linear_regression)
vif_values
# No perfect multicollinearity

# Looking at correlation between variables
numeric_spotify <- spotify[, sapply(spotify, is.numeric)]  # Select only numeric columns
correlation_matrix <- cor(numeric_spotify)  # Calculate correlation for numeric variables
correlation_matrix
?corrplot
corrplot(correlation_matrix,# Customize the correlation plot
         method = "number",  # Use color to represent correlation values
         type = "upper",    # Display the upper triangle of the matrix
         order = "hclust",  # Reorder variables based on hierarchical clustering
         tl.cex = 0.8,
         number.cex = 0.8,
         # Adjust the size of variable labels
         tl.col = "black",  # Set label color to black
         col = colorRampPalette(c("blue", "white", "red"))(50),  # Define custom color palette
         title = "Correlation Matrix of Variables"
)
# We see a correlation between energy and loudness

# Residuals vs. Fitted Values (Homoscedasticity)
plot(linear_regression, which = 1)

# Normality of Residuals
hist(resid(linear_regression), main = "Histogram of Residuals")
qqnorm(resid(linear_regression))
qqline(resid(linear_regression))

# LOG TRANSFORMATION OF DV
# ------------------------------------------------------------------------------

# Since the assumptions were not met, we made a log transformation on the DV
spotify_log <- spotify
spotify_log$Stream <- log(spotify_log$Stream)

# Running a linear regression with the logged DV
linear_log <- lm(Stream ~ Album_type + Danceability + Energy + Key
                   + Loudness + Speechiness + Acousticness + Instrumentalness +
                     Liveness + Valence + Tempo + Duration_ms, data = spotify_log, singular.ok = TRUE)
summary(linear_log)

# No Perfect Multicollinearity
vif_values_log <- car::vif(linear_log)
print(vif_values_log)

# Residuals vs. Fitted Values (Homoscedasticity)
plot(linear_log, which = 1)

# Normality of Residuals
hist(resid(linear_log), main = "Histogram of Residuals")
qqnorm(resid(linear_log))
qqline(resid(linear_log))

# Looks better now!

# SPLITTING INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------------------------

sample_size <- floor(0.8*nrow(spotify_log))
set.seed(777)
picked = sample(seq_len(nrow(spotify_log)),size=sample_size)
train_spotify = spotify_log[picked, ]
test_spotify = spotify_log[-picked,]

# RIDGE
# ------------------------------------------------------------------------------

# Defining our Y and X variables
Y <- train_spotify$Stream
X <- model.matrix(~ Album_type + Danceability + Energy + Key
                  + Loudness + Speechiness + Acousticness + Instrumentalness +
                    Liveness + Valence + Tempo + Duration_ms, data = train_spotify)
X <- X[, -1]

# Ridge
ridge <- glmnet(X, Y, alpha = 0, lambda = 10^seq(-2, 5, length.out = 50), standardize = FALSE) 
plot(ridge, xvar = "lambda", label = TRUE, las = 1)
legend("bottomright", legend = rownames(coef(ridge)), col = 1:nrow(coef(ridge)), lwd = 2, 
       cex = 0.60)

# Cross validation
ridge <- cv.glmnet(X, Y, alpha = 0, lambda = 10^seq(-2, 5, length.out = 50), nfolds = 10) 
ridge$lambda.min    
ridge$lambda.1se    

# Best ridge
ridge_best <- glmnet(X, Y, alpha = 0, lambda = ridge$lambda.1se) 
coef(ridge_best)

# LASSO
# ------------------------------------------------------------------------------

# Lasso
lasso <- glmnet(X, Y, alpha = 1, lambda = 10^seq(-2, 5, length.out = 50), standardize = FALSE)
plot(lasso, xvar = "lambda", label = TRUE, las = 1)
legend("bottomright", legend = rownames(coef(ridge)), col = 1:nrow(coef(ridge)), lwd = 2, 
       cex = 0.60)

# Cross validation
lasso <- cv.glmnet(X, Y, alpha = 1, lambda = 10^seq(-2, 5, length.out = 50), nfolds = 10) 
lasso$lambda.min    
lasso$lambda.1se 

# Best lasso
lasso_best <- glmnet(X, Y, alpha = 1, lambda = lasso$lambda.1se) 
coef(lasso_best)

# library(stargazer)
# stargazer(lasso_best, type = "text", title="Coefficients Lasso", digits=2, out="table1.txt")
# write.table(coef(lasso_best), file = "lasso_coefficients.txt", row.names = TRUE)

# Lasso regression gets rid of 3 variables (key, tempo, and duration)

# ELASTIC NET
# ------------------------------------------------------------------------------

# Elastic net
elastic <- glmnet(X, Y, alpha = 0.5, 
                  lambda = 10^seq(-2, 5, length.out = 50))
plot(elastic, xvar = "lambda", label = TRUE, las = 1)
legend("bottomright", legend = rownames(coef(ridge)), col = 1:nrow(coef(ridge)), lwd = 2, 
       cex = 0.60)

# Cross validation
elastic <- cv.glmnet(X, Y, alpha = 0.5, lambda = 10^seq(-2, 5, length.out = 50)) 
elastic$lambda.min    
elastic$lambda.1se 

# Best elastic net
elastic_best <- glmnet(X, Y, alpha = 0.15, lambda = elastic$lambda.1se) 
coef(elastic_best)

predictions <- predict(elastic_best, s = "lambda.1se", newx = X)
rmse <- sqrt(mean((test_spotify$Stream - predictions)^2))
rmse

predictions2 <- predict(ridge_best, s = "lambda.1se", newx = X)
rmse2 <- sqrt(mean((test_spotify$Stream - predictions2)^2))
rmse2

predictions3 <- predict(lasso_best, s = "lambda.1se", newx = X)
rmse3 <- sqrt(mean((test_spotify$Stream - predictions3)^2))
rmse3

## Elastic Net with ideal alpha 
lambda_seq <- 10^seq(-2, 5, length.out = 50)

# Initialize variables to store results
best_alpha <- NULL
min_cv_error <- Inf

# Loop through alpha values and perform cross-validation
for (alpha_val in alpha_seq) {
  cv_result <- cv.glmnet(X, Y, alpha = alpha_val, lambda = lambda_seq)
  
  # Find the lambda that minimizes cross-validation error
  min_lambda_idx <- which.min(cv_result$cvm)
  cv_error <- cv_result$cvm[min_lambda_idx]
  
  # Check if the current alpha has a lower cross-validation error
  if (cv_error < min_cv_error) {
    best_alpha <- alpha_val
    min_cv_error <- cv_error
  }
}

# Fit Elastic Net with the best alpha and lambda
elastic_best_alpha <- glmnet(X, Y, alpha = best_alpha, lambda = lambda_seq)
# Print coefficients
coef(elastic_best_alpha)
best_alpha

# COMPARISON
# ------------------------------------------------------------------------------

# Creating a table with the coefficients for each model
comparison <- cbind(coef(ridge_best), coef(lasso_best), coef(elastic_best))
colnames(comparison) <- c("Ridge", "Lasso", "Elastic net")
round(comparison, digits = 2)





elastic$cvm <- elastic$cvm^0.5
elastic$cvup <- elastic$cvup^0.5
elastic$cvlo <- elastic$cvlo^0.5
plot(elastic, ylab = "Root Mean Squared Error")

lasso$cvm <- lasso$cvm^0.5
lasso$cvup <- lasso$cvup^0.5
lasso$cvlo <- lasso$cvlo^0.5
plot(lasso, ylab = "Root Mean Squared Error")

ridge$cvm <- ridge$cvm^0.5
ridge$cvup <- ridge$cvup^0.5
ridge$cvlo <- ridge$cvlo^0.5
plot(ridge, ylab = "Root Mean Squared Error")

# listOfModels <- list(lasso_best, elastic_best, linear_log, ridge_best)
# res <- resamples(listOfModels)
# summary(res) 
# xyplot(res, metric = 'RMSE')

# RMSE lasso
lasso_lambda_1se <- lasso$lambda.1se
lasso_rmse <- min(lasso$cvm[lasso$lambda == lasso_lambda_1se])
print("Lasso RMSE using lambda.1se:")
print(sqrt(lasso_rmse))

# RMSE ridge
ridge_lambda_1se <- ridge$lambda.1se
ridge_rmse <- min(ridge$cvm[ridge$lambda == ridge_lambda_1se])
print("Ridge RMSE using lambda.1se:")
print(sqrt(ridge_rmse))

# RMSE elastic net
elastic_lambda_1se <- elastic$lambda.1se
elastic_rmse <- min(elastic$cvm[elastic$lambda == elastic_lambda_1se])
print("Elastic Net RMSE using lambda.1se:")
print(sqrt(elastic_rmse))

# Importance plot
install.packages("vip")
library(vip)
vip(elastic_best)


# ------------------------------------------------------------------------------

# # Linearity (Partial Residual Plots)
# library(car)
# crPlots(linear_regression)
# 
# # No Autocorrelation (Durbin-Watson Test)
# library(lmtest)
# dwtest(linear_regression)
# 
# # No Endogeneity (Ramsey RESET Test)
# resettest(linear_regression)
# 
# # No Heteroscedasticity (Breusch-Pagan Test)
# bptest(linear_regression)
# 
# # Independence of Errors (Ljung-Box Test for Autocorrelation)
# Box.test(linear_regression$residuals, type = "Ljung-Box")
# 
# # Check for Outliers (Cook's Distance)
# cooksd <- cooks.distance(linear_regression)
# plot(cooksd, pch = 19, cex = 1, main = "Cook's Distance Plot")
# abline(h = 4/length(linear_regression$residuals), col = "red")
# 
# # Check for Influence (Leverage and Influence Plot)
# influencePlot(linear_regression)
# 
# # Check for Normality of Predicted Values (Shapiro-Wilk Test)
# shapiro.test(predict(linear_regression))
# 
# # Check for Homoscedasticity in Predicted Values
# plot(predict(linear_regression), resid(linear_regression), main = "Residuals vs. Fitted Values")
# abline(h = 0, col = "red")

# linear_selection <- lm(Stream ~ Album_type + Danceability
#                        + Loudness + Speechiness + Acousticness + Instrumentalness +
#                          Liveness + Valence, data = spotify_log, singular.ok = TRUE)
# 
# summary(linear_selection)

# correlation_matrix <- cor(spotify[, c("Danceability", "Energy", "Key",
#                                       "Loudness", "Speechiness", "Acousticness", "Instrumentalness",
#                                       "Liveness", "Valence", "Tempo", "Duration_ms")])
# plot(correlation_matrix)
# spotify_num <- spotify[ , sapply(spotify, is.numeric)]
# cor(spotify_num, upper.panel = TRUE)
# str(spotify)

# # Fit the linear regression model
# linear_regression <- lm(Stream ~ Danceability + Energy + Key +
#                           Loudness + Speechiness + Acousticness + Instrumentalness +
#                           Liveness + Valence + Tempo + Duration_ms, data = spotify, singular.ok = TRUE)
