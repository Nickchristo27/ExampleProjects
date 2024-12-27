#Install and load the necessary packages to access the chosen data.
install.packages("NHANES")
install.packages("boot")
install.packages("tinytex")
tinytex::install_tinytex()
library(dplyr)
library(boot)
library(NHANES)

data("NHANES")

#Data is now loaded. Preview all the data for the first person.
NHANES[1, 1:76]

#Delete duplicate entries.
NHANES=subset(NHANES, !duplicated(NHANES$ID))

#Filter out individuals below the age of 18.
NHANES=subset(NHANES, Age>=18)

#Show the new dimension of the data after filtration.
dim(NHANES)

#This shows us all the data that will be available for each person.
#For our hypothesis, let us create a data frame containing only relevant data.
dataframe=data.frame(
  AlcoholYear=NHANES$AlcoholYear,
  HardDrugs=NHANES$HardDrugs,
  HouseholdIncome=NHANES$HHIncomeMid
  )

#Replace Yes and No in HardDrugs with a binary variable.
dataframe$HardDrugs=gsub("Yes", 1, dataframe$HardDrugs)
dataframe$HardDrugs=gsub("No", 0, dataframe$HardDrugs)

#Preview the new data frame.
head(dataframe)

#We should deal with missing data in any column by removing it. 
dataframe=na.omit(dataframe)

#Again, show the new dimension of the data frame after filtration.
dim(dataframe)

#Provide some insight into the data.
table(dataframe$AlcoholYear)
table(dataframe$HardDrugs)
table(dataframe$HouseholdIncome)

#Create a function that will be used to re-sample the data along with boot().
resampleFunc = function(dataframe, indicies){
  boot_data=dataframe[indicies,]
  model=lm(HouseholdIncome~HardDrugs + AlcoholYear, data=boot_data)
  return(coef(model)[2])
}

#Perform the bootstrapping.
resampled=boot(data=dataframe, statistic=resampleFunc, R=5000)

#View the distribution of the coefficients
plot(resampled)

#Confirm the normality assumption with Anderson-Darling
ad.test(resampled$t)

#View the confidence interval to determine if there is significance.
boot.ci(resampled, type="perc", conf = 0.99)

#At this point, we would like to test the effects on our results if we imputed
#values for AlcoholYear based on other categories.

#Create data frame for alternative testing.
dataframe_alt=data.frame(
  AlcoholYear=NHANES$AlcoholYear,
  Alcohol12PlusYear=NHANES$Alcohol12PlusYr,
  HardDrugs=NHANES$HardDrugs,
  HouseholdIncome=NHANES$HHIncomeMid
)

#Delete missing values from Alcohol12PlusYear and HouseholdIncome.
dataframe_alt=subset(dataframe_alt, !is.na(Alcohol12PlusYear))
dataframe_alt=subset(dataframe_alt, !is.na(HouseholdIncome))

#Replace Yes and No in HardDrugs with a binary variable.
dataframe_alt$HardDrugs=gsub("Yes", 1, dataframe_alt$HardDrugs)
dataframe_alt$HardDrugs=gsub("No", 0, dataframe_alt$HardDrugs)

#Replace missing values in AlcoholYear with 6 if Alcohol12PlusYear is 0 and 
#replace missing values in HardDrugs with 0 if Alcohol12Plus Year is 0.
dataframe_alt=dataframe_alt %>% mutate(
  AlcoholYear=ifelse(
    Alcohol12PlusYear=="No" & is.na(AlcoholYear), 6, AlcoholYear),
  HardDrugs=ifelse(
    Alcohol12PlusYear=="No" & is.na(HardDrugs), "0", HardDrugs)
)

#Now remove potential remaining missing values that cannot be imputed.
dataframe_alt=na.omit(dataframe_alt)

#We gained an extra 516 values by imputing data based on the criteria above.
dim(dataframe_alt)

#Redo the bootstrapping.
resampleFunc_alt = function(dataframe_alt, indicies){
  boot_data=dataframe_alt[indicies,]
  model=lm(HouseholdIncome~HardDrugs + AlcoholYear, data=boot_data)
  return(coef(model)[2])
}
resampled_alt=boot(data=dataframe_alt, statistic=resampleFunc_alt, R=5000)

#Review distribution of coefficients and check for normality.
plot(resampled_alt)
ad.test(resampled_alt$t)

#Provide confidence interval of the coefficient.
boot.ci(resampled_alt, type="perc", conf = 0.99)
