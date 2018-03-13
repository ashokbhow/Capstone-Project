setwd("C:/Data Science/Ryerson Capstone Project CKME136/My Capstone")
getwd()

## Reading the data file
eeg <- read.csv("EEG_data_1.csv", header = TRUE, sep = ",")
#View(eeg)

## Data basics
head(eeg)
tail(eeg)
summary(eeg)
str(eeg)

## Number of instances in whole data set
nrow(eeg)

## separating the class 1 values by slicing for y=1
eeg_1 <- eeg[eeg$y==1, c(1:179)]
nrow(eeg_1)
#View(eeg_1)
## Save Class 1 Epileptic data as a separate data file
## write.csv(eeg_1, "eeg_class1.csv")

## separating the class 2 values by slicing for y=2
eeg_2 <- eeg[eeg$y==2, c(1:179)]
nrow(eeg_2)
## write.csv(eeg_2, "eeg_class2.csv")

## separating the class 3 values by slicing for y=3
eeg_3 <- eeg[eeg$y==3, c(1:179)]
nrow(eeg_3)
## write.csv(eeg_3, "eeg_class3.csv")

## separating the class 3 values by slicing for y=3
eeg_3 <- eeg[eeg$y==3, c(1:179)]
nrow(eeg_3)
## write.csv(eeg_3, "eeg_class3.csv")

## separating the class 4 values by slicing for y=4
eeg_4 <- eeg[eeg$y==4, c(1:179)]
nrow(eeg_4)
## write.csv(eeg_4, "eeg_class4.csv")

## separating the class 4 values by slicing for y=5
eeg_5 <- eeg[eeg$y==5, c(1:179)]
nrow(eeg_5)
## write.csv(eeg_5, "eeg_class5.csv")

## Separating class 1 vs. all
eeg_c1 <- eeg[, c(1:178)]
eeg_c2 <- eeg[, c(178:179)]

## reassigning class values
eeg_c2[eeg_c2$y == 2, ] <- 0
eeg_c2[eeg_c2$y == 3, ] <- 0
eeg_c2[eeg_c2$y == 4, ] <- 0
eeg_c2[eeg_c2$y == 5, ] <- 0

# recombining the data set as 1 vs 0 class i.e. epileptic vs. all non epileptic
eeg_class <- cbind(eeg_c1, eeg_c2)
eeg_class <- eeg_class[, -179]
#View(eeg_class)
## write.csv(eeg_class, "eeg_class.csv")

# creating datafile without class attributes
eeg_p <- eeg[, -179]
#View(eeg_p)
## write.csv(eeg_p, "eeg_p.csv")

# Finding correlation betwwen attributes
C <- cor(eeg_p)
library(corrplot)
corrplot(C, method="circle")
## There is no correlation between attributes

library(caret)
library(class)
library(gmodels)
library(party)
library(partykit)
library(glmulti)

## Logistic regression epileptic vs all other class
eeg_logit <- glm(y ~ X1+X2+X3+X4+X5+X6+X6+X7+X8+X9+X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X25+X26+X27+X28+X29+X30+X31+X32+X33+X34+X35+X36+X37+X38+X39+X40+X41+X42+X43+X44+X45+X46+X47+X48+X49+X50+X51+X52+X53+X54+X55+X56+X57+X58+X59+X60+X61+X62+X63+X64+X65+X66+X67+X68+X69+X70+X71+X72+X73+X74+X75+X76+X77+X78+X79+X80+X81+X82+X83+X84+X85+X86+X87+X88+X89+X90+X91+X92+X93+X94+X95+X96+X97+X98+X99+X100+X101+X102+X103+X104+X105+X106+X107+X108+X109+X110+X111+X112+X113+X114+X115+X116+X117+X118+X119+X120+X121+X122+X123+X124+X125+X126+X127+X128+X129+X130+X131+X132+X133+X134+X135+X136+X137+X138+X139+X140+X141+X142+X143+X144+X145+X146+X147+X148+X149+X150+X151+X152+X153+X154+X155+X156+X157+X158+X159+X160+X161+X162+X163+X164+X165+X166+X167+X168+X169+X170+X171+X172+X173+X174+X175+X176+X177+X178, data=eeg_class, family = "binomial")
summary(eeg_logit)

eeg_logit <- glm(y ~ X4+X5+X6+X7+X9+X10+X11+X13+X23+X24+X26+X27+X42+X49+X50+X51+X52+X55+X56+X60+X61+X63+X64+X71+X72+X79+X86+X92+X93+X95+X99+X100+X102+X103+X105+X107+X108+X119+X120+X127+X133+X136+X137+X139+X144+X147+X148+X149+X152+X153+X155+X157+X158+X159+X170+X174+X175+X177+X178, data=eeg_class, family = "binomial")
summary(eeg_logit)

# Final number of significant features (Total : 27)
eeg_logit <- glm(y ~ X4+X7+X9+X10+X13+X23+X24+X26+X27+X42+X99+X100+X102+X107+X108+X119+X120+X127+X147+X148+X149+X152+X153+X155+X157+X158+X159, data=eeg_class, family = "binomial")
summary(eeg_logit)

## Best valued features
eeg_logit_best <- glm(y ~ X4+X7+X9+X10+X23+X24+X26+X99+X100+X102, data=eeg_class, family = "binomial")
summary(eeg_logit_best)
plot(eeg_logit_best)

## Selecting important features from original file
eeg_select <- eeg[, c(4, 7, 9, 10, 23, 24, 26, 99, 100, 102, 179)]
#View(eeg_select)
nrow(eeg_select)
head(eeg_select)
## write.csv(eeg_select, "eeg_select.csv")

## glmulti with selected features
eeg_select_glmulti <- glmulti(y ~ X4+X7+X23+X24, data=eeg_select, name="glmulti.analysis", intercept = TRUE, marginality = FALSE, plotty=TRUE)
#summary(eeg_select_glmulti)
plot.glmulti(eeg_select_glmulti)
eeg_select_glmulti@formulas
summary(eeg_select_glmulti@objects[[1]])

##
## Principal Componet Analysis on the whole data set
library(lattice)
eeg_pca <- princomp(eeg_p, cor = TRUE)
summary(eeg_pca)
screeplot(eeg_pca, type = "line", ylim = c(0, 10))
eeg_pca_scores <- eeg_pca$scores
#View(eeg_pca_scores)
nrow(eeg_pca_scores)
## write.csv(eeg_pca_scores, "eeg_pca_scores.csv")

## keeping the primary componenets with epileptic vs nonepileptic class
#View(eeg_class)
x <- eeg_class[, c(178, 179)]
#View(x)
nrow(x)
eeg_pca_out <- cbind(eeg_pca_scores, x)
eeg_pca_out <- eeg_pca_out[, -179]
eeg_pca_out <- eeg_pca_out[, -(41:178)]
#View(eeg_pca_out)
## write.csv(eeg_pca_out, "eeg_pca_out.csv")

## keeping the primary componenets with all 1 2 3 4 5 classes
y <- eeg[, c(178, 179)]
eeg_pca_out_all <- cbind(eeg_pca_scores, y)
eeg_pca_out_all <- eeg_pca_out_all[, -(41:178)]
eeg_pca_out_all <- eeg_pca_out_all[, -41]
#View(eeg_pca_out_all)
## write.csv(eeg_pca_out_all, "eeg_pca_out_all.csv")

## Logistic regression with the significant principal components
hist(eeg_pca_out$Comp.1)
eeg_pca_logit <- glm(y ~ Comp.1 + Comp.5 + Comp.6 + Comp.7 + Comp.17 + Comp.29 + Comp.38, data=eeg_pca_out, family="binomial")
summary(eeg_pca_logit)

## Plotting the values
plot(eeg_pca_logit)


## glmulti investigation on the pca dataset with all 5 classes
eeg_pca_glmulti_1 <- glmulti(y ~ Comp.1+Comp.5+Comp.6+Comp.7, data=eeg_pca_out_all, name="glmulti.analysis", intercept = TRUE, marginality = FALSE, plotty=TRUE)
eeg_pca_glmulti_1@formulas
summary(eeg_pca_glmulti_1@objects[[1]])
#summary(eeg_pca_glmulti_1)
plot.glmulti(eeg_pca_glmulti_1)

eeg_pca_glmulti_2 <- glmulti(y ~ Comp.1+Comp.17+Comp.29+Comp.38, data=eeg_pca_out_all, name="glmulti.analysis", intercept = TRUE, marginality = FALSE, plotty=TRUE)
#summary(eeg_pca_glmulti_2)
plot.glmulti(eeg_pca_glmulti_2)
coef.glmulti(eeg_pca_glmulti_2)
eeg_pca_glmulti_2@formulas
summary(eeg_pca_glmulti_2@objects[[1]])

## end

