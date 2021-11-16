library(ggplot2)
library(tidyr)
library(dplyr)
library(zoo)
library(patchwork)

# --- losses

losses_heart <- read.csv("results/final-heart-local_losses.csv")[,1:2]
cn <- c("iteration", "loss")
colnames(losses_heart) <- cn
losses_heart$scenario <- "heart"
losses_apartment <- read.csv("results/final-apartment-global_losses.csv")[,1:2]
cn <- c("iteration", "loss")
colnames(losses_apartment) <- cn
losses_apartment$scenario <- "apartment"

df1 <- rbind(losses_heart, losses_apartment) %>% 
  group_by(scenario) %>% 
  mutate(roll_mean = rollmean(loss, 5, na.pad = T))

p1 <- ggplot(df1) +
  geom_line(aes(x=iteration, y=roll_mean, color=scenario), size=1.5) +
  scale_color_manual(values=DALEX::colors_discrete_drwhy(3)[c(1, 2)]) +
  scale_x_continuous(expand = c(0, 2), limits = c(0, 405)) +
  scale_y_continuous(expand = c(0, 0.02)) +
  theme_bw() +
  theme(legend.position = c(0.85, 0.85)) + 
  labs(y="Loss", x="Iteration", color=NULL) + 
  theme(text = element_text(size=10)) + 
  theme(axis.text.x = element_text(size=9),
        legend.text=element_text(size=10),
        legend.background=element_blank())

p1
ggsave("results/final_losses.pdf", width=5, height=2)


# --- explanations

explanation <- read.csv("results/final-heart-local_explanation.csv")
colnames(explanation)[2] <- "manipulated"
sum(abs(explanation$original - explanation$manipulated))

df2 <- pivot_longer(
  cbind(explanation,
        variable=factor(c("sex: Female", "age: 46", "thalach: 152", 
                          "trestbps: 138", "chol: 243", "oldpeak: 0"),
                        levels=c("age: 46", "sex: Female", "chol: 243", 
                                 "thalach: 152", "trestbps: 138", "oldpeak: 0"))),
  cols=colnames(explanation))

p2 <- ggplot(df2) +
  geom_col(aes(y=value, x=variable, fill=name), position = "dodge") +
  labs(y="Variable Attribution", x=NULL, fill=NULL) + 
  scale_fill_manual(values=c(DALEX::colors_discrete_drwhy(3)[c(2, 1)], "grey")) +
  scale_y_continuous(expand = c(0, 0), limits = c(-0.1, 0.25)) + 
  #scale_x_discrete(guide = guide_axis(n.dodge = 2)) + 
  theme_bw() +
  theme(legend.position = c(0.135, 0.825)) + 
  theme(#axis.title.y = element_text(angle = 0), 
        text = element_text(size=10)) + 
  theme(axis.text.x = element_text(size=8),
        legend.text=element_text(size=10),
        legend.background=element_blank())

ggsave("results/final-heart-local_explanation.pdf", width=5, height=2.5)


explanation <- read.csv("results/final-apartment-global_explanation.csv")
colnames(explanation)[2] <- "manipulated"
sum(abs(explanation$original - explanation$manipulated))

df5 <- pivot_longer(
  cbind(explanation,
        variable=factor(c("baths", "bedrooms", "market", "fee", "school", "sqft", "subway", "rooms"),
                        levels=c("rooms", "subway", "school", "fee", "market", "bedrooms", "sqft", "baths"))),
  cols=colnames(explanation))
df5[df5$name=="manipulated", ]
df5[df5$name=="original", ]

p5 <- ggplot(df5) +
  geom_col(aes(y=value, x=variable, fill=name), position = "dodge") +
  labs(y="Variable Importance", x=NULL, fill=NULL) + 
  scale_fill_manual(values=c(DALEX::colors_discrete_drwhy(3)[c(2, 1)], "grey")) +
  theme_bw() +
  theme(legend.position = c(0.14, 0.825)) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 350000)) + 
  theme(#axis.title.y = element_text(angle = 0), 
        text = element_text(size=10)) + 
  theme(axis.text.x = element_text(size=8),
        legend.text=element_text(size=10),
        legend.background=element_blank())


ggsave("results/final-apartment-global_explanation.pdf", width=5, height=2.5)


# p2 + p5 +
#   plot_layout(widths=c(5, 7)) &
#   theme(plot.margin=unit(c(0.01, 0.3, 0.01, 0.01), "cm"))
# ggsave("results/final_explanation.pdf", width=10, height=2.5)


# --- data

library(philentropy)
library(emdist)

calculate_distance <- function(data, method = "jensen-shannon") {
  xy <- split(data, data$dataset)
  x <- xy[[1]]
  y <- xy[[2]]
  x$dataset <- y$dataset <- NULL
  
  js_distance <- list()
  for (variable in colnames(x)) {
    bins <- nclass.Sturges(x[, variable])
    minval <- min(c(x[, variable], y[, variable]))
    maxval <- max(c(x[, variable], y[, variable]))
    breaks <- seq(minval, maxval, length.out = bins + 1)
    js_distance[variable] <- dist_one_one(
      table(cut(x[, variable], breaks, include.lowest = T)) / dim(x)[1],
      table(cut(y[, variable], breaks, include.lowest = T)) / dim(y)[1],
      method = method
    )
  }
  
  unlist(js_distance)
}

df <- read.csv("results/final-heart-local_data.csv")
ret <- calculate_distance(df)
mean(ret)

df <- read.csv("results/final-apartment-global_data.csv")
ret <- calculate_distance(df)
mean(ret)
