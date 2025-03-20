##from utils import db_connect
##engine = db_connect()


# your code here
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest

# Importar el dataset

total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")

total_data.to_csv("../data/raw/total_data.csv", index = False)

print(total_data.shape)

total_data.info()

## Eliminar duplicados

if total_data.drop("id", axis = 1).duplicated().sum() > 0:
    total_data.drop_duplicates(subset = total_data.columns.difference(['id']))

print(total_data.shape)

## Eliminar información irrelevante

total_data.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

total_data.info()

# Análisis de variables univariante

## Categóricas

fig, axis = plt.subplots(2, 3, figsize = (10, 7))

unique_values_neighbourhood_group = total_data["neighbourhood_group"].unique()
unique_values_room_type = total_data["room_type"].unique()

sns.histplot(ax = axis[0, 0], data = total_data, x = "host_id")
sns.histplot(ax = axis[0, 1], data = total_data, x = "neighbourhood_group")
axis[0, 1].set_xticks(unique_values_neighbourhood_group) 
axis[0, 1].set_xticklabels(unique_values_neighbourhood_group, rotation=45)
sns.histplot(ax = axis[0, 2], data = total_data, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1, 0], data = total_data, x = "room_type")
axis[1, 0].set_xticks(unique_values_room_type)
axis[1, 0].set_xticklabels(unique_values_room_type, rotation=45)
sns.histplot(ax = axis[1, 1], data = total_data, x = "availability_365")

fig.delaxes(axis[1, 2])

plt.tight_layout()

plt.show()

## Numéricas

fig, axis = plt.subplots(6, 2, figsize=(10, 16), gridspec_kw={"height_ratios": [6, 1] * 3})

sns.histplot(ax=axis[0, 0], data=total_data, x="latitude")
sns.boxplot(ax=axis[1, 0], data=total_data, x="latitude")

sns.histplot(ax=axis[0, 1], data=total_data, x="longitude")
sns.boxplot(ax=axis[1, 1], data=total_data, x="longitude")

sns.histplot(ax=axis[2, 0], data=total_data, x="price")
sns.boxplot(ax=axis[3, 0], data=total_data, x="price")

sns.histplot(ax=axis[2, 1], data=total_data, x="minimum_nights")
sns.boxplot(ax=axis[3, 1], data=total_data, x="minimum_nights")

sns.histplot(ax=axis[4, 0], data=total_data, x="number_of_reviews")
sns.boxplot(ax=axis[5, 0], data=total_data, x="number_of_reviews")

sns.histplot(ax=axis[4, 1], data=total_data, x="calculated_host_listings_count")
sns.boxplot(ax=axis[5, 1], data=total_data, x="calculated_host_listings_count")

plt.tight_layout()

plt.show()

# Análisis de variables multivariante

## Categórico - categórico

### Room type y neighbourhood group
fig, axis = plt.subplots(figsize = (5, 5))

sns.countplot(data = total_data, x = "room_type", hue = "neighbourhood_group")

plt.show()

### Room type y availability

fig, axis = plt.subplots(figsize = (5, 3))

sns.countplot(data = total_data, x = "room_type", hue = "availability_365")

axis.set_ylim(0, 300) 

plt.show()

### neighbourhood group y availability

fig, axis = plt.subplots(figsize=(6, 3))

sns.countplot(data=total_data, x="neighbourhood_group", hue="availability_365", ax=axis)

axis.set_ylim(0, 200) 

plt.show()

### Correlaciones

total_data["neighbourhood_group_n"] = pd.factorize(total_data["neighbourhood_group"])[0]
total_data["room_type_n"] = pd.factorize(total_data["room_type"])[0]
total_data["neighbourhood_n"] = pd.factorize(total_data["neighbourhood"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[["host_id", "neighbourhood_group_n" , "neighbourhood_n" , "room_type_n" , "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

## Transformaciones de categórico a numérico

neighbourhood_group_transformation_rules = {row["neighbourhood_group"]: row["neighbourhood_group_n"] for _, row in total_data[["neighbourhood_group", "neighbourhood_group_n"]].drop_duplicates().iterrows()}
with open("neighbourhood_group_transformation_rules.json", "w") as f:
  json.dump(neighbourhood_group_transformation_rules, f)

room_type_transformation_rules = {row["room_type"]: row["room_type_n"] for _, row in total_data[["room_type", "room_type_n"]].drop_duplicates().iterrows()}
with open("room_type_transformation_rules.json", "w") as f:
  json.dump(room_type_transformation_rules, f)

neighbourhood_transformation_rules = {row["neighbourhood"]: row["neighbourhood_n"] for _, row in total_data[["neighbourhood", "neighbourhood_n"]].drop_duplicates().iterrows()}
with open("neighbourhood_transformation_rules.json", "w") as f:
  json.dump(neighbourhood_transformation_rules, f)

## Análisis Numérico-numérico

fig, axis = plt.subplots(4, 2, figsize = (10, 16))

sns.regplot(ax = axis[0, 0], data = total_data, x = "minimum_nights", y = "price")
sns.heatmap(total_data[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = total_data, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(total_data[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = total_data, x = "calculated_host_listings_count", y = "price").set(ylabel = "price")
sns.heatmap(total_data[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)

sns.regplot(ax = axis[2, 1], data = total_data, x = "minimum_nights", y = "number_of_reviews").set(ylabel = "number_of_reviews")
sns.heatmap(total_data[["minimum_nights", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 1]).set(ylabel = None)

plt.tight_layout()

plt.show()

## Numérico - Categórico

fig, axes = plt.subplots(figsize=(10, 7))

sns.heatmap(total_data[["neighbourhood_group_n", "neighbourhood_n", "room_type_n", "price", "minimum_nights", "number_of_reviews",
                         "calculated_host_listings_count", "availability_365"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(figsize = (10, 5))

sns.regplot(data = total_data, x = "room_type_n", y = "price")


plt.tight_layout()

plt.show()

## Pairplot

sns.pairplot(data = total_data)

# Ingeniería de características

total_data.describe()

fig, axis = plt.subplots(2, 3, figsize = (12, 8))

sns.boxplot(ax = axis[0, 0], data = total_data, y = "price")
sns.boxplot(ax = axis[0, 1], data = total_data, y = "minimum_nights")
sns.boxplot(ax = axis[0, 2], data = total_data, y = "number_of_reviews")
sns.boxplot(ax = axis[1, 0], data = total_data, y = "calculated_host_listings_count")
sns.boxplot(ax = axis[1, 1], data = total_data, y = "availability_365")
sns.boxplot(ax = axis[1, 2], data = total_data, y = "neighbourhood_n")

plt.tight_layout()

plt.show()

## Análisis y reemplazar outliers

total_data_con_outliers = total_data.copy()
total_data_sin_outliers = total_data.copy()

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr
  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}
for column in ["price", "minimum_nights", "number_of_reviews", "calculated_host_listings_count"]:
  total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
  outliers_dict[column] = limits_list

with open("outliers_replacement.json", "w") as f:
    json.dump(outliers_dict, f)

## Análisis de valores faltantes

total_data_con_outliers.isnull().sum().sort_values(ascending=False)
total_data_sin_outliers.isnull().sum().sort_values(ascending=False)

## Escalado de valores

num_variables = ["minimum_nights", "number_of_reviews","calculated_host_listings_count", "availability_365", "neighbourhood_group_n", "room_type_n"]


X_con_outliers = total_data_con_outliers.drop("price", axis = 1)[num_variables]
X_sin_outliers = total_data_sin_outliers.drop("price", axis = 1)[num_variables]
y = total_data_con_outliers["price"]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)


X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

## Normalización

from sklearn.preprocessing import StandardScaler

scaler_con_outliers = StandardScaler()
scaler_con_outliers.fit(X_train_con_outliers)

X_train_con_outliers_norm = scaler_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_norm = pd.DataFrame(X_train_con_outliers_norm, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_norm = scaler_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_norm = pd.DataFrame(X_test_con_outliers_norm, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_norm.to_excel("../data/processed/X_train_con_outliers_norm.xlsx", index = False)
X_test_con_outliers_norm.to_excel("../data/processed/X_test_con_outliers_norm.xlsx", index = False)

scaler_sin_outliers = StandardScaler()
scaler_sin_outliers.fit(X_train_sin_outliers)

X_train_sin_outliers_norm = scaler_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_norm = pd.DataFrame(X_train_sin_outliers_norm, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_norm = scaler_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_norm = pd.DataFrame(X_test_sin_outliers_norm, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_norm.to_excel("../data/processed/X_train_sin_outliers_norm.xlsx", index = False)
X_test_sin_outliers_norm.to_excel("../data/processed/X_test_sin_outliers_norm.xlsx", index = False)

## Escalado min-max

scaler_con_outliers = MinMaxScaler()
scaler_con_outliers.fit(X_train_con_outliers)

X_train_con_outliers_scal = scaler_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_scal = pd.DataFrame(X_train_con_outliers_scal, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_scal = scaler_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_scal = pd.DataFrame(X_test_con_outliers_scal, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_scal.to_excel("../data/processed/X_train_con_outliers_scal.xlsx", index = False)
X_test_con_outliers_scal.to_excel("../data/processed/X_test_con_outliers_scal.xlsx", index = False)

scaler_sin_outliers = StandardScaler()
scaler_sin_outliers.fit(X_train_sin_outliers)

X_train_sin_outliers_scal = scaler_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_scal = pd.DataFrame(X_train_sin_outliers_scal, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_scal = scaler_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_scal = pd.DataFrame(X_test_sin_outliers_scal, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_scal.to_excel("../data/processed/X_train_sin_outliers_scal.xlsx", index = False)
X_test_sin_outliers_scal.to_excel("../data/processed/X_test_sin_outliers_scal.xlsx", index = False)

X_train_con_outliers_scal.head()

# Selección de características

selection_model = SelectKBest(f_classif, k = 4)
selection_model.fit(X_train_con_outliers_scal, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_con_outliers_scal), columns = X_train_con_outliers_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_con_outliers_scal), columns = X_test_con_outliers_scal.columns.values[ix])

import json

with open("feature_selection_k_5.json", "w") as f:
    json.dump(X_train_sel.columns.tolist(), f)

X_train_sel.head()
X_test_sel.head()

X_train_sel["Survived"] = list(y_train)
X_test_sel["Survived"] = list(y_test)

X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)