import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

data = pd.read_csv('aegypti_albopictus.csv')

print(data.head())

for c in data.columns:
    counts = data[c].value_counts()
    print("\nFrequency of each value of %s:\n" % c)
    print(counts.head())
    print("\nUnique values of %s: %i\n" % (c, len(counts)))

data = data.drop(['OCCURRENCE_ID', 'COUNTRY', 'COUNTRY_ID'], axis=1)

categoricals = ['VECTOR', 'SOURCE_TYPE', 'LOCATION_TYPE', 
                'POLYGON_ADMIN', 'GAUL_AD0', 'YEAR', 'STATUS']

for c in categoricals:
    data[c] = pd.factorize(np.array(data[c]))[0]

for c in data.columns:
    counts = data[c].value_counts()
    print("\nFrequency of each value of %s:\n" % c)
    print(counts.head())
    print("\nUnique values of %s: %i\n" % (c, len(counts)))

sns.heatmap(data.corr())

grid = sns.FacetGrid(data, hue='VECTOR')
grid.map(plt.scatter, 'LOCATION_TYPE', 'POLYGON_ADMIN')

grid = sns.FacetGrid(data, hue='VECTOR', row='LOCATION_TYPE')
grid.map(plt.scatter, 'X', 'Y')

grid = sns.FacetGrid(data, hue='VECTOR', row='POLYGON_ADMIN')
grid.map(plt.scatter, 'X', 'Y')

plt.show()

# POLYGON_ADMIN = -999 indicates LOCATION_TYPE = point

# GAUL_AD0 is the country-level administrative unit layer code,
# which seems to have a one-to-one correspondence with COUNTRY

# STATUS can be E (established) or T (transient) population

# Potential predictors:
# x_neg100_neg70_y_20_40
# x_neg110_neg50_y_neg40_30
# polynan_xneg5to30_y30to60
# poly1_xneg100toneg70_y20to50
# poly3_xneg80toneg50_yneg40to10
# poly1_xneg100toneg50_yneg40to30
