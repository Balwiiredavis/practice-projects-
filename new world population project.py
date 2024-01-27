#!/usr/bin/env python
# coding: utf-8

# In[2]:


import ipywidgets as widgets 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import numpy as np 


# In[30]:


import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go


# In[4]:


df=pd.read_csv("world_population.csv")


# In[5]:


df


# In[7]:


df.info()


# In[5]:


# Checking for null values
df.isnull().sum()


# In[6]:


df.shape


# In[9]:


# Investigate all the elements whithin each Feature 

for column in df:
    unique_vals = np.unique(df[column])
    nr_values = len(unique_vals)
    if nr_values < 36:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# In[10]:


df.describe()


# In[8]:


df.columns


# In[24]:


# continents in 2022

continents = df.nlargest(5,'2022 Population')

plt.figure(figsize=(10, 6)) 
plt.pie(df_rank['2022 Population'], labels=df_rank['Continent'], autopct='%1.1f%%', startangle=140, explode=explode)
plt.axis('equal')  
plt.title('Most populous Continents 2022 (Percentage)')
plt.show()


# In[16]:


# most populous countries in 2022

most_populous_countries = df.nlargest(20, '2022 Population')

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='2022 Population', y='Country/Territory', data=most_populous_countries)
plt.xlabel('2022 Population')
plt.ylabel('Country/Territory')
plt.title('Most populous Countries 2022')
plt.show()



# In[25]:


# countries by growth rate

top_20_countries_growth = df.nlargest(20,'Growth Rate')
total_world_growth_rate = df['Growth Rate'].sum()
print (f"World Growth Rate: {total_world_growth_rate} " )
# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Growth Rate', y='Country/Territory', data=top_20_countries_growth)
plt.xlabel('Growth Rate')
plt.ylabel('Country/Territory')
plt.title('Top 20 Countries by Growth Rate')
plt.show()


# In[29]:


# Calculate the total world population for each year
world_population = df[['2022 Population', '2020 Population', '2015 Population',
       '2010 Population', '2000 Population', '1990 Population',
       '1980 Population', '1970 Population']].sum()

# Convert population to billions
world_population_billion = world_population / 1e9

# Create a bar plot to visualize the world population for each year
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=world_population_billion.index, y=world_population_billion.values)
plt.xlabel('Year')
plt.ylabel('World Population')
plt.title('World Population by Year')

# Create a custom formatter to display y-axis labels in billions
def billions_formatter(x, pos):
    return f'{x:.1f}B'

ax.yaxis.set_major_formatter(ticker.FuncFormatter(billions_formatter))

plt.xticks(rotation=45)
plt.show()


# In[31]:


df.columns


# In[32]:


# CHROPLETH MAP OF WORLD POPULATION (2022,2015,2000 AND 1980)

fig = px.choropleth(df, locations = 'Country/Territory', locationmode = 'country names',
                    color = '2022 Population', hover_name = 'Country/Territory', 
                    title='World 2022 Population')
fig.show()


# In[33]:


fig = px.choropleth(df, locations = 'Country/Territory', locationmode = 'country names',
                    color = '2015 Population', hover_name = 'Country/Territory', 
                    title='World 2015 Population')
fig.show()


# In[34]:


fig = px.choropleth(df, locations = 'Country/Territory', locationmode = 'country names',
                    color = '2000 Population', hover_name = 'Country/Territory', 
                    title='World 2000 Population')
fig.show()


# In[35]:


fig = px.choropleth(df, locations = 'Country/Territory', locationmode = 'country names',
                    color = '1980 Population', hover_name = 'Country/Territory', 
                    title='World 1980 Population')
fig.show()


# In[38]:


# by area

l_mass = df.groupby(['Country/Territory']).mean(numeric_only=True).sort_values(by='Area (km²)', ascending=False)
l_mass.head()


# In[39]:


# by density

density = df.groupby(['Country/Territory']).mean(numeric_only=True).sort_values(by='Density (per km²)', ascending=False)
density.head()


# In[50]:


df2 = df.groupby("Continent")[['2022 Population', '2020 Population', '2015 Population',
       '2010 Population', '2000 Population', '1990 Population',
       '1980 Population', '1970 Population']].mean(numeric_only=True).sort_values(by='2022 Population',ascending=False)
df2.plot()


# In[52]:


df3 = df2.transpose()
df3.plot()


# In[53]:


df3.plot.barh(stacked=True)


# In[ ]:




