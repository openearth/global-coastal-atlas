# %%
import itertools
import os
import pandas as pd
import pystac_client

# File paths
file_path_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
file_path_catalog = r'https://storage.googleapis.com/dgds-data-public/gca/gca-stac/catalog.json'
file_path_readme = os.path.join(file_path_repo, 'README.md')

# Function to convert dimensions to dict
def convert_dim_to_dict(k, v):
    d = {'name': k,
         'long_name': v['description'] if 'description' in v else '',
         'units': v['units'] if 'units' in v else '',
         'stucture_type': 'dim',
         'dtype': v['dtype'] if 'dtype' in v else '',
         'ncollections': 1,
         'collections': [collection_dict['id']]}
    return d

# Function to convert variables to dict
def convert_var_to_dict(k, v):
    d = {'name': k,
         'long_name': v['attrs']['long_name'] if 'long_name' in v['attrs'] else '',
         'units': v['attrs']['units'] if 'units' in v['attrs'] else '',
         'stucture_type': 'var',
         'dtype': v['dtype'] if 'dtype' in v else '',
         'ncollections': 1,
         'collections': [collection_dict['id']]}
    return d

# Get catalog and collections
catalog = pystac_client.Client.open(file_path_catalog)
collections = list(catalog.get_all_collections())

# Get vocabulary for dimensions and variables in all collections
vocab_ls = []
for collection in collections:
    collection_dict = collection.to_dict()
    if 'cube:dimensions' in collection_dict:
        for k, v in collection_dict['cube:dimensions'].items():
            vocab_ls.append(convert_dim_to_dict(k, v))

    if 'cube:variables' in collection_dict:
        for k, v in collection_dict['cube:variables'].items():
            if 'attrs' in v:
                vocab_ls.append(convert_var_to_dict(k, v))	
    
# Convert vocabulary list to dataframe
vocab_df = pd.DataFrame(vocab_ls)

# Group by name, long_name, unit and dtype
vocab_df = vocab_df.groupby(['name', 'long_name', 'units', 'dtype', 'stucture_type']).agg(
    ncollections=('ncollections', 'sum'),
    collections = ('collections', lambda x: list(set(itertools.chain.from_iterable(x))))).reset_index()

# Sort by n and alphabetically by name
vocab_df = vocab_df.sort_values(['stucture_type','ncollections', 'name'],
                                      ascending=[True, False, True]).reset_index(drop=True)

# Function to convert dataframe to markdown
def convert_df_to_md(df, comment):
    df = df.copy()

    # Shorten collections to n and convert to string
    n = 2
    df['collections'] = df['collections'].apply(lambda x: ', '.join(x[:n]+['...']) if len(x) > n else ', '.join(x))

    # Convert dataframe to markdown
    md = df.to_markdown(index=False)

    # Add comments around markdown table
    md = f'[comment]: <{comment}>\n\n {md} \n\n[comment]: <{comment}>'
    
    return md

# Function to replace table in readme
def replace_table_in_readme(readme, table):
      readme_split = readme.split(table.split('\n')[0])
      return readme_split[0] + table + readme_split[2]

vocab_mk = convert_df_to_md(vocab_df, comment='vocab table')

# Replace table in readme
with open(file_path_readme, 'r') as file:
    readme = file.read()

readme = replace_table_in_readme(readme, vocab_mk)

with open(file_path_readme, 'w') as file:
    file.write(readme)

# %%
