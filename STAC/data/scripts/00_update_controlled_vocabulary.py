# %%
import itertools
import os
import pandas as pd
import pystac_client
import xarray as xr
from collections.abc import Iterable

# File paths
file_path_repo = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
file_path_catalog = r'https://storage.googleapis.com/dgds-data-public/gca/gca-stac/catalog.json'
file_path_readme = os.path.join(file_path_repo, 'README.md')
file_path_vocab_xlsx = os.path.join(file_path_repo, 'STAC', 'data', 'vocab.xlsx')

# Function to open dataset
def open_dataset(href):
    extension = os.path.splitext(href)[1]
    if extension == '.nc':
        ds = xr.open_dataset(href)
    elif extension == '.zarr':
        ds = xr.open_zarr(href)
    else:
        raise ValueError('File extension not supported')
    return ds

# Get catalog and collections
catalog = pystac_client.Client.open(file_path_catalog)
collections = list(catalog.get_all_collections())

# Get vocabulary for dimensions and variables in all collections
vocab_ls = []
for collection in collections:
    href = collection.assets['data'].href
    ds = open_dataset(href)

    for var in ds.variables:
        # Get group
        if var in list(ds.keys()):
            group = 'variable'
        elif var in ds.dims:
            group = 'dimension'
        elif var in ds.coords:
            group = 'coordinate'
        
        # Create vocabulary dictionary
        vocab_dict = {
            'group': group,
            'name': var,
            'long_name': ds[var].long_name if 'long_name' in ds[var].attrs else '',
            'units': ds[var].units if 'units' in ds[var].attrs else '',
            'type': str(ds[var].dtype),
            'collections': [collection.id],
            'ncollections': 1}
        
        if vocab_dict['type'].startswith('|S'):
            vocab_dict['type'] = 'string'
            
        
        # Add first value to vocabulary dictionary
        values = ds[var].isel({dim: 0 for dim in ds[var].dims}).values.tolist()
        vocab_dict['values'] = [values]

        # Add collection to vocabulary dictionary
        vocab_ls.append(vocab_dict)

# Convert vocabulary list to dataframe
vocab_df = pd.DataFrame(vocab_ls)

# Group by group, name, long_name, units, and type
vocab_df = vocab_df.groupby(['group', 'name', 'long_name', 'units', 'type']).agg(
    ncollections=('ncollections', 'sum'),
    collections = ('collections', lambda x: list(set(itertools.chain.from_iterable(x)))),
    values = ('values', lambda x: list(set(itertools.chain.from_iterable(x)))),
    ).reset_index()

# Add temporary column to sort by group (dimension, coordinate, variable)
vocab_df['group_sort'] = vocab_df['group'].map({'dimension': 1, 'coordinate': 2, 'variable': 3})

# Sort by group_sort (dimension, coordinate, variable), name, and ncollections
vocab_df = vocab_df.sort_values(['group_sort', 'name', 'ncollections'], ascending=[True, True, False])

# Remove temporary column
vocab_df = vocab_df.drop(columns=['group_sort'])

# Add column with duplicate if name or long_name is duplicated
vocab_df['duplicate_name'] = vocab_df.duplicated(subset=['name'])
vocab_df['duplicate_name'] = vocab_df['duplicate_name'].map(lambda x: x>0)
vocab_df['duplicate_long_name'] = vocab_df.duplicated(subset=['long_name'])
vocab_df['duplicate_long_name'] = vocab_df['duplicate_long_name'].map(lambda x: x>0)
vocab_df['duplicate'] = vocab_df['duplicate_name'] | vocab_df['duplicate_long_name']
vocab_df['duplicate'] = vocab_df['duplicate'].map(lambda x: 'X' if x else '')

# Remove duplicates
#vocab_df = vocab_df.drop_duplicates(subset=['name', 'long_name'])

# Reorder columns
vocab_df = vocab_df[['group', 'name', 'long_name', 'units', 'type', 'ncollections', 'duplicate', 'collections', 'values']]

# Reset index
vocab_df = vocab_df.reset_index(drop=True)

# Save vocabulary to xlsx
vocab_df.to_excel(file_path_vocab_xlsx, index=False)

# Remove collections and values columns
vocab_mk_df = vocab_df.copy()
vocab_mk_df = vocab_mk_df.drop(columns=['collections', 'values'])

# Function to convert dataframe to markdown
def convert_df_to_md(df, comment):
    df = df.copy()

    # Convert dataframe to markdown
    md = df.to_markdown(index=False)

    # Add comments around markdown table
    md = f'[comment]: <{comment}>\n\n{md}\n\n[comment]: <{comment}>'
    
    # Return markdown
    return md

# Function to replace table in readme
def replace_table_in_readme(readme, table):
      readme_split = readme.split(table.split('\n')[0])
      return readme_split[0] + table + readme_split[2]

vocab_mk = convert_df_to_md(vocab_mk_df, comment='vocab table')

# Replace table in readme
with open(file_path_readme, 'r') as file:
    readme = file.read()

readme = replace_table_in_readme(readme, vocab_mk)

with open(file_path_readme, 'w') as file:
    file.write(readme)
