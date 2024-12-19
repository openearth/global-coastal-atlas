import os
from openai import AzureOpenAI
import numpy as np
import xarray as xr
from typing import Union

def describe_data(xarr: xr.Dataset, dataset_id: str) -> str:
     # Create prompt
    prompt = make_prompt(xarr, dataset_id)

    # Load environment variables
    api_version = '2024-03-01-preview'
    api_base_url = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

    # Initialise large language model
    model = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f'{api_base_url}/deployments/{deployment_name}',
    )

    # Trigger model
    response = model.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        messages=[
            {'role': 'system', 'content': prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    # Return response
    return response.choices[0].message.content

def describe_overview(polygon, dataset_contents) -> str:
    para_list = [dataset_contents[ind].text for ind in range(len(dataset_contents))]
    coor_list = list(zip(*polygon.boundary.xy))

    prompt = """
    You are a coastal scientist tasked with writing for a report describing the state of the coast. This report includes key information about a location,
    such as population, sediment characteristic of that location, land subsidence risk, shoreline erosion or accretion, future sea level rise, future extreme sea level, and
    future shoreline change. You should write two paragraphs with the provided information. The first paragraph is a factual description about an area of location.
    The coordinates (longitude, latitude) of this area are provided at the end. You should only describe the geographical information of this location, such as longitude, latitude, in which countries, or which big cities is in the nearby.
    You should describe it in about 100 words. After the first paragraph, you should also write a second paragraph, which contains approximately 250 words, 
    summarising the provided texts. The text are also provided at the end
    Ensure the description is clear, professional, and aligned with the dataset.

    * Coordinates of this area: {}
    * texts: {}
    """.format(str(coor_list), str(para_list))

    # Load environment variables
    api_version = '2024-03-01-preview'
    api_base_url = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

    # Initialise large language model
    model = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f'{api_base_url}/deployments/{deployment_name}',
    )

    # Trigger model
    response = model.chat.completions.create(
        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        messages=[
            {'role': 'system', 'content': prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )

    # Return response
    return response.choices[0].message.content


def make_prompt(xarr: Union[xr.Dataset, dict], dataset_id: str) -> str:
    match dataset_id: 
        case 'sediment_class':
            var = 'sediment_label'
                        
            sand_port  = np.round(len(np.where(xarr[var].values == 0)[0]) / len(xarr[var].values) * 100, 1)
            mud_port   = np.round(len(np.where(xarr[var].values == 1)[0]) / len(xarr[var].values) * 100, 1)
            cliff_port = np.round(len(np.where(xarr[var].values == 2)[0]) / len(xarr[var].values) * 100, 1)
            veg_port   = np.round(len(np.where(xarr[var].values == 3)[0]) / len(xarr[var].values) * 100, 1)
            other_port = np.round(len(np.where(xarr[var].values == 4)[0]) / len(xarr[var].values) * 100, 1)


            prompt = """
            You are a coastal scientist tasked with writing a concise paragraph (maximum 100 words) for a report describing the state of the coast.
            The dataset contains the proportion of different materials existing along a coastline. There are 5 different numbers under the dataset here.
            The first one is the proportion of sand in percentage. The second one is the propotion of mud in percentage.
            The third one is the proportion of coastal cliff in percentage. The fourth one is the proportion of vegetated area in percentage.
            The last one is the propotion of other materials in percentage.
            You should summarise the dataset and list the proportions of each material. You should describe what material is the majority of the coast and how materials are distributed along the shoreline. 
            Ensure the description is clear, professional, and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by...".
            
            * Dataset: {} {} {} {} {}
            """.format(sand_port, mud_port, cliff_port, veg_port, other_port)


        case 'world_pop':
            var = 'pop_tot'

            # Create data dictionary
            data_dict = {}
            data_dict['lon'] = {'value': np.round(xarr['lon'].values, 2), 'long_name': xarr['lon'].attrs['long_name'], 'units': xarr['lon'].attrs['units']}
            data_dict['lat'] = {'value': np.round(xarr['lat'].values, 2), 'long_name': xarr['lat'].attrs['long_name'], 'units': xarr['lat'].attrs['units']}
            data_dict[var] = {'value': np.round(xarr[var].values, 0), 'long_name': xarr[var].attrs['long_name']}

            prompt = """
            You are a coastal scientist tasked with writing a concise paragraph (maximum 100 words) for a report describing the state of the coast.
            This paragraph is related to human population living along coastlines. Higher population along a coastline means that there will be higher potential loss,
            including property and human loss, if there is any coastal hazard. Smaller loss or impact if ther population is smaller.
            You should summarise the population and its distribution along the coast. You should also spot if there is any important location, where you can find a high population.
            The dataset is provided below, which contains the longitude, the lattitude and the population at each location.
            You should at least specify at which longitude and latitude where we can find a substantial amount of population.
            Please explicitly mention that this is the population along the coastline.
            Ensure the description is clear, professional, and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by...".
            
            * Dataset: {}
            """.format(str(data_dict))



        case 'shoreline_change':
            var = 'changerate'
            
            # Create data dictionary
            data_dict = {}
            data_dict['lon'] = {'value': np.round(xarr['lon'].values, 2), 'long_name': xarr['lon'].attrs['long_name'], 'units': xarr['lon'].attrs['units']}
            data_dict['lat'] = {'value': np.round(xarr['lat'].values, 2), 'long_name': xarr['lat'].attrs['long_name'], 'units': xarr['lat'].attrs['units']}
            data_dict[var] = {'value': np.round(xarr[var].values, 2), 'long_name': xarr[var].attrs['long_name'], 'units': xarr[var].attrs['units']}

            # Create changerate classes dictionary
            classes_dict = {}
            classes_dict['extreme_accretion'] = {'min': 5, 'max': np.inf, 'unit': xarr[var].attrs['units']}
            classes_dict['severe_accretion'] = {'min': 3, 'max': 5, 'unit': xarr[var].attrs['units']}
            classes_dict['intense_accretion'] = {'min': 1, 'max': 3, 'unit': xarr[var].attrs['units']}
            classes_dict['accretion'] = {'min': 0.5, 'max': 1, 'unit': xarr[var].attrs['units']}
            classes_dict['stable'] = {'min': -0.5, 'max': 0.5, 'unit': xarr[var].attrs['units']}
            classes_dict['erosion'] = {'min': -1, 'max': -0.5, 'unit': xarr[var].attrs['units']}
            classes_dict['intense_erosion'] = {'min': -3, 'max': -1, 'unit': xarr[var].attrs['units']}
            classes_dict['severe_erosion'] = {'min': -5, 'max': -3, 'unit': xarr[var].attrs['units']}
            classes_dict['extreme_erosion'] = {'min': -np.inf, 'max': -5, 'unit': xarr[var].attrs['units']}

            prompt = """
            You are a coastal scientist tasked with writing a concise paragraph (maximum 100 words) for a report describing the state of the coast.
            Use the dataset below, which contains coastal change rates (positive values indicate accretion, negative values indicate erosion).
            Categorize the observed change rates using the coastal erosion classes, also provided below. Ensure the description is clear, professional,
            and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by...".
            
            * Dataset: {}
            * coastal erosion classes: {}
            """.format(str(data_dict), str(classes_dict))


        case 'land_sub':
            None


        # case 'slr_RCP26' | 'slr_RCP45' | 'slr_RCP85':
        #     # Create data dictionary
        #     data_dict = {}
        #     data_dict['years'] = {'value': ["2031","2041","2051", "2061", "2071", "2081", "2091", "2101","2111","2121","2131","2141","2151"], 'long_name': 'year'}
        #     data_dict['high'] = {'value': [slp['value'] for slp in xarr if (slp['msl'] == 'msl_h')], 'long_name': 'Sea level rise projection with high confidence', 'units': 'mm'}
        #     data_dict['medium'] = {'value': [slp['value'] for slp in xarr if (slp['msl'] == 'msl_m')], 'long_name': 'Sea level rise projection with medium confidence', 'units': 'mm'}
        #     data_dict['low'] = {'value': [slp['value'] for slp in xarr if (slp['msl'] == 'msl_l')], 'long_name': 'Sea level rise projection with low confidence', 'units': 'mm'}

        #     prompt1 = """
        #     You are a coastal scientist tasked with writing a concise paragraph (maximum 100 words) for a report describing the state of the coast.
        #     Use the dataset below, which contains the future sea level rise every ten years from 2031 to 2151. The dataset contains the sea level rise projection with
        #     high, medium and low confidence at every ten years from 2031. The high confidence and the low confidence projection states the maximum and minimum possible sea level rise,
        #     while the medium confidence projection states the mean possible sea level rise.
        #     """
        #     match dataset_id:
        #         case 'slr_RCP26':
        #             prompt2 = "The dataset is projected based on SSP126 scenario from AR6."
        #         case 'slr_RCP45':
        #             prompt2 = "The dataset is projected based on SSP245 scenario from AR6."
        #         case 'slr_RCP85':
        #             prompt2 = "The dataset is projected based on SSP585 scenario from AR6."

        #     prompt3 = """
        #     Please describe the trend of the sea level rise with different confidence of projections and Ensure the description is clear, professional,
        #     and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by...".
            
        #     * Dataset: {}
        #     """.format(str(data_dict))

        #     prompt = prompt1 + prompt2 + prompt3

        case 'slr':
            # Create data dictionary
            data_dict = {}
            data_dict['years'] = {'value': ["2031","2041","2051", "2061", "2071", "2081", "2091", "2101","2111","2121","2131","2141","2151"], 'long_name': 'year'}
            data_dict['high_end'] = {'value': [slp['value'] for slp in xarr if (slp['ssp'] == 'high_end')], 'long_name': 'Sea level rise projection in high-end scenario', 'units': 'mm'}
            data_dict['ssp126'] = {'value': [slp['value'] for slp in xarr if (slp['ssp'] == 'ssp126')], 'long_name': 'Sea level rise projection in SSP126 scenario', 'units': 'mm'}
            data_dict['ssp245'] = {'value': [slp['value'] for slp in xarr if (slp['ssp'] == 'ssp245')], 'long_name': 'Sea level rise projection in SSP245 scenario', 'units': 'mm'}
            data_dict['ssp585'] = {'value': [slp['value'] for slp in xarr if (slp['ssp'] == 'ssp585')], 'long_name': 'Sea level rise projection in SSP585 scenario', 'units': 'mm'}

            prompt = """
            You are a coastal scientist tasked with writing a concise paragraph (maximum 100 words) for a report describing the state of the coast.
            Use the dataset below, which contains the future sea level rise every ten years from 2031 to 2151. 
            The dataset contains the sea level rise projectionin differente in four different scenarios (i.e. high-end, SSP126, SSP245 and SSP585) every ten years from 2031. 
            Please describe the trend of the sea level rise in all four scenarios and Ensure the description is clear, professional,
            and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by....
            
            * Dataset: {}
            """.format(str(data_dict))


        case 'esl_RCP26' | 'esl_RCP45' | 'esl_RCP85':
            None


        case 'future_shoreline_change_2050' | 'future_shoreline_change_2100':
            # recalculate average shoreline charnge rate
            if dataset_id == 'future_shoreline_change_2050':
                rate = xarr.diff('time', 1).sel(time='2050') / (2050 - 2021)
            else:
                rate = xarr.diff('time', 1).sel(time='2100') / (2100 - 2050)

            rate45 = rate['sp_rcp45_p50']
            rate85 = rate['sp_rcp85_p50']

            # Create data dictionary
            data_dict = {}
            data_dict['lon'] = {'value': np.round(xarr['lon'].values, 2), 'long_name': xarr['lon'].attrs['long_name'], 'units': xarr['lon'].attrs['units']}
            data_dict['lat'] = {'value': np.round(xarr['lat'].values, 2), 'long_name': xarr['lat'].attrs['long_name'], 'units': xarr['lat'].attrs['units']}
            data_dict['RCP45'] = {'value': np.round(rate45.values, 2), 'long_name': 'average shoreline change rate in RCP4.5 scenario', 'units': 'm/yr'}
            data_dict['RCP85'] = {'value': np.round(rate85.values, 2), 'long_name': 'average shoreline change rate in RCP8.5 scenario', 'units': 'm/yr'}

            # Create changerate classes dictionary
            classes_dict = {}
            classes_dict['extreme_accretion'] = {'min': 5, 'max': np.inf, 'unit': 'm/yr'}
            classes_dict['severe_accretion'] = {'min': 3, 'max': 5, 'unit': 'm/yr'}
            classes_dict['intense_accretion'] = {'min': 1, 'max': 3, 'unit': 'm/yr'}
            classes_dict['accretion'] = {'min': 0.5, 'max': 1, 'unit': 'm/yr'}
            classes_dict['stable'] = {'min': -0.5, 'max': 0.5, 'unit': 'm/yr'}
            classes_dict['erosion'] = {'min': -1, 'max': -0.5, 'unit': 'm/yr'}
            classes_dict['intense_erosion'] = {'min': -3, 'max': -1, 'unit': 'm/yr'}
            classes_dict['severe_erosion'] = {'min': -5, 'max': -3, 'unit': 'm/yr'}
            classes_dict['extreme_erosion'] = {'min': -np.inf, 'max': -5, 'unit': 'm/yr'}

            prompt1 = """
            You are a coastal scientist tasked with writing a concise paragraph (maximum 200 words) for a report describing the state of the coast.
            Use the dataset below, which contains the future average coastal change rates (positive values indicate accretion, negative values indicate erosion)
            under two scenarios, RCP4.5 (indicated as the 'RCP45' dictionary) and RCP8.5 (indicated as the 'RCP85' dictionary).
            """

            if dataset_id == 'future_shoreline_change_2050':
                prompt2 = "The dataset is projected to 2050"
            else:
                prompt2 = "The dataset is projected to 2100"

            prompt3 = """
            Categorize the observed change rates using the coastal erosion classes, also provided below. Please also describe how different the coastal change rates
            is between two periods and describe some key changes on the rates between two periods. Please also be cautious of the change rates with non-realistic values,
            which is bigger than 100 m/yr. You should ignore the change rates with this non-realistic values.
            Ensure the description is clear, professional,
            and aligned with the dataset's trends. Begin your paragraph with: "The coast in this area is characterized by...".
            
            * Dataset: {}
            * coastal erosion classes: {}
            """.format(str(data_dict), str(classes_dict))

            prompt = prompt1 + prompt2 + prompt3
    
    return prompt