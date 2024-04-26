# Import necessary libraries
import os
import pandas as pd
import geopandas as gpd
from libpysal import weights
import esda
from functools import reduce
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import combine_pvalues
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# Step 1: Clustering Analysis:
#For each Excel file, it iterates over columns (representing different variables).
#It creates a GeoDataFrame by merging the loaded geographical data with the data from the Excel file based on a common identifier ('LAU_ID').
#It calculates the Within Cluster Sum of Squared Errors (WCSS) for different values of K (number of nearest neighbors) using K-means clustering.
#It finds the elbow point in the WCSS analysis, which indicates an optimal number of clusters.
#It plots the WCSS analysis for each variable and saves the plots as images (..OUTPUTS\\Plots_K_suppliment\\{column}_plot.png').
#It saves the K values to Excel files (....\\OUTPUTS\\Plots_K_suppliment\\k_values.xlsx',).

#1.1 .Create a geospatial dataframe joining tabular data (xlsx) and geospatial layer (.shp),
#the spatial data, in our case 'LAU_point_2013.shp'

#The tabular data, Exposure of population and built up (absolute and relative-%) to various hazards
# with various Return Periods (please see table 1 in the Antofie et al. 2024) .

# Set Paths to Excel files containing tabular data
excel_files = [
    r'D:\Data\MHRA_publication\Topublish\Data_MH\Population.xlsx',
    r'D:\Data\MHRA_publication\Topublish\Data_MH\Residential%.xlsx',
    r'D:\Data\MHRA_publication\Topublish\DATA_MH\Population%.xlsx',
    r'D:\Data\MHRA_publication\Topublish\DATA_MH\Residential.xlsx',
]

# Load shapefile containing geographical data
gdf_shapefile = gpd.read_file(r'D:\Data\MHRA_publication\Topublish\Data_MH\LAU_point_2013.shp')

# Create a DataFrame to store elbow K values for different variables and files
k_values_df = pd.DataFrame()

# Initialize an empty dictionary to store DataFrames grouped by variable names
grouped_dfs = {}

# Iterate over Excel files
for excel_file in excel_files:
    # Load Excel file
    df_excel = pd.read_excel(excel_file, header=0)

    # Iterate over columns (starting from column F and ending at column K)
    for column in df_excel.columns[5:11]:
        # Create GeoDataFrame and merge with shapefile based on 'LAU_ID' column
        db = gpd.GeoDataFrame(gdf_shapefile.merge(df_excel, on='LAU_ID'), crs=gdf_shapefile.crs) \
            .to_crs(epsg=3035) \
            [['LAU_ID', column, 'geometry']] \
            .dropna()

        # Calculate Within Cluster Sum of Squared Errors (WCSS) for different K values
        WCSS = []
        for order in range(1, 51, 5):
            knn = weights.distance.KNN.from_dataframe(db, k=1)
            knn.reweight(k=order, inplace=True)
            knn.transform = 'r'
            lag_residual = weights.spatial_lag.lag_spatial(knn, db[column])
            db = db.reset_index(drop=True)
            WCSS.append(sum(db[column] - lag_residual) ** 2)


        # Function to find elbow point in WCSS analysis
        def find_elbow_point(x, y):
            # Calculate distances from each point to a line connecting the first and last points
            distances = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0]) / \
                        np.sqrt((y[-1] - y[0]) ** 2 + (x[-1] - x[0]) ** 2)
            # Rotate distances and find the index of the maximum value, which corresponds to the elbow point
            rotated_distances = distances * np.sqrt(2) / 2
            elbow_index = np.argmax(rotated_distances)
            elbow_k = x[elbow_index]
            return elbow_k


        # Find elbow point in the WCSS analysis
        x_values = np.arange(1, 51, 5)
        y_values = np.array(WCSS)
        elbow_k = find_elbow_point(x_values, y_values)

        # Add K value to the DataFrame
        k_values_df.at[column, excel_file] = elbow_k

        # Plot results and save the plot as an image
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values, marker='o', markerfacecolor='red', markersize=5, label='WCSS')
        ax.plot([x_values[0], x_values[-1]], [y_values[0], y_values[-1]], linestyle='--', label='Diagonal Line')
        ax.scatter(elbow_k, y_values[x_values.tolist().index(elbow_k)], color='blue', label='Elbow Point', s=100,
                   zorder=5)
        plt.text(s=f'Elbow for WCSS clustering at $K$={elbow_k}', x=elbow_k,
                 y=y_values[x_values.tolist().index(elbow_k)], color='blue')
        plt.ylabel("Within Cluster Sum of Squared Errors (WCSS) \n for different values of $K$")
        plt.xlabel('$K$: number of nearest neighbors')
        plot_filename = f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Plots_K_suppliment\\{column}_plot.png'
        plt.title(f'WCSS Analysis for {column} in {plot_filename}')
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()

        # Save K values to Excel file
        k_values_df.to_excel(r'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Plots_K_suppliment\\k_values.xlsx',
                             index_label='Variable')

# Step 2: Hotspot Analysis:
#Step 2.1.:It builds a spatial weights matrix using a Kernel function based on the optimal_k value found in the previous step.
#Step 2.2: It performs hotspot analysis using Getis and Ord’s G∗i statistic (G_Local).
# It saves the g_Zs and the p_sim to Excel files by each exposure type
#and by aggregation (absolute and %) for LAU ('OUTPUTS\\Excel_Gordis_single\\{column}_result.xlsx',).
# It groups the dataframes by exposure type and by
# aggregation (absolute and %) for LAU ('\OUTPUTS\\Appended\\{key}_grouped.xlsx'')

# Step 2.1. Generate W from the GeoDataFrame using distance

#Build a spatial weights matrix (W) that encodes the spatial relationships on our data.
        # Kernel function which uses the proximity (k) information computed above
        W = weights.distance.KNN.from_dataframe(db, k=int(elbow_k))
        W.transform = 'R'
#Step 2.2. Perform hotspot analysis using Getis and Ord’s G∗i
#It produces:
#- standard deviations (  Z_score  named: g_Zs)
#- probability (p-value named  p_sim  ).
        gostars = esda.getisord.G_Local(db[column], W, star=True)

        # Save results in Excel file
        result_df = pd.DataFrame({
            'LAU_ID': db['LAU_ID'],
            f'{column}_g_Zs': gostars.Zs,
            f'{column}_p_sim': gostars.p_sim
        })
        result_excel_file = f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Excel_Gordis_single\\{column}_result.xlsx'
        print(f"Saving to: {result_excel_file}")
        result_df.to_excel(result_excel_file, index_label='Index')

        # Group DataFrame by the first 4 left characters of the '{column}_g_Zs' name
        grouped_key = f'{column}_g_Zs'[:4]
        if grouped_key not in grouped_dfs:
            grouped_dfs[grouped_key] = [result_df]
        else:
            grouped_dfs[grouped_key].append(result_df)

# Group DataFrames within each type of exposure
combined_dfs = {}
for key, dfs in grouped_dfs.items():
    # Merge each dataframe with gdf_shapefile based on 'LAU_ID'
    merged_dfs = [pd.merge(gdf_shapefile[['LAU_ID']], df, on='LAU_ID', how='left') for df in dfs]

    # Group dataframes within the group
    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['LAU_ID'], how='outer'), merged_dfs)

    # Add 'CNTR_ID' based on the first 2 characters of 'COMM_ID'
    combined_df['CNTR_ID'] = combined_df['LAU_ID'].astype(str).str[:2]

#    print(combined_df)
    combined_dfs[key] = combined_df
    # Save grouped DataFrame to Excel file
    grouped_excel_file = f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Appended\\{key}_grouped.xlsx'
    print(f"Saving to: {grouped_excel_file}")
    combined_df.to_excel(grouped_excel_file, index_label='Index')


# Step 3: Combine single hazards' exposure  to multiple-hazards' exposure using Stouffer method

#Step 3.1:This operation calculates the 'Combined_p_val' and 'Combined_z_score'/,
#which contains the Stouffer combined p-values and z_scores calculated for each row(LAU) across all hazards'exposures
#Step 3.2: Assign a 'Bin' value based on conditions involving 'Combined_z_score' and 'Combined_p_val'.
#It adds a column 'Nr_HZ' based on the count of non-zero and non-NaN values in the '_g_Zs' columns.
#Step 3.3.: Saves the combined Stouffer z_scores and p_values to excel file by exposure type(..\OUTPUTS\\Appended\\{key}_combined.xlsx')

#Step 3.1: Calculate the combined Stouffer z-score using the formula: sum_Z / sqrt_count.

for key, combined_df in combined_dfs.items():
# Combine '{column}_g_Zs' values across columns
    # Additional processing and calculations for combined DataFrame
    g_Zs_columns = [col for col in combined_df.columns if '_g_Zs' in col]
    sum_Z = combined_df[g_Zs_columns].sum(axis=1)

    # Count non-zero and non-NaN values
    count_nonzero = combined_df[g_Zs_columns].apply(lambda x: x[x != 0].count(), axis=1)

    # Calculate sqrt(count)
    sqrt_count = np.sqrt(count_nonzero)

    # Calculate combined Stouffer z_score
    combined_df['Combined_z_score'] = sum_Z / sqrt_count

# It calculates the combined p_values  using Stouffer's function 'combine_pvalues'
    #Calculate the Stouffer p_value
    p_Sim_columns = [col for col in combined_df.columns if '_p_sim' in col]
    f = lambda x: combine_pvalues(x.dropna().values, method='stouffer', weights=None)[1]
    combined_df['Combined_p_val'] = combined_df[p_Sim_columns].apply(f, axis=1)

#Step 3.2 It assigns a 'Bin' value based on conditions involving 'Combined_z_score' and 'Combined_p_val'.

    # Add 'Bin' column based on conditions
    combined_df['Bin'] = 0  # Initialize 'Bin' column
    combined_df.loc[(combined_df['Combined_z_score'] >= 0) & (combined_df['Combined_p_val'] <= 0.01), 'Bin'] = 1
    combined_df.loc[(combined_df['Combined_z_score'] >= 0) & (combined_df['Combined_p_val'] > 0.01) &
                    (combined_df['Combined_p_val'] <= 0.05), 'Bin'] = 2
    combined_df.loc[(combined_df['Combined_z_score'] >= 0) & (combined_df['Combined_p_val'] > 0.05) &
                    (combined_df['Combined_p_val'] <= 0.1), 'Bin'] = 3
    combined_df.loc[(combined_df['Combined_p_val'] > 0.1), 'Bin'] = 4
    combined_df.loc[(combined_df['Combined_z_score'] < 0) & (combined_df['Combined_p_val'] > 0.05) &
                    (combined_df['Combined_p_val'] <= 0.1), 'Bin'] = 5
    combined_df.loc[(combined_df['Combined_z_score'] < 0) & (combined_df['Combined_p_val'] > 0.01) &
                    (combined_df['Combined_p_val'] <= 0.05), 'Bin'] = 6
    combined_df.loc[(combined_df['Combined_z_score'] < 0) & (combined_df['Combined_p_val'] <= 0.01), 'Bin'] = 7

    # Add 'Nr_HZ' column based on 'Count non-zero and non-NaN values'
    g_Zs_columns = [col for col in combined_df.columns if '_g_Zs' in col]
    combined_df['Nr_HZ'] = combined_df[g_Zs_columns].apply(lambda x: x[x != 0].count(), axis=1)


#Step 3.3. Save the combined Stouffer z_scores and p_values to excel file by exposure type
    # Select the columns to be save in Excel
    selected_columns_df = ['LAU_ID', 'CNTR_ID', 'Combined_z_score',
                       'Combined_p_val', 'Bin', 'Nr_HZ']

    combined_df = combined_df[selected_columns_df]

    # Additional operations such as stripping column names and merging with
# another DataFrame (gdf_shapefile) based on a common column ('LAU_ID').
    combined_df.columns = combined_df.columns.str.strip()
    merged_df = pd.merge(combined_df, gdf_shapefile[['LAU_ID', 'NUTS3_ID']], on='LAU_ID', how='left')
    combined_df = merged_df

    combined_excel_file = f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Appended\\{key}_combined.xlsx'
    print(f"Saving to: {combined_excel_file}")  # Add this line for debugging
    combined_df.to_excel(combined_excel_file, index_label='Index')


#Step 4: Adding additional colums to our dataframe (combined_df): the GDP per capita clssess and the Total population of the LAU
##Step 4.1.:The script reads GDP data from a CSV file named 'GDP_capita.csv'
#It performs a classification analysis on the GDP data using quantiles to devide the GDP values in 4 classes
#Define a Function to Map 'INC_Capita' Values to Corresponding Labels ('low income', 'low middle income', 'high middle income', 'high income')
#The GDP data is merged with combined_df based on 'NUTS3_ID' using a left merge, resulting in combined_df_gdp.
##Step 4.2.:Population data is read from a CSV file named 'POP_total.csv'
#Population data is merged with combined_df_gdp based on 'LAU_ID' using a left merge, resulting in final_combined_df.
#Step 4.3: Save the df with GDP and Total population to Excel for Each Group: The' final_combined_df' is
# saved to an Excel file named 'final_combined_dataframe_{key}.xlsx' for each group in combined_dfs.


#Step 4.1.: Add GDP classes. Perform a classification analysis on the GDP data using quantiles to devide the GDP values in 4 classes
    # Read GDP data
    gdp_data = pd.read_csv(r'D:\Data\MHRA_publication\MH_DATA_shared\GDP_capita.csv')

    # Perform classification analysis using quantiles
    gdp_data['INC_Capita'] = pd.qcut(gdp_data['GDP_capita'], q=4, labels=False) + 1


    # Define a function to map 'INC_Capita' values to corresponding labels
    def label_income(row):
        if row['INC_Capita'] == 1:
            return 'low income'
        elif row['INC_Capita'] == 2:
            return 'low middle income'
        elif row['INC_Capita'] == 3:
            return 'high middle income'
        elif row['INC_Capita'] == 4:
            return 'high income'
        else:
            return 'Unknown'  # Add a default label for unknown values, if any


    # Apply the function to create the new 'INC_label' column
    gdp_data['INC_label'] = gdp_data.apply(label_income, axis=1)

    gdp_data_excel_file = f'D:\Data\MHRA_publication\Topublish\Data_MH\GDP_capita.xlsx'
    gdp_data.to_excel(gdp_data_excel_file, index_label='Index')

#Step 4.2.:Population data is read from a CSV file named 'POP_total.csv'  and addedd to the df
    #LAU's total population
    Pop_data = pd.read_csv(r'D:\Data\MHRA_publication\Topublish\Data_MH\POP_total.csv')
    Pop_total_data= Pop_data[['LAU_ID','Total_pop']]

    # Iterate over each group in combined_dfs
    final_combined_dfs = {}

    # Merge dataframes on 'NUTS3_ID' using left merge
    gdp_data.reset_index(drop=True, inplace=True)

    combined_df_gdp=pd.merge(combined_df, gdp_data, on='NUTS3_ID', how='left')

    #Save as Excel
    combined_df_gdp_excel_file = f'D:\Data\MHRA_publication\Topublish\combined_df_gdp_{key}.xlsx'
    print(f"Saving to: {combined_df_gdp_excel_file}")  # Add this line for debugging
    combined_df_gdp.to_excel(combined_df_gdp_excel_file, index_label='Index')

#Step 4.3: Save the df with GDP and Total population to Excel for Each Group
    # Merge dataframes on 'NUTS3_ID' using left merge
    #Final combined df with GDP and Total population
    final_combined_df = pd.merge(Pop_total_data,combined_df_gdp, on='LAU_ID', how='left')

    selected_columns_combined_df = ['LAU_ID','NUTS3_ID', 'CNTR_ID', 'Combined_z_score',
                            'Combined_p_val', 'Bin', 'Nr_HZ',
                        'Total_pop', 'INC_Capita','INC_label']
    #'Sum_hz_type', 'Hz_type_label',
#    final_combined_df = final_combined_df.rename(columns={'NUTS3_ID_x': 'NUTS3_ID'})
    final_combined_df = final_combined_df[selected_columns_combined_df]

    # Save the final dataframe to Excel for each group
    final_excel_file = f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Final\\final_combined_dataframe_{key}.xlsx'
    print(f"Saving to: {final_excel_file}")  # Add this line for debugging
    final_combined_df.to_excel(final_excel_file, index_label='Index')

#Step 5. Is completing the final dataframe wihch will be used in all the following plots and analysis

#Step 5.1:Add URAU data (Urban Audit 2021,  the Urban Audit category: with C = City and F = Functional Urban Area) to the dataframe by spatial join of LAU spatial data with URAU spatila data
#The LAU point shapefile (point_shapefile) is reprojected to match the Coordinate Reference System (CRS) of the URAU urau_shapefile.
#The spatial join is based on intersection (op='intersects') and is a left join (how='left'), meaning that all features from the left layer (point_shapefile) are retained,
# and attributes from the right layer (urau_shapefile) are added where they intersect.
#Step 5.2: The final_urau_combined_df DataFrame is saved to an Excel file ('OUTPUTS/Final/final_uaru_combined_dataframe_{key}.xlsx'')

#Step 5.1 # Add URAU data (Urban Audit 2021,  the Urban Audit category: with C = City and F = Functional Urban Area) to the datafrmae

# Path to the local shapefile
    shapefile_path = r'D:/Data/MHRA_publication/MH_DATA_shared/ref-urau-2021-100k.shp/URAU_RG_100K_2021_3035.shp/URAU_RG_100K_2021_3035.shp'
#Read the shapefile as 'URAU'
    urau_shapefile = gpd.read_file(shapefile_path)

    # Read the point shapefile as 'point_shapefile'
    point_shapefile = gpd.read_file(r'D:\Data\MHRA_publication\Topublish\Data_MH\LAU_point_2013.shp')

    # Reproject point_shapefile to match the CRS of urau_shapefile
    point_shapefile = point_shapefile.to_crs(urau_shapefile.crs)

    # Perform spatial join and keep specified columns
    urau = gpd.sjoin(point_shapefile,urau_shapefile,
                     how='left', op='intersects')
    selected_urau = ['LAU_ID','NUTS3_ID','LAU_name','URAU_CODE', 'URAU_CATG', 'FUA_CODE']
    urau_excel = urau[selected_urau]

#    urau_excel.to_excel(r'D:/Data/MHRA_publication/Topublish/urau.xlsx', index=False)
#    print(f"Saving to: {urau_excel}")  # Line for debugging
    combined_df_urau = pd.merge(final_combined_df, urau_excel, on='LAU_ID', how='left')

# Step 5.2: The final_urau_combined_df DataFrame is saved to an Excel file

    # Check if 'URAU_CAT' column has blank rows
    combined_df_urau['URAU_Type'] = 'Urban'
    combined_df_urau.loc[combined_df_urau['URAU_CATG'].isnull(), 'URAU_Type'] = 'Rural'

    selected_columns_urau = ['LAU_ID', 'LAU_name', 'CNTR_ID', 'Combined_z_score',
                             'Combined_p_val', 'Bin', 'Nr_HZ',
                             'Total_pop', 'INC_Capita', 'INC_label', 'URAU_CODE', 'URAU_CATG', 'URAU_Type', 'FUA_CODE']

    combined_df_urau = combined_df_urau[selected_columns_urau]
# Extracting the last 4 digits from the key of the DataFrame
    extracted_key = key[-4:]

# Constructing the file name for saving the Excel file using the extracted key
    final_uaru_excel_file = f'D:/Data/MHRA_publication/Topublish/OUTPUTS/Final/final_uaru_combined_dataframe_{extracted_key}.xlsx'
    print(f"Saving to: {final_uaru_excel_file}")

# Saving the DataFrame to Excel with the constructed file name
    combined_df_urau.to_excel(final_uaru_excel_file, index_label='Index')
    print(f"Saved DataFrame for key {extracted_key} to Excel.")

###################     FIGURE 3

import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set up color settings
bins = [1, 2, 3, 4, 5, 6, 7]
label_names = ['hotspot 99% confidence', 'hotspot 95% confidence', 'hotspot 90% confidence',
               'not significant', 'coldspot 90% confidence', 'coldspot 95% confidence', 'coldspot 99% confidence']
color_scheme = ['red', 'tomato', 'lightsalmon', 'yellow', 'lightblue', 'lightsteelblue', 'steelblue']
cmap = LinearSegmentedColormap.from_list(name='custom', colors=color_scheme)

# Load the country shapefile
CNTR = gpd.read_file(r'D:\Data\HOME_work\Admin_Units\ADMIN_all\EU_country\EU_countries.shp')
CNTR = CNTR.to_crs(epsg=3035)

# Iterate over each key and dataframe
for key in combined_dfs.keys():
    # Assuming combined_df is your DataFrame
    combined_df = combined_dfs[key]

    # Chunk the dataframe
    chunked_df = combined_df[(combined_df['Bin'] >= 4) & (combined_df['Nr_HZ'] > 1)]

    # Merge with the country shapefile
    gdf = gpd.read_file(r'D:\Data\MHRA_publication\Topublish\Data_MH\LAU_2013.shp')
    db = gpd.GeoDataFrame(gdf.merge(combined_df, on='LAU_ID'), crs=gdf.crs) \
        .to_crs(epsg=3035) \
        [['LAU_ID', 'Combined_z_score', 'Combined_p_val', 'Bin', 'CNTR_ID', 'geometry']].dropna()

    # Create subplot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot country boundaries
    CNTR.plot(ax=ax, facecolor='white', edgecolor='black')

    # Plot merged geodataframe
    db.plot('Bin', ax=ax, cmap=cmap, missing_kwds={"color": "white", "label": "No exposure"})
    legend_elements = [Patch(facecolor=color_scheme[0], label=label_names[0]),
                       Patch(facecolor=color_scheme[1], label=label_names[1]),
                       Patch(facecolor=color_scheme[2], label=label_names[2]),
                       Patch(facecolor=color_scheme[3], label=label_names[3]),
                       Patch(facecolor=color_scheme[4], label=label_names[4]),
                       Patch(facecolor=color_scheme[5], label=label_names[5]),
                       Patch(facecolor=color_scheme[6], label=label_names[6])]
    # ax.legend(handles=legend_elements, legend_kwds={'loc': 'upper right','fontsize': 20,'handlelength':1,
    #                                   'handletextpad':0.5, 'handleheight'  : 0.1,'labelspacing':0.1})
    ax.legend(handles=legend_elements, shadow=True,
              fancybox=True, facecolor='#fefdfd', prop={'size': 13}, loc='upper right')

    ax.set_ylim([1300000, 5500000])
    ax.set_xlim([2490000, 6500000])

    ax.set_axis_off()
    plt.tight_layout()


    # Annotate subplot with letters
    ax.annotate(f'{key}.)', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=12, fontweight='bold')


###########  Figure 3 zoom
    ax2 = ax.inset_axes([0.72, 0.35, 0.42, 0.42], facecolor='floralwhite')  # posx, posy, width, height
    CNTR.plot(figsize=(12, 12), facecolor='white', edgecolor='black', linewidth=2, ax=ax2)
    db.plot('Bin', legend=False, figsize=(12, 12), ax=ax2, linewidth=0.5,
            cmap=cmap, missing_kwds={"color": "white",
                                     "label": "No exposure Scripts/Myscipts/Multi_hazard/LOSSdata_hotspot/URAU_analysis.py:130"})
    ax2.set_title('Zoomed FRANCE', fontsize=18, fontweight='bold')

    ax2.set_xlim([3200000, 4200000])  ##on map coordinates France
    ax2.set_ylim([2150000, 3140000])  # on map coordinates France

    ax2.set(ylabel=None)
    ax.indicate_inset_zoom(ax2, edgecolor="black", linewidth=4)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
    # ax2.set_axis_off()

    ax2.tick_params(axis='x', colors="black", width=4)


    # Save the plot
    plt.savefig(f'D:\\Data\\MHRA_publication\\Topublish\\OUTPUTS\\Figure\\Figure_{key}.png')

    # Show the plot (optional)
    plt.show()

