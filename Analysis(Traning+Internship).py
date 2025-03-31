# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os
import pytz
import datetime as dt

# %%
nltk.download('vader_lexicon')

# %%
# Step 1: Load the Dataset
apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')

# %%
apps_df


# %%
# Step 2: Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]
reviews_df.dropna(subset=['Translated_Review'], inplace=True)

# %%
# Merge datasets on 'App' and handle non-matching apps
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')
merged_df

# %%
# Step 3: Data Transformation
apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)

# %%
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    else:
        return np.nan

# %%
apps_df['Size'] = apps_df['Size'].apply(convert_size)

# %%
# Add log_installs and log_reviews columns
apps_df['Log_Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log_Reviews'] = np.log1p(apps_df['Reviews'])

# %%
# Add Rating Group column
def rating_group(rating):
    if rating >= 4:
        return 'Top rated'
    elif rating >= 3:
        return 'Above average'
    elif rating >= 2:
        return 'Average'
    else:
        return 'Below average'

apps_df['Rating_Group'] = apps_df['Rating'].apply(rating_group)

# %%
# Add Revenue column
apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']

# %%
# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
reviews_df['Sentiment_Score'] = reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# %%
# Extract year from 'Last Updated' and create 'Year' column
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Year'] = apps_df['Last Updated'].dt.year

# %%
import plotly.express as px

# %%
# Define the path for your HTML files
html_files_path = "./"

# %%
# Make sure the directory exists
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)

# %%
# Initialize plot_containers
plot_containers = ""

# %%
# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')

# %%
# Define your plots
plot_width = 400
plot_height = 300
plot_bg_color = 'black'
text_color = 'white'
title_font = {'size': 16}
axis_font = {'size': 12}

# %%
#Figure 1
# Category Analysis Plot
category_counts = apps_df['Category'].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x': 'Category', 'y': 'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width,
    height=plot_height
)
fig1.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig1.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig1, "category_analysis.html", "The top categories on the Play Store are dominated by tools, entertainment, and productivity apps. This suggests users are looking for apps that either provide utility or offer leisure activities.")

# %%
#Figure 2
# Type Analysis Plot
type_counts = apps_df['Type'].value_counts()
fig2 = px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=plot_width,
    height=plot_height
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig2, "type_analysis.html", "Most apps on the Play Store are free, indicating a strategy to attract users first and monetize through ads or in-app purchases.")

# %%
#Figure 3
# Rating Distribution Plot
fig3 = px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=plot_width,
    height=plot_height
)
fig3.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig3, "rating_distribution.html", "Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users.")

# %%
#Figure 4
#Sentiment Analysis Plot
sentiment_counts = reviews_df['Sentiment_Score'].value_counts()
fig4 = px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x': 'Sentiment Score', 'y': 'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width,
    height=plot_height
)
fig4.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig4.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig4, "sentiment_distribution.html", "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")

# %%
#Figure 5
# Installs by Category Plot
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.values,
    y=installs_by_category.index,
    orientation='h',
    labels={'x': 'Installs', 'y': 'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=plot_width,
    height=plot_height
)
fig5.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig5.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig5, "installs_by_category.html", "The categories with the most installs are social and communication apps, which reflects their broad appeal and daily usage.")

# %%
#Figure 6
# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "updates_per_year.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# %%
#Figure 7
# Revenue by Category Plot
revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width,
    height=plot_height
)
fig7.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig7.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig7, "revenue_by_category.html", "Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential.")

# %%
#Figure 8
# Genre Count Plot
genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=plot_width,
    height=plot_height
)
fig8.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
fig8.update_traces(marker=dict(line=dict(color=text_color, width=1)))
save_plot_as_html(fig8, "genres_counts.html", "Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games.")

# %%
#Figure 9
# Impact of Last Update on Rating
fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=plot_width,
    height=plot_height
)
fig9.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig9, "update_on_rating.html", "The scatter plot shows a weak correlation between the last update date and ratings, suggesting that more frequent updates don't always result in better ratings.")

# %%
#figure 10
# Ratings for Paid vs Free Apps
fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Ratings for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width,
    height=plot_height
)
fig10.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig10, "ratings_paid_free.html", "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")

# %%
# Split plot_containers to handle the last plot properly
plot_containers_split = plot_containers.split('</div>')
if len(plot_containers_split) > 1:
    final_plot = plot_containers_split[-2] + '</div>'
else:
    final_plot = plot_containers  # Use plot_containers as default if splitting isn't sufficient

# %%
# HTML template for the dashboard
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""

# %%
# Use these containers to fill in your dashboard HTML
final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)

# %%
# Save the final dashboard to an HTML file
dashboard_path = os.path.join(html_files_path, "dashboard.html")
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)

# %%
# Automatically open the generated HTML file in a web browser
webbrowser.open('file://' + os.path.realpath(dashboard_path))

# %% [markdown]
# **Nullclass Internship Tasks (11/03/2025-11/05/2025)**
# 

# %% [markdown]
# **Task 1.** Create a scatter plot to visualize the relationship between revenue and the number of installs for paid apps only. Add a trendline to show the correlation and color-code the points based on app categories.

# %%
# Step 1: Filtering the apps data for paid apps only
paid_apps = apps_df[apps_df['Type'] == "Paid"]
paid_apps

# %%
#Step 2: Ensuring revenue and installs are numeric for proper visualization
paid_apps['Revenue'] = pd.to_numeric(paid_apps['Revenue'], errors='coerce')
paid_apps['Installs'] = pd.to_numeric(paid_apps['Installs'], errors='coerce')

# %%
# Step 3: Adding a trendline (using numpy for computation)
x = paid_apps['Installs']
y = paid_apps['Revenue']
coeff = np.polyfit(x, y, 1)  
tl = coeff[0] * x + coeff[1]

# %%
#Figure 11
# Step 4: Creating an interactive scatter plot based on App Categories with Plotly
fig11 = px.scatter(
    paid_apps,
    x='Installs',
    y='Revenue',
    color='Category',
    title="Relationship Between Revenue and Installs for Paid Apps",
    labels={"Installs": "Number of Installs", "Revenue": "Revenue (in $)"},
    hover_data=['App'],
    opacity=0.8,  
    template='plotly_dark' 
)

# Step 5: Adding the trendline to the plot
fig11.add_scatter(
    x=paid_apps['Installs'], 
    y=tl, 
    mode='lines', 
    name='Trendline',
    line=dict(color='yellow', width=2)
)

# Step 6: Customing the layout
fig11.update_layout(
    title="Relationship Between Revenue and Installs for Paid Apps",
    xaxis=dict(title="Number of Installs", gridcolor='gray'),
    yaxis=dict(title="Revenue (in $)", gridcolor='gray'),
    legend_title="App Categories",
    plot_bgcolor='black',
    paper_bgcolor="black",
    font_color="white",
    title_font=dict(size=18, color="white"),
    xaxis_title_font=dict(size=14),
    yaxis_title_font=dict(size=14),
    legend_title_font=dict(size=14),
    legend=dict(font=dict(size=12)) 
)

save_plot_as_html(fig11, "revenue_installs_paid_apps.html", "The scatter plot shows the relationship between revenue and installs for paid apps. The trendline indicates a positive correlation between the two variables.")

import webbrowser
webbrowser.open("revenue_installs_paid_apps.html")

# %% [markdown]
# **Task 2.** Create a dual-axis chart comparing the average installs and revenue for free vs. paid apps within the top 3 app categories. Apply filters to exclude apps with fewer than 10,000 installs and revenue below $10,000 and android version should be more than 4.0 as well as size should be more than 15M and content rating should be Everyone and app name should not have more than 30 characters including space and special character .this graph should work only between 1 PM IST to 2 PM IST apart from that time we should not show this graph in dashboard itself.

# %%
#Installing the needed Library
import plotly.graph_objects as go 
from datetime import datetime, time

# %%
#Filter 1: Apps with installs > 10,000
filtered_apps = apps_df[apps_df['Installs'] > 10_000]
filtered_apps

# %%
#Filter 2: Revenue > $10,000
filtered_apps = filtered_apps[filtered_apps['Revenue'] > 10_000]
filtered_apps

# %%
# Converting the 'Android Ver' column to numeric by extracting the numeric part
filtered_apps['Android Ver'] = filtered_apps['Android Ver'].str.extract(r'(\d+(\.\d+)?)')[0]

# %%
# Converting to numeric type
filtered_apps['Android Ver'] = pd.to_numeric(filtered_apps['Android Ver'], errors='coerce')

# %%
# Filter 3: Android version > 4.0
filtered_apps = filtered_apps[filtered_apps['Android Ver'] > 4.0]

# Filter 4: Size > 15M
filtered_apps = filtered_apps[filtered_apps['Size'] > 15.0]

# Filter 5: Content Rating == "Everyone"
filtered_apps = filtered_apps[filtered_apps['Content Rating'] == 'Everyone']

# Filter 6: App name length <= 30 characters (including spaces and special characters)
filtered_apps = filtered_apps[filtered_apps['App'].str.len() <=30]
filtered_apps

 

# %%
#Selecting Top 3 Categories
top_categories = (
    filtered_apps.groupby('Category')['Installs']
    .sum()
    .nlargest(3)
    .index
)
filtered_apps = filtered_apps[filtered_apps['Category'].isin(top_categories)]



# %%
# Calculating average installs and revenue for free and paid apps
comparison_data = filtered_apps.groupby(['Category', 'Type'])[['Installs', 'Revenue']].mean().reset_index()

# %%

# Ensuring that the chart is displayed only between 1 PM and 2 PM IST
start_time = time(13, 0)  
end_time = time(14, 0)

# Getting current time in IST
current_time_utc = datetime.utcnow()
current_time_ist = (current_time_utc + pd.Timedelta(hours=5, minutes=30)).time()

# %%
#Figure 12
if start_time <= current_time_ist <= end_time:
    fig12= go.Figure()

    # Adding bar traces for average revenue
    for c in top_categories:
        fig12.add_trace(
            go.Bar(
                x=comparison_data[comparison_data['Category'] == c]['Type'],
                y=comparison_data[comparison_data['Category'] == c]['Revenue'],
                name=f"{c} Revenue",
                yaxis="y1",  # Mapping to the first y-axis
                text=comparison_data[comparison_data['Category'] == c]['Revenue'],
                textposition='auto'
            )
        )
    
    # Adding line traces for average installs
    for c1 in top_categories:
        fig12.add_trace(
            go.Scatter(
                x=comparison_data[comparison_data['Category'] ==c1]['Type'],
                y=comparison_data[comparison_data['Category'] == c1]['Installs'],
                name=f"{c1} Installs",
                yaxis="y2", # Mapping to the second y-axis
                mode="lines+markers"
            )
        )

    # Layout configuration
    fig12.update_layout(
        title="Average Installs vs Revenue for Free vs Paid Apps (Top 3 Categories)",
        xaxis=dict(title="App Type (Free/Paid)"),
        yaxis=dict(
            title="Average Revenue (in USD)",
            title_font=dict(color="blue"),
            tickfont=dict(color="blue"),
        ),
        yaxis2=dict(
            title="Average Installs (in Millions)",
            title_font=dict(color="green"),
            tickfont=dict(color="green"),
            overlaying="y",
            side="right",
        ),
        
        legend=dict(x=0.1, y=1.1),
        barmode="group"
    )
    # Saving chart as HTML file
    save_plot_as_html(fig12,"dual_axis_chart.html","")
    
# Opening the chart in the browser
    webbrowser.open("dual_axis_chart.html")
else:
    print("Chart is not available outside the time range (1 PM - 2 PM IST).")

# %% [markdown]
# **Task 3.** Use a grouped bar chart to compare the average rating and total review count for the top 10 app categories by number of installs. Filter out any categories where the average rating is below 4.0 and size below 10 M and last update should be Jan month . this graph should work only between 3PM IST to 5 PM IST apart from that time we should not show this graph in dashboard itself.

# %%
print(apps_df['Last Updated'].dtype)

# %%
# Adding a extra "Month" column extracting from "Last Updated" column for filtering according to the task
apps_df['Month'] = apps_df['Last Updated'].dt.month

# %%
# Applying filters
filtered_apps_T3 = apps_df[
    (apps_df['Rating'] >= 4.0) & (apps_df['Size'] >= 10.0) & (apps_df['Month'] == 1)
]
filtered_apps_T3

# %%
# Getting the top 10 categories by number of installs
top_catg_T3= (
    filtered_apps_T3.groupby('Category')['Installs']
    .sum()
    .nlargest(10)
    .index
)
top_catg_T3

# %%
# Now that we have top 10 categories based on the no. of Installs done, moving with filtering the apps based on it.
filtered_apps_T3 = filtered_apps_T3[filtered_apps_T3['Category'].isin(top_catg_T3)]
filtered_apps_T3

# %%
# Calculating average rating and total review counts for each category
comp_T3 = (
    filtered_apps_T3.groupby('Category')[['Rating', 'Reviews']]
    .agg({'Rating': 'mean', 'Reviews': 'sum'})
    .reset_index()
)
comp_T3

# %%
# Ensuring that the chart is displayed only between 3 PM and 5 PM IST
start_T3 = time(15, 0)  
end_T3 = time(17, 0)

current_time_T3 = datetime.utcnow()
current_time_T3 = (current_time_T3+ pd.Timedelta(hours=5, minutes=30)).time()

# %%
#Figure 13
if start_T3  <= current_time_T3  <= end_T3:
    
    fig13 = go.Figure()

    # Adding bar trace for average rating
    fig13.add_trace(
        go.Bar(
            x=comp_T3['Category'],
            y=comp_T3['Rating'],
            name="Average Rating",
            text=comp_T3['Rating'],
            textposition='auto',
            marker_color='purple',
            opacity=0.7
        )
    )

    # Adding bar trace for total review counts
    fig13.add_trace(
        go.Bar(
            x=comp_T3['Category'],
            y=comp_T3['Reviews'],
            name="Total Review Count",
            text=comp_T3['Reviews'],
            textposition='auto',
            marker_color='orange',
            opacity=0.7
        )
    )

    fig13.update_layout(
        title="Comparison of Average Rating and Total Review Counts (Top 10 Categories by Installs)",
        xaxis=dict(title="App Category", tickangle=-45),
        yaxis=dict(title="Average Rating"),
        yaxis2=dict(
            title="Total Review Count",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.1, y=1.1),
        barmode="group"
    )

    # Saving chart as HTML file
    save_plot_as_html(fig13,"grouped_bar_chart.html","")

    # Opening the chart in the browser
    webbrowser.open("grouped_bar_chart.html")
else:
    print("Chart is not available outside the time range (3 PM - 5 PM IST).")

# %% [markdown]
# **Task 4** Plot a time series line chart to show the trend of total installs over time, segmented by app category. Highlight periods of significant growth by shading the areas under the curve where the increase in installs exceeds 20% month-over-month and app name should not starts with x, y ,z and app category should start with letter " E " or " C " or " B " and reviews should be more than 500 as well as this graph should work only between 6 PM IST to 9 PM IST apart from that time we should not show this graph in dashboard itself.

# %%
filtered_data_T4 = apps_df[(apps_df['Content Rating'] == 'Teen')]
filtered_data_T4

# %%
filtered_data_T4 =filtered_data_T4[(filtered_data_T4['App'].str.startswith('E'))]
filtered_data_T4

# %%
filtered_data_T4 = filtered_data_T4[(filtered_data_T4['Installs'] > 10000)]
filtered_data_T4

# %%
# Extract year and month
filtered_data_T4['YearMonth'] = filtered_data_T4['Last Updated'].dt.to_period('M')
filtered_data_T4

# %%
# Step 3: Aggregating the installs by category and month
installs = (
    filtered_data_T4.groupby(['YearMonth', 'Category'])['Installs']
    .sum()
    .reset_index()
    .sort_values('YearMonth')
)

# %%
# Step 4: Calculating the Month-over-Month Growth
installs['Pct_Change'] = (
    installs.groupby('Category')['Installs']
    .pct_change() * 100
)

# %%
# Step 5: Adding a 'Significant Growth' flag
installs['Significant Growth'] = installs['Pct_Change'] > 20
installs

# %% [markdown]
# **Base on the above output we can see that -**
# 1. The 'YearMonth' column offers a clear chronological sequence for monitoring app install trends and is well-structured.
# 2. The monthly percentage change in installs is appropriately shown in the Pct_Change column. Changes, both positive and bad, are accurately documented.
# 3. The Significant Growth column correctly flags periods where the percentage increase in installs exceeds 20%.
# 4. Instances (such as first entries for categories) where the Pct_Change cannot be computed because of missing prior data are suitably denoted as NaN.
# 5. The output accurately depicts key growth periods, including:
# 
# 
# a. Growth rate for 2017-04 (GAME): 1900%. b. FAMILY 2018-05: 2000% growth. c. Family: 9923.08% increase in 2018–07.

# %%
installs.info()

# %%
#Before creating the chart, converting the YearMonth column to a string format first
installs['YearMonth'] = installs['YearMonth'].astype(str)
installs.info()

# %%
#There are values in the 'YearMonth' column that are not in the correct YYYY-MM format. In particular the string may have extra text at the end.
#Checking the actual values in the YearMonth column to understand the format.
print(installs['YearMonth'].unique())

# %%
# After cleaning, converting the 'YearMonth' to the datetime format finally.

installs['YearMonth'] = pd.to_datetime(installs['YearMonth'], format='%Y-%m', errors='coerce')
installs.info()

# %%
# Step 5: Plot the Time Series Chart
current_time = datetime.now().time()
start_time = datetime.strptime("18:00", "%H:%M").time()
end_time = datetime.strptime("21:00", "%H:%M").time()

# %%
#Figure 14
if start_time <= current_time <= end_time:
    # Creating the line chart with Plotly
    fig14 = px.line(
        installs,
        x='YearMonth',
        y='Installs',
        color='Category',
        line_group='Category',
        title="Trend of Total Installs Over Time (Teen, Apps Starting with 'E')",
        labels={'YearMonth': 'Month-Year', 'Installs': 'Total Installs', 'Category': 'App Category'},
    )

    # Highlighting significant growth areas by adding filled areas
    for category in installs['Category'].unique():
        category_data = installs[(installs['Category'] == category) & (installs['Significant Growth'])]
        fig14.add_scatter(
            x=category_data['YearMonth'],
            y=category_data['Installs'],
            fill='tozeroy',
            mode='lines',
            name=f"Significant Growth: {category}",
            opacity=0.3
        )

    # Updating the layout for better visualization
    fig14.update_layout(
        xaxis_title="Month-Year",
        yaxis_title="Total Installs",
        template="plotly_white",
        legend_title="App Categories",
    )

    # Saving the chart as an HTML file
    html_file = "time_series.html"
    fig14.write_html(html_file)

    # Opening the chart in a web browser
    webbrowser.open(html_file)
else:
    print("Time Series Chart is not available outside the time range (6 PM - 9 PM IST).")

# %% [markdown]
# **Task 5.** Plot a bubble chart to analyze the relationship between app size (in MB) and average rating, with the bubble size representing the number of installs. Include a filter to show only apps with a rating higher than 3.5 and that belong to the Game, Beauty ,business , commics , commication , Dating , Entertainment , social and event categories. Reviews should be greater than 500 and sentiment subjectivity should be more than 0.5 and Installs should be more than 50k as well as this graph should work only between 5 PM IST to 7 PM IST apart from that time we should not show this graph in dashboard itself.

# %%
apps_df.info()

# %%
# Filter 1: Rating > 3.5
filtered_data_T5 = apps_df[apps_df['Rating'] > 3.5]
filtered_data_T5

# %%
#Listing all the castegories present along with their counts before applying filter-2

# Count of apps in each category (original apps dataset)
category_counts_original = apps_df['Category'].value_counts()
print("App counts per category in the original dataset:")
print(category_counts_original)

# Count of apps in each category (after applying filter-1)
category_counts_filtered = filtered_data_T5['Category'].value_counts()
print("\nApp counts per category in the filtered dataset after Filter 1:")
print(category_counts_filtered)

# %%
# Filter 2: Category is "GAME"
filtered_data_T5 = filtered_data_T5[filtered_data_T5['Category'] == 'GAME']
filtered_data_T5 

# %%
# Filter 3: Installs > 50,000
filtered_data_T5 = filtered_data_T5 [filtered_data_T5 ['Installs'] > 50000]
filtered_data_T5

# %%
# Time-based Display
start_time = time(17, 0) 
end_time = time(19, 0)   
current_time = datetime.now().time()

# %%
#Figure 15
# Displaying Bubble Chart only within that above time range
if start_time <= current_time <= end_time:
    #Creating the Bubble Chart
    fig15 = px.scatter(
        filtered_data_T5,
        x='Size',
        y='Rating',
        size='Installs',
        color='Installs',
        hover_name='App',
        title="Bubble Chart: Relationship between App Size and Ratings (Games Category)",
        labels={'Size_MB': 'App Size (MB)', 'Rating': 'Average Rating', 'Installs': 'Number of Installs'},
    )

    # Updating the layout for better visualization
    fig15.update_layout(
        xaxis_title="App Size (in MB)",
        yaxis_title="Average Rating",
        coloraxis_colorbar=dict(title="Installs"),
        template="plotly_white",
    )
    
    # Saving the chart as an HTML file
    html_file = "bubble_chart.html"
    fig15.write_html(html_file)
    
    # Opening the chart in a web browser
    webbrowser.open(html_file)
else:
    print("Bubble chart is not available outside the time range (5 PM - 7 PM IST).")

# %% [markdown]
# **Creating Final Dashboard for 5 Tasks**

# %%
import webbrowser

# Saving the dashboard HTML content
dashboard_filename = "Final-Dashboard.html"

dashboard_html = """ 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Analytics Dashboard</title>
    <style>
        /* General Styles */
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #121212;
            color: white;
            transition: background-color 0.5s, color 0.5s;
            overflow-x: hidden;
        }

        /* Light Mode */
        body.light-mode {
            background-color: white;
            color: black;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #34A853, #0F9D58);
            color: white;
            font-size: 24px;
            font-weight: bold;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            position: relative;
        }

        .header img {
            height: 50px;
            cursor: pointer;
            transition: transform 0.3s ease-in-out;
        }

        .light-mode .header {
            background: linear-gradient(90deg, #ffcc00, #ff9900);
        }

        /* Toggle Mode Button */
        .toggle-container {
            position: absolute;
            right: 20px;
            top: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .toggle-icon {
            width: 40px;
            height: 40px;
            transition: transform 0.3s ease-in-out;
        }

        .light-mode .toggle-icon {
            transform: rotate(180deg);
        }

        /* Container */
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            width: 100%;
        }

        /* Plot Cards */
        .plot-card {
            width: 90%;
            height: 600px;
            background: #1E1E1E;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            margin-bottom: 30px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        }

        .light-mode .plot-card {
            background: #f9f9f9;
            color: black;
        }

        .plot-card:hover {
            transform: scale(1.02);
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.4);
        }
        
        /* Disabled Graphs */
        .disabled {
            background: #333 !important;
            cursor: not-allowed;
            color: #bbb;
            text-align: centr;
            font-size: 20px;
            padding: 50px;
        }

        .light-mode .disabled {
            background: #e0e0e0;
            color: #666;
        }

        /* Plot Titles */
        .plot-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
        }

        /* Embed Graphs */
        .plot-card embed {
            width: 100%;
            height: 100%;
            border: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .plot-card {
                width: 100%;
                height: 500px;
            }
        }
    </style>
    <script>
        function openPlot(filename) {
            window.open(filename, '_blank');
        }

        // Toggle Light/Dark Mode
        function toggleMode() {
            document.body.classList.toggle("light-mode");
            let modeIcon = document.getElementById("modeIcon");

            if (document.body.classList.contains("light-mode")) {
                localStorage.setItem("theme", "light");
                modeIcon.src = "https://cdn-icons-png.flaticon.com/512/1164/1164954.png"; // Light mode icon
            } else {
                localStorage.setItem("theme", "dark");
                modeIcon.src = "https://cdn-icons-png.flaticon.com/512/747/747374.png"; // Dark mode icon
            }
        }

        // Load the theme from localStorage
        window.onload = function () {
            if (localStorage.getItem("theme") === "light") {
                document.body.classList.add("light-mode");
                document.getElementById("modeIcon").src = "https://cdn-icons-png.flaticon.com/512/1164/1164954.png";
            }
        };
    </script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg" alt="Google Play Store Logo">
        Google Play Store Review Analytics
        <div class="toggle-container" onclick="toggleMode()">
            <img id="modeIcon" class="toggle-icon" src="https://cdn-icons-png.flaticon.com/512/747/747374.png" alt="Toggle Theme">
        </div>
    </div>
    <div class="container">
        <!-- Available Graphs -->
                
        <div class="plot-card" onclick="openPlot('revenue_installs_paid_apps.html')">
            <embed src="revenue_installs_paid_apps.html">
            <p class="plot-title">Revenue vs Installs (Paid Apps)</p>
        </div>

        <!-- Time-Restricted Graphs -->
        <div class="plot-card disabled">
            <p class="plot-title">Dual Axis Chart (Available 1 PM - 2 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Grouped Bar Chart (Available 3 PM - 5 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Time Series Chart (Available 6 PM - 9 PM)</p>
        </div>
        <div class="plot-card disabled">
            <p class="plot-title">Bubble Chart (Available 5 PM - 7 PM)</p>
        </div>
       
    </div>
</body>
</html>
"""

# Saving the dashboard as an HTML file
with open(dashboard_filename, "w", encoding="utf-8") as file:
    file.write(dashboard_html)

# Opening the dashboard in the web browser
webbrowser.open(dashboard_filename)

print("Dashboard has been successfully opened in the browser.")

# %%
import webbrowser
import http.server
import socketserver
import threading

# Defining the port
PORT = 8000

# Function to start the server in a separate thread
def start_server():
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

# Starting the server in the background
threading.Thread(target=start_server, daemon=True).start()

# Opening the dashboard in the browser
webbrowser.open(f"http://localhost:{PORT}/Final-Dashboard.html")
print("Dashboard is now accessible at http://localhost:8000/Final-Dashboard.html")


