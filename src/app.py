import os
import pandas as pd
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from flask import Flask

# Read the data from CSV
df = pd.read_csv("./imports/sentiments/sentiment_results.csv")
df_ner = pd.read_csv("./imports/ner/ner_results.csv")

# Define the app
app = dash.Dash(__name__, routes_pathname_prefix='/graph/')
server = app.server
debug = False if os.environ["DASH_DEBUG_MODE"] == "False" else True


# Define the layout
app.layout = html.Div([
    html.H1("Sentiment Analysis Results"),
    html.Div([
        html.Button("0 to 3 days", id="btn-0-3", n_clicks=0),
        html.Button("4 to 7 days", id="btn-4-7", n_clicks=0),
        html.Button("8 to 16 days", id="btn-8-16", n_clicks=0),
    ]),
    html.Div([
        dcc.Graph(id="graph-1"),
        dcc.Graph(id="graph-2"),
    ], style={"display": "flex", "flex-direction": "row"}),
     html.Div([
        dcc.Graph(id="graph-3"),
        dcc.Graph(id="graph-4"),
    ], style={"display": "flex", "flex-direction": "row"})
])


# Define the callbacks
@app.callback(
    [Output("graph-1", "figure"), Output("graph-2", "figure"), Output("graph-3", "figure"), Output("graph-4", "figure")],
    [Input("btn-0-3", "n_clicks"), Input("btn-4-7", "n_clicks"), Input("btn-8-16", "n_clicks")],
    [State("graph-1", "figure"), State("graph-2", "figure"),State("graph-3", "figure"), State("graph-4", "figure")]
)

def update_figure(btn1_clicks, btn2_clicks, btn3_clicks, figure1, figure2, figure3, figure4):
    # Determine which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # default chart
    
    # sentiment analysis
    row = df.iloc[0]
    xname = "Between 0 and 3 Days"
    
    # ner analysis
    row_ner = df_ner.iloc[0]
    xname = "Between 0 and 3 Days" 
    

    # Determine which row to use
    if button_id == "btn-0-3":
        row = df.iloc[0]
        row_ner = df_ner.iloc[0]
        xname = "Between 0 and 3 Days"
    elif button_id == "btn-4-7":
        row = df.iloc[1]
        row_ner = df_ner.iloc[1]
        xname = "Between 4 and 7 Days"
    elif button_id == "btn-8-16":
        row = df.iloc[2]
        row_ner = df_ner.iloc[2]
        xname = "Between 8 and 16 Days"
        
    # sentiment analysis values    

    pos = row["total_tweets"] - row["negative_tweets"]
    neg = row["negative_tweets"]
    
    # ner analysis values
    
    Turkey = row_ner["Turkey"]
    Syria = row_ner["Syria"]
    US = row_ner["US"]
    Ukraine = row_ner["Ukraine"]
    Israel = row_ner["Israel"]
    Hatay = row_ner["Hatay"]
    Allah = row_ner["Allah"]
    Lebanon = row_ner["Lebanon"]
    Twitter = row_ner["Twitter"]
    Facebook = row_ner["Facebook"]
    UN = row_ner["UN"]


    # Create the bar chart
    data1 = [go.Bar(x=["positive_tweets", "negative_tweets"], y=[pos, neg])]
    layout1 = go.Layout(title=xname)
    fig1 = go.Figure(data=data1, layout=layout1)
    

    # Create the pie chart
    labels = ["Negative Tweets", "Positive Tweets"]
    values = [row["negative_tweets"], row["total_tweets"] - row["negative_tweets"]]

    data2 = [go.Pie(labels=labels, values=values)]
    layout2 = go.Layout(title="Negative Tweets & Positive Tweets")
    fig2 = go.Figure(data=data2, layout=layout2)
    
    # Create the bar chart 2
    data3 = [go.Bar(x=["Turkey", "Syria", "US", "Ukraine" , "Israel" , "Hatay" , "Allah" , "Lebanon", "Twitter", "Facebook", "UN"],y=[row_ner["Turkey"], row_ner["Syria"], row_ner["US"], row_ner["Ukraine"], row_ner["Israel"], row_ner["Hatay"], row_ner["Allah"], row_ner["Lebanon"], row_ner["Twitter"], row_ner["Facebook"], row_ner["UN"]])]
    layout3 = go.Layout(title=xname)
    fig3 = go.Figure(data=data3, layout=layout3)
    
     
    # Create the pie chart 2
    labels_pie2 = ["Turkey", "Syria", "US", "Ukraine" , "Israel" , "Hatay" , "Allah" , "Lebanon", "Twitter", "Facebook", "UN"]
    values_pie2 = [row_ner["Turkey"], row_ner["Syria"], row_ner["US"], row_ner["Ukraine"], row_ner["Israel"], row_ner["Hatay"], row_ner["Allah"], row_ner["Lebanon"], row_ner["Twitter"], row_ner["Facebook"], row_ner["UN"]]
    
    data4 = [go.Pie(labels=labels_pie2, values=values_pie2)]
    layout4 = go.Layout(title="NER Results")
    fig4 = go.Figure(data=data4, layout=layout4)

    return fig1, fig2, fig3, fig4


# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=debug)
