import pandas as pd
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Read the data from CSV
df = pd.read_csv("sentiments/sentiment_results.csv")

# Define the app
app = dash.Dash(__name__, routes_pathname_prefix='/graph/')


# Define the layout
app.layout = html.Div([
    html.H1("Sentiment Analysis Results"),
    html.Div([
        html.Button("0 to 3 days", id="btn-0-3", n_clicks=0),
        html.Button("4 to 7 days", id="btn-4-7", n_clicks=0),
        html.Button("8 to 16 days", id="btn-8-16", n_clicks=0),
    ]),
    dcc.Graph(id="graph-1"),
    dcc.Graph(id="graph-2")
])


# Define the callbacks
@app.callback(
    [Output("graph-1", "figure"), Output("graph-2", "figure")],
    [Input("btn-0-3", "n_clicks"), Input("btn-4-7", "n_clicks"), Input("btn-8-16", "n_clicks")],
    [State("graph-1", "figure"), State("graph-2", "figure")]
)

def update_figure(btn1_clicks, btn2_clicks, btn3_clicks, figure1, figure2):
    # Determine which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # default chart
    row = df.iloc[0]
    xname = "Between 0 and 3 Days"

    # Determine which row to use
    if button_id == "btn-0-3":
        row = df.iloc[0]
        xname = "Between 0 and 3 Days"
    elif button_id == "btn-4-7":
        row = df.iloc[1]
        xname = "Between 4 and 7 Days"
    elif button_id == "btn-8-16":
        row = df.iloc[2]
        xname = "Between 8 and 16 Days"

    pos = row["total_tweets"] - row["negative_tweets"]
    neg = row["negative_tweets"]

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

    return fig1, fig2


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=5000)
