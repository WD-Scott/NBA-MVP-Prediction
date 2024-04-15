import requests
import pandas as pd
from io import StringIO
import plotly.graph_objs as go

DATA_URL = "https://raw.githubusercontent.com/WD-Scott/DS5110_Project/main/Data%20Files/interactive.csv"

def fetch_csv(data_url: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(data_url).text))

def display_bar_chart(df, player_name, year, outfile):
    if player_name is None or year is None:
        print("Please select a player and a year.")
        return

    filtered_data = df[(df['name'] == player_name) & (df['Season'] == year)]
    if filtered_data.empty:
        print("No data found for the selected player and year.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=filtered_data["name"],
                         y=filtered_data["predicted"],
                         name="Predicted",
                         marker_color='rgb(229, 114, 0)'
                         ))
    fig.add_trace(go.Bar(x=filtered_data["name"],
                         y=filtered_data["actual"],
                         name="Actual",
                         marker_color='rgb(35, 45, 75)'
                         ))

    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                      title=f"Player: {player_name}, Year: {year}",
                      xaxis=dict(title='Player'),
                      yaxis=dict(title='Value'),
                      legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)')
                      )

    # Save the HTML file
    fig.write_html(outfile, include_plotlyjs='cdn')

if __name__ == "__main__":
    df = fetch_csv(DATA_URL)

    # Generate the interactive chart
    display_bar_chart(df, player_name=None, year=None, outfile="index.html")