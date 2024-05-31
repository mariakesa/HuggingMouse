import plotly.express as px

import pandas as pd


def plot():
    merged_data = pd.read_csv('merged_data_df.csv')
    print('plotting')
    merged_data.drop(columns=['Unnamed: 0'], inplace=True)

    # Exclude the 'cell_ids' column from the heatmap
    data = merged_data.drop(columns=['cell_ids'])

    data.clip(lower=-1, inplace=True)

    # Create the heatmap
    fig = px.imshow(data,
                    labels=dict(x="Trials", y="Neurons", color="Score"),
                    x=data.columns,
                    y=data.index
                    )

    # Update x-axis to place it on the top
    fig.update_xaxes(side="top")

    # Add custom hover text
    hover_text = [[f'cell_id: {cell_id}<br>Trial: {trial}<br>Score: {score}'
                   for trial, score, cell_id in zip(merged_data.columns[1:], row[1:], [int(row['cell_ids'])] * len(merged_data.columns[1:]))]
                  for idx, row in merged_data.iterrows()]

    fig.data[0].update(hovertemplate='<br>'.join([
        '%{customdata}'
    ]))

    fig.update_traces(customdata=hover_text)

    fig.show()


merged_data = pd.read_csv('merged_data_df.csv')
merged_data.drop(columns=['Unnamed: 0'], inplace=True)
hover_text = [[f'cell_id: {cell_id}<br>Trial: {trial}<br>Score: {score}'
               for trial, score, cell_id in zip(merged_data.columns[1:], row[1:], merged_data['cell_ids'])]
              for idx, row in merged_data.iterrows()]
print(hover_text)
print(len(hover_text))
print(len(merged_data))
print(merged_data)
print(len(hover_text[0]))
print(len(hover_text))
plot()

hover_text = [print([row['cell_ids']])
              for idx, row in merged_data.iterrows()]
