def gf_freq(dataframe, column, agg, title):
    import matplotlib.pyplot as plt
    dataframe[column].asfreq(agg).plot(linewidth=3)
    plt.title(title +' per '+agg)
    plt.xlabel(agg)
    plt.xlabel(title);

def gf_nullvalues(dataFrame):
    import seaborn as sns
    sns.heatmap(dataFrame.isnull())

def gf_epochhistory(epochhist):
    import matplotlib.pyplot as plt
    plt.plot(epochhist.history['loss'])
    plt.plot(epochhist.history['val_loss'])
    plt.title('Model loss progress during training')
    plt.xlabel('Epochs')
    plt.ylabel('Training and Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss' ])

def gf_bubblechart(df, x, y):
    import plotly.express as px
    df_sorted = df = df.sort_values(y, ascending = False)
    fig = px.scatter(
        df, y=y, x=x, color='Exports', size='Exports', size_max=20,
        color_continuous_scale = px.color.sequential.RbBu,
    )
    fig.update_layout(
        paper_bg_color='white',
        plot_bgcolor='white'
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_layout(height=500, width=1000)
    fig.update_coloraxes(colorbar=dict(title='Exports'))
    fig.update_traces(marker=dict(sizeref=0.09))
    fig.update_yaxes(title=y)
    fig.update_xaxes(title=x)
    fig.update_layout(showlegend=False)
    fig.show()