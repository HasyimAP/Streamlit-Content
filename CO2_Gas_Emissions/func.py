import plotly.graph_objects as go
import plotly.figure_factory as ff

def the_histogram(df, col_selected):
    data = [x for x in df[col_selected]]
    fig = ff.create_distplot([data], 
                             [col_selected],
                             curve_type='kde',
                             show_rug=False
                            )

    fig2 = ff.create_distplot([data], 
                             [col_selected],
                             curve_type='normal',
                             show_rug=False
                            )

    normal_x = fig2.data[1]['x']
    normal_y = fig2.data[1]['y']

    fig.add_traces(go.Scatter(x=normal_x, 
                              y=normal_y, 
                              mode = 'lines',
                              line = dict(color='red',
                                          dash = 'dash',
                                          width = 2),
                              name = 'normal distribution'
                             ))

    fig.update_layout(xaxis_title='Value',
                      yaxis_title='Density of Probability',
                      width=1280, 
                      height=480
                     )

    return fig

def single_boxplot(df, col_select):
    fig = go.Figure()

    fig.add_trace(go.Box(
        x=df[df['country'] == 'World'][col_select],
        name=col_select,
        boxmean='sd'
    ))

    fig.update_layout(
        width=1280,
        height=480,
        showlegend=False
    )

    return fig

