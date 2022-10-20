import random
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.linear_model import LinearRegression

def boxplot_fig(df, box_title):
    fig = go.Figure()

    for var in df[df.select_dtypes(exclude=['object']).columns.tolist()]:
        fig.add_trace(go.Box(y=df[var],
                             name=var,
                             marker_color=["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]
        ))
    
    fig.update_layout(yaxis_title='Value',
                      xaxis_title='Variable',
                      title=box_title,
                      showlegend=False,
                      width=1280,
                      height=480
    )

    
    return fig

def imp_miss_data(df, labels_drop):
    miss_columns = df.columns[df.isna().any()].tolist()
    
    df_lr = df.copy()
    df_lr = df_lr.drop(labels=labels_drop, axis=1)
    all_col = df_lr.columns.tolist()

    for col in miss_columns:
        train_df = df_lr.copy()
        test_df = df_lr[df_lr[col].isna() == True]
        train_df = train_df.drop(test_df.index.values.tolist())

        # temporary change the nan value of other column to 0
        test_df = test_df.fillna(0.0)
        train_df =train_df.fillna(0.0)
    
        y_train = train_df[col]
        x_train = train_df.drop(labels=[col], axis=1)
    
        # create linear regression model
        model = LinearRegression()
        model.fit(x_train, y_train)
    
        # drop the column where we want to do the prediction on missing values
        test_df = test_df.drop(labels=[col], axis=1)
    
        # predict the missing value
        pred = model.predict(test_df)
        test_df[col] = pred
    
        df_lr = train_df.append(test_df)
    
        # return 0.0 value to nan
        df_lr[all_col] = df_lr[all_col].replace(0, np.nan)
    
        # sort the index
        df_lr = df_lr.sort_index()
    
    # don't forget to return df_lr to df_1
    df[miss_columns] = df_lr[miss_columns]

    return df

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

