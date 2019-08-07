import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
# from app import app

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-3dscatterplot'
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv")
#df_list=['meta_data__model_1.json','meta_data__model_2.json','meta_data__model_3.json','meta_data__model_4.json',\
    # 'meta_data__model_5.json']
#df_list=['meta_data_amzon_ff_model_2.json','meta_data_amzon_ff_model_9.json','meta_data_amzon_ff_model_11.json','meta_data_amzon_ff_model_15.json']

def read_path(df,path,dat):
    #print( os.listdir(path))
    files = [ fil for fil in os.listdir(path) if fil.endswith('.json')]
    for f  in files:
        f = path + '/'+f
        print(f)
        df[dat].append(pd.read_json(f))  
    return df


data_folders = ['amz_ffr_json','bank_json','kick_json']
df = {'amz_ffr_json': [],'bank_json': [],'kick_json': []}
amz_sub_dir = ['baseline','baseline_bow','baseline_sentiment_bow','baseline_sentiment'] 
config=['/config1','/config2']
for fol in data_folders:
    for con in config:
        for alg in ['machine_learning','deep_learning']:
            path=os.getcwd()+'/'+fol+'/json'+ con +'/'+ alg
            if fol is 'amz_ffr_json':
                for sub in amz_sub_dir:
                    path_amz = path +'/'+ sub
                    print(path_amz)
                    df = read_path(df,path_amz,fol)
            else:
                print(path)
                df = read_path(df,path,fol)
print(df)

                 

app.layout = html.Div([
    html.Div([html.H1("Performance Stats")],
             style={'textAlign': "center", "padding-bottom": "10", "padding-top": "10"}),
    html.Div(
        [html.Div(dcc.Dropdown(id="select-file", options=[{'label': data_folders[0], 'value': data_folders[0]},{'label': data_folders[1], 'value': data_folders[1]},\
            {'label': data_folders[2], 'value': data_folders[2]} ],
                               value='kick_json', ), className="four columns",
                  style={"display": "block", "margin-left": "auto",
                         "margin-right": "auto", "width": "33%"})],className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "80%"}),
        # [html.Div(dcc.Dropdown(id="select-model", options=[{'label': df[i].columns[0].title(), 'value': [i,df[i].columns[0]]} for i in range(len(df)) ],
        #                        value=[0,df[0].columns[0]], ), className="four columns",
        #           style={"display": "block", "margin-left": "auto",
        #                  "margin-right": "auto", "width": "33%"})],className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "80%"}),
    
    html.Div([dcc.Graph(id="my-graph")])
], className="container")


@app.callback(
    dash.dependencies.Output("my-graph", "figure"),
    [dash.dependencies.Input("select-file", "value"),]#dash.dependencies.Input("select-model", "value"),]

)

def update_figure(select_file):
    
    # model_name = select_model[1]#df.columns[0]
    # index = select_model[0]
    # print(index)
    # print(model_name)
    # model=df[index]
    # z = [model[model_name]['Metrics']['Error']]
    # y = [model[model_name]['Metrics']['Training_Time_in_s']]
    # x = [model[model_name]['Metrics']['Accuracy']]
    names=['dummy']
    x,y,z=[0],[0],[0]  # the first entries of this list are dummmy values to \
                       #indicate the origin
    print(select_file)
    frame = df[select_file]
    for model in frame:
        model_name=model.columns[0]
        names.append(model_name)
        z.append(model[model_name]['Metrics']['Error'])
        y.append(model[model_name]['Metrics']['Training_Time_in_s'])
        x.append(model[model_name]['Metrics']['Accuracy'])
    
    #print(x,y,z,names)
    trace = [go.Scatter3d(
        x=x,y=y,z=z,
        mode='markers', marker={'size': 8, 'color': z, 'colorscale': 'Blackbody', 'opacity': 0.8, "showscale": True,
                                "colorbar": {"thickness": 15, "len": 0.5, "x": 0.8, "y": 0.6, }, },hovertext=names,xsrc='0',
        ysrc='0',zsrc='0',)]
    return {"data": trace,
            "layout": go.Layout(
                height=700, title=f"Metrics<br>{'Dataset Name='+select_file, 'accuracy','training_time', 'error'}",
                paper_bgcolor="#f3f3f3",
                scene={"aspectmode": "cube", "xaxis": {"title": f"{'X:Accuracy'}", },
                       "yaxis": {"title": f"{'Y:training_time'} (s)", },
                       "zaxis": {"title": f"{'Z:Error'} ", }})
            }

if __name__ == '__main__':
    app.run_server(debug=True,port=8051)

