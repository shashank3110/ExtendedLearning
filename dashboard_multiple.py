import os
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
# from app import app

if 'DYNO' in os.environ:
    app_name = os.environ['DASH_APP_NAME']
else:
    app_name = 'dash-3dscatterplot'

def read_path(df,path,dat,al):
    #print( os.listdir(path))
    print(al)
    for algos in al:
        print(algos)
        files = [ fil for fil in os.listdir(path) if fil.endswith('.json') and algos in fil]
        for f  in files:
            f = path + '/'+f
            print(f)
            df[dat].append(pd.read_json(f))  
    return df

algo= ['Deep Learning','SVM','Random Forest','NB','SVC','Logistic']
configs= ['config1','config2']
data_folders = ['amz_baseline','amz_bow_base','amz_bow_sentiment_base','amz_sentiment_base','bank_json','kick_json']

def get_data(al,conf,data_folders):

    #data_folders = ['amz_ffr_json','bank_json','kick_json']
    df = {'amz_baseline': [],'amz_bow_base':[],'amz_bow_sentiment_base':[],'amz_sentiment_base':[],'bank_json': [],'kick_json': []}
    amz_sub_dir = ['baseline','baseline_bow','baseline_sentiment_bow','baseline_sentiment'] 
    config=['/config1','/config2']
    for fol in data_folders:
        for con in config:
            if conf in con:
                for alg in ['machine_learning','deep_learning']:
                    path=os.getcwd()+'/'+fol+'/json'+ con +'/'+ alg
                    if fol is 'amz_ffr_json':
                        for sub in amz_sub_dir:
                            path_amz = path +'/'+ sub
                            print(path_amz)
                            df = read_path(df,path_amz,fol,al)
                    else:
                        print(path)
                        df = read_path(df,path,fol,al)
    print(df)
    return df

                 

app.layout = html.Div(
    className = "row",
    children = [
    html.Div([
    html.Div([html.H1("Performance Stats")],
             style={'textAlign': "center", "padding-bottom": "10", "padding-top": "10"}),
    html.Div(
        [html.P('Select Data Set:'),html.Div(dcc.Dropdown(id="select-file", options=[{'label': data_folders[0], 'value': data_folders[0]},{'label': data_folders[1], 'value': data_folders[1]},\
            {'label': data_folders[2], 'value': data_folders[2]},{'label': data_folders[3], 'value': data_folders[3]},{'label': data_folders[4], 'value': data_folders[4]},
                                                          {'label': data_folders[5], 'value': data_folders[5]}],
                               value='kick_json', ), className="four columns",
                  style={"display": "block", "margin-left": "auto",
                         "margin-right": "auto", "width": "33%"})],className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "80%"}),
    html.Div(
        [html.P('Select algorithm family:'),html.Div(dcc.Dropdown(id="select-algo", options=[{'label': algo[0], 'value': algo[0]},{'label': algo[1], 'value': algo[1]},\
            {'label': algo[2], 'value': algo[2]},{'label': algo[3], 'value': algo[3]},{'label': algo[4], 'value': algo[4]},\
                {'label': algo[5], 'value': algo[5]}],
                               value=['Deep'],multi=True ), className="four columns",
                  style={"display": "block", "margin-left": "auto",
                         "margin-right": "auto", "width": "33%"})],className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "80%"}),
    html.Div(
        [html.P('Select HW config:'),html.Div(dcc.Dropdown(id="select-config", options=[{'label': configs[0], 'value': configs[0]},{'label': configs[1], 'value': configs[1]}],\
            value='config1',multi=False ), className="four columns",
                  style={"display": "block", "margin-left": "auto",
                         "margin-right": "auto", "width": "33%"})],className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "80%"}),
        
    html.Div([dcc.Graph(id="my-graph")])
], className="container"),

html.Div(dash_table.DataTable(
    id='my-table',
    columns=[
        {"name": col, "id": col} for col in ['Model Name','Accuracy','Precision','Training Time (in s)']
    ],
    
))])

def update_points_set(model,model_name):
    if 'Amazon' in model[model_name]['Data_Meta_Data']['Name']:
            avgPrecision = np.mean(model[model_name]['Metrics']['Precision'])
            y_coordinate = avgPrecision
    else:
            y_coordinate = model[model_name]['Metrics']['Precision']
    z_coordinate = model[model_name]['Metrics']['Training_Time_in_s']
    x_coordinate = model[model_name]['Metrics']['Accuracy']

    return x_coordinate,y_coordinate,z_coordinate

@app.callback(
    dash.dependencies.Output("my-graph", "figure"),
    [dash.dependencies.Input("select-file", "value"),\
    dash.dependencies.Input("select-algo", "value"),\
    dash.dependencies.Input("select-config", "value")]#dash.dependencies.Input("select-model", "value"),]

)

def update_figure(select_file,select_algo,select_config):
    
    
    names=['origin']
    familyNames=['black']
    x,y,z=[0],[0],[0]  # the first entries of this list are dummmy values to \
                       #indicate the origin
    print(select_file)
    df= get_data(select_algo,select_config,data_folders)
    frame = df[select_file]
    
    color_map = {'Deep Learning': '#4287f5','NB':'#4ef542','Decision tree':'#42bcf5',\
    'LogisticRegression':'#f5424b','SVC':'#ad42f5','Linear SVM':'#ad42f5',\
    'Random Forest': '#f5b042','Extra Random Forest':'#eff542','Gradient Boosting':'tomato'}
    
    for model in frame:
        model_name=model.columns[0]
        names.append(model_name)
        family = re.sub(r"\d*$", "", model_name)

        for key,val in color_map.items():
            if key in family:
                color = val
        
        familyNames.append(color)

        x_coordinate,y_coordinate,z_coordinate = update_points_set(model,model_name)

        x.append(x_coordinate)
        y.append(y_coordinate)
        z.append(z_coordinate)

        # if 'Amazon' in model[model_name]['Data_Meta_Data']['Name']:
        #     avgPrecision = np.mean(model[model_name]['Metrics']['Precision'])
        #     y.append(avgPrecision)
        # else:
        #     y.append(model[model_name]['Metrics']['Precision'])
        # z.append(model[model_name]['Metrics']['Training_Time_in_s'])
        # x.append(model[model_name]['Metrics']['Accuracy'])
    
    
    #print(x,y,z,names)
    trace = [go.Scatter3d(
        x=x,y=y,z=z,
        mode='markers', marker={'size': 8, 'color': familyNames, 'colorscale': 'Blackbody', 'opacity': 0.8, "showscale": False,
                                "colorbar": {"thickness": 15, "len": 0.5, "x": 0.8, "y": 0.6, }, },hovertext=names,xsrc='0',
        ysrc='0',zsrc='0',)]
    return {"data": trace,
            "layout": go.Layout(
                height=700, title=f"Metrics<br>{'Dataset Name='+select_file, 'accuracy','Precision', 'training_time','Number of Models= '+str(len(names)-1)}",
                paper_bgcolor="#f3f3f3",
                scene={"aspectmode": "cube", "xaxis": {"title": f"{'X:Accuracy'}", },
                       "yaxis": {"title": f"{'Y:Precision'}", },
                       "zaxis": {"title": f"{'Z:training_time'} (s) ", }})
            }

@app.callback(
    dash.dependencies.Output("my-table", "data"),
    [dash.dependencies.Input("select-file", "value"),\
    dash.dependencies.Input("select-algo", "value"),\
    dash.dependencies.Input("select-config", "value")]
)

def update_table(select_file,select_algo,select_config):

    df= get_data(select_algo,select_config,data_folders)
    frame = df[select_file]
    x,y,z, names =[],[],[],[]
    cols = ['Model Name','Accuracy','Precision','Training Time (in s)']
    for model in frame:
        model_name=model.columns[0]
        names.append(model_name)
        
        x_coordinate,y_coordinate,z_coordinate = update_points_set(model,model_name)

        x.append(x_coordinate)
        y.append(y_coordinate)
        z.append(z_coordinate)
    print(x,y,z,names)
    data = pd.DataFrame(columns= cols,data = list(zip(*[names,x,y,z])))
    data_list = data.to_dict('records')
    print(data_list)
    return data_list
# {'data': [names,x,y,z]
#     }


if __name__ == '__main__':
    app.run_server(debug=True,port=8056)
