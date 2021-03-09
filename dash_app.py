#load library
from pycaret.regression import *
import pandas as pd
import joblib
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import base64


# load data
data=pd.read_csv('diamonds.csv') 
data.drop('Unnamed: 0',inplace=True,axis=1)


#saved_model = load_model('model_catboost')
#saved_model=joblib.load('model_catboost.pkl')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    # headig
    html.Center(html.H1('Diamond Dataset Dashboard And Model Deployment'),                
    style={'color': 'blue','background-color':'black','opacity': 0.6,'fontSize': '300%','border-radius': '25px','font-family':'verdana'}),
    #block 1   sample dataset
    html.Div([
        html.Center(html.H1('Diamonds data samples'), style={'color': 'black', 'fontSize': 5}),
        dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i}  for i in data.head(10).columns],
        data=data.sample(10).to_dict('records'),
        style_cell=dict(textAlign='left'),
        style_header=dict(backgroundColor="paleturquoise"),
        style_data=dict(backgroundColor="lavender"))
    ]),
    # block2
    html.Div([

    #block 21
    html.Div([
    html.Center(html.H1(' Scatter plot of variables'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
    html.Label('X -axis'),         
    dcc.Dropdown(
                id='xaxis_name1',
                options=[{'label': i, 'value': i} for i in ['depth','table','price','x','y','z']],
                value='depth',
                clearable=False,
            ),
    html.Label('Y -axis'),
    dcc.Dropdown(
                id='yaxis_name1',
                options=[{'label': i, 'value': i} for i in ['depth','table','price','x','y','z']],
                value='x',
                clearable=False,
            ),
        dcc.Graph(id="graph1")
        
    ],style={'width': '50%', 'display':'inline-block','backgroundColor':'#111111',
             'color':'red','fontSize':20,'border-radius': '25px'}),
    #block 22
    html.Div([
        html.Center(html.H1(' Pie chart of variables'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
        html.Label('X -axis'),
        dcc.Dropdown(
                id='xaxis_name2',
                options=[{'label': i, 'value': i} for i in ['cut','color','clarity']],
                value='cut',
                
               
        ),
        html.Label('Y -axis'),
        dcc.Dropdown(
                id='yaxis_name2',
                options=[{'label': i, 'value': i} for i in ['depth','table','price','x','y','z']],
                value='x',
        ),
        dcc.Graph(id='graph2')
    ],style={'width': '50%', 'display': 'inline-block','backgroundColor':'#111111',
             'color':'blue','fontSize':20,'border-radius': '25px'})
        
    ]),
    # block 3
    html.Div([
        
        # block 31
        html.Div([
            html.Center(html.H1('Bar plot of variables '),style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
            dcc.Graph(id='graph3')],
            style={'width': '50%', 'display': 'inline-block','backgroundColor':'#111111',
                   'color':'red','fontSize':20,'border-radius': '25px','padding': '5px,5px,5px,5px'}),
        # block 32
        html.Div([
            html.Center(html.H1(' Tree plot of variables'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
            dcc.Graph(id='graph4')],
            style={'width': '50%','display': 'inline-block','backgroundColor':'#111111',
                   'color':'red','fontSize':20,'border-radius': '25px','padding': '5px,5px,5px,5px'})        
    ]),
    
    
    # block 4
    html.Div([
        
        # block 41
        html.Div([
        html.Center(html.H1('Histogram plot of variable'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
        html.P("Numeric variable :"),
        dcc.RadioItems(
                id='hist_var', 
                options=[{'value': x, 'label': x} 
                for x in ['depth','table','price','x','y','z']],
                value='x', 
                labelStyle={'display': 'inline-block','color':'white'}),
                dcc.Graph(id='graph7')],
        style={'width': '50%', 'display': 'inline-block','backgroundColor':'#111111',
             'color':'#7FDBFF','fontSize':20,'border-radius': '25px'}),
        
        # block 42
        html.Div([
            html.Center(html.H1('Correlation matrix of numeric variables'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
            html.P("Selecte  multiple variable :"),
            dcc.Checklist(
            id='col_list',
            options=[{'label': x, 'value': x} 
            for x in ['carat','depth','table','price','x','y','z']],
            value=['carat','depth','table','price','x','y','z'],
            labelStyle={'display': 'inline-block','color':'white'}),
            dcc.Graph(id='graph6')],
            style={'width': '50%', 'display': 'inline-block','backgroundColor':'#111111',
                   'color':'red','fontSize':20,'border-radius': '25px'})        
       ]),
    
    # block 5
     html.Div([
            
            html.Center(html.H1('Box plot of variables '),style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
            
            html.P("Categorical variable :"),
            
            dcc.RadioItems(
                id='bx1', 
                options=[{'value': x, 'label': x} 
                for x in ['cut','color','clarity']],
                value='cut', 
                labelStyle={'display': 'inline-block','color':'white'}),
        
            html.P("Numeric variable :"),
            
            dcc.RadioItems(
                id='bx2', 
                options=[{'value': x, 'label': x} 
                for x in ['depth','table','price','x','y','z']],
                value='x', 
                labelStyle={'display': 'inline-block','color':'white'}),
            
            html.P("Choose grouped variables :"),
            dcc.RadioItems(
                id='bx3', 
                options=[{'value': x, 'label': x} 
                for x in ['cut','color','clarity']],
                value='color',
                labelStyle={'display': 'inline-block','color':'white'}),
            
            dcc.Graph(id='graph5')],
            style={'width': '100%', 'display': 'inline-block','backgroundColor':'#111111',
                   'color':'red','fontSize':20,'border-radius': '25px'}),
    
    #block 6
    html.Div([ 
        
        #block 61
        html.Div([ 
            
        html.Center(html.H1('Diamond price prediction'), style={'color': 'white', 'fontSize': 10,'font-family': "Sofia"}),
        html.Label('Cart :'),
        dcc.Slider(id='n1',min=0,max=6,step=0.1,value=5),
        html.Label('Cut :'),
        dcc.Dropdown(
                id='n2',
                options=[{'label': i, 'value': i} for i in data['cut'].unique().tolist()],
                value='Good :'),
        html.Label('Color :'),
        dcc.Dropdown(
                id='n3',
                options=[{'label': i, 'value': i} for i in data['color'].unique().tolist()],
                value='E'),
        html.Label('Clarity :'),
        dcc.Dropdown(
                id='n4',
                options=[{'label': i, 'value': i} for i in data['clarity'].unique().tolist()],
                value='SI1'),
        html.Label('Depth :'),
        dcc.Slider(id='n5',min=50,max=100,step=1,value=46,
                   marks={50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
        html.Label('Table :'),
        dcc.Slider(id='n6',min=20,max=100,step=1,value=40,
                  marks={20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
        html.Label('x'),
        dcc.Slider(id='n8',min=0,max=100,step=1,value=10,
                  marks={1:'1',10:'10',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
        html.Label('y'),
        dcc.Slider(id='n9',min=0,max=100,step=1,value=10,
                  marks={1:'1',10:'10',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
        html.Label('z :'),
        dcc.Slider(id='n7',min=0,max=100,step=1,value=1000,
                  marks={1:'1',10:'10',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),
        html.Center(html.H2(id='pred'), style={'color': 'rgba(0, 255, 0, 0.9)', 'fontSize': 15})],
        style={'width': '40%', 'display': 'inline-block','backgroundColor':'#111111',
             'color':'#7FDBFF','fontSize':20,'border-radius': '25px'}),
        
        # block 62
        html.Div([    
            html.Center(html.H1('Business Scenario'), style={'color':'white', 'fontSize': 20,'font-family': "Sofia"}),
            html.P('''Business Scenario:A diamond merchant has come to you for help. 
            They want to create an automated system to predict the apt price of a diamond based on its shape/size/color etc. '''),
            html.Hr(),
            html.P(''' Our task is to create a machine learning model which can predict the price of a diamond based on its
            characteristics.The business meaning of each column in the data is as below '''),
             html.Hr(),
            dcc.Markdown('* **price:** The price of the Diamond.'),
            dcc.Markdown('* **carat:** The carat value of the Diamond.'),
            dcc.Markdown('* **cut:** The cut type of the Diamond, it determines the shine.'),
            dcc.Markdown('* **color:** The color value of the Diamond.'),
            dcc.Markdown('* **clarity:** The carat type of the Diamond .'),
            dcc.Markdown('* **depth:** The depth value of the Diamond.'),
            dcc.Markdown('* **table:** Flat facet on its surface.'),
            dcc.Markdown('* **x:** Width of the diamond.'),      
            dcc.Markdown('* **y:** Length of the diamond.'),
            dcc.Markdown('* **z:** Height of the diamond.')],
            style={'width': '60%','display': 'inline-block','backgroundColor':'black','opacity': 0.6,
             'color':'#7FDBFF','fontSize':20,'border-radius': '25px'})
            
        ])
        
    ])
        




@app.callback(Output("graph1", "figure"),Input("xaxis_name1", "value"),Input('yaxis_name1','value'))
def my_graph1(xaxis_name1,yaxis_name1):
    #fig = px.scatter(data, x=xaxis_name1, y=yaxis_name1)
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig





@app.callback(Output("graph2", "figure"),Input("xaxis_name2", "value"),Input('yaxis_name2','value'))
def my_graph2(xaxis_name2,yaxis_name2):   
    #fig = px.pie(data, values=yaxis_name2, names=xaxis_name2)
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig






@app.callback(Output("graph3", "figure"),Input("xaxis_name2", "value"),Input('yaxis_name2','value'))
def my_grapg3(xaxis_name2,yaxis_name2):
    #df=pd.pivot_table(data,values=[yaxis_name2],columns=[xaxis_name2],aggfunc=np.sum)
    df=pd.DataFrame({'name':df.columns.tolist(),'values':list(df.values[0])})
    fig = px.bar(df,x='name', y='values',text='values')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig




@app.callback(Output("graph4", "figure"),Input("xaxis_name1", "value"))
def my_grapg4(xaxis_name1):
    #fig = px.treemap(data, path=['cut','color','clarity'], values=xaxis_name1,color='cut')
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig 




@app.callback(Output('pred','children'),
             Input("n1","value"),Input("n2","value"),Input("n3","value"),Input("n4","value"),
             Input("n6","value"),Input("n7","value"),Input("n8","value"),Input("n9","value"))
def my_prediction(n1,n2,n3,n4,n5,n6,n7,n8,n9):
    #df=pd.DataFrame({'carat':[n1],'cut':[n2],'color':[n3],'clarity':[n4],'depth':[n5],'table':[n6],'x':[n7],'y':[n8],'z':[n9]})
    result=predict_model(saved_model,df)
    price_dm=round(result['Label'][0],2)
    return "Diamond price is near to {} ".format(price_dm)





@app.callback(Output("graph5", "figure"),Input('bx1','value'),Input('bx2','value'),Input('bx3','value'))
def my_graph5(bx1,bx2,bx3):
    #fig = px.box(data, x=bx1, y=bx2,color=bx3)
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig




@app.callback(Output("graph6", "figure"),Input('col_list','value'))
def my_graph6(list_):
    #cm=data[list_].corr()
    fig = px.imshow(cm)
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig




@app.callback(Output("graph7", "figure"),Input('hist_var','value'))
def my_graph7(hist_var):
    #fig = px.histogram(data, x=hist_var, histnorm='probability density')
    fig.update_layout(plot_bgcolor='black',paper_bgcolor='black',font_color='white')
    return fig  




app.run_server()