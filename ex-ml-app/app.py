import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State
import dash_auth
import dash_bootstrap_components as dbc
from retroviz import RetroScore, run_retro_score
from descriptions import text, titles, source
import numpy as np
import pandas as pd

import shap
import pickle
from pdpbox import pdp

#%% Unpacking
with open('model_set.pkl', 'rb') as f:
            model_set = pickle.load(f)

keys = list(model_set.keys())
for i in keys:
    locals()[i] = model_set[i]


with open('trust_set.pkl', 'rb') as f:
            trust_set = pickle.load(f)

keys = list(trust_set.keys())
for i in keys:
    locals()[i] = trust_set[i]


with open('Expl_set.pkl', 'rb') as f:
            Expl_set = pickle.load(f)

keys = list(Expl_set.keys())
for i in keys:
    locals()[i] = Expl_set[i]

del(model_set,Expl_set,trust_set,keys,i,f)

state_list = []
for i in list(X_valid.columns):
    state_list.append(State(i, "value"))


#%% Styles
SIDEBAR_STYLE = {
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "border-right": "solid",
    "border-width": "2px",
    "border-radius":"12px",
    "border-color": "#e9ecef",
    "position": "absolute"
}


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",


}

OFFCANV_STYLE = {
    "top": "0.45rem",
    "left": "90rem",
    "position": "absolute"
    }
#%% Offcanvas setting
Offcanvas = html.Div([
    dbc.Button(
        "?",
        id="offcanvas_button",
        n_clicks = 0, n_clicks_timestamp = '0' ,outline=True, color="warning", className="me-1", active=True),

    html.Div(id='offcanvas')], style =  OFFCANV_STYLE)

#%% login - password establishing
USERNAME_PASSWORD_PAIRS = {
    'test': 'test_pass'
}
#%% Define the app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Explainable ML Dashboard")

auth = dash_auth.BasicAuth(
    app,
    USERNAME_PASSWORD_PAIRS
)

app.layout = html.Div([
        dcc.Tabs(
        id="tabs-with-classes",
        value='tab-1',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Explainability',
                value='tab-1',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Monitoring',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='What if...?',
                value='tab-3',
                className='custom-tab',
                selected_className='custom-tab--selected'
            )
        ],style={
            "margin-left": "15rem"}),
    html.Div(id='tabs-content-classes'),
    html.Div([

])])



@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
        html.H4('Explainable ML for SON', className="display-10"),
        html.Hr(),
        html.P("Select method:"),
        dcc.Dropdown(
            id="dropdown",
            options=['Permutation Importance', 'Partial Dependence Plots', 'SHAP'],
            value='Permutation Importance',
            clearable=False,
        ),
        html.Hr()], style=SIDEBAR_STYLE), html.Div(children="loading...", id="output")

    elif tab == 'tab-2':
        return html.Div([
        html.H4('Monitoring for SON', className="display-10"),
        html.Hr(),

        html.P("Select method:"),
        dcc.Dropdown(
            id="dropdown2",
            options=['KPI', 'Trust', 'Metric distributions'],
            clearable=False,
        ),
        ], style=SIDEBAR_STYLE), html.Div(children="loading...", id="output_monitoring"), html.Div([dcc.Loading(
            id="loading_monitoring",
            children=[html.Div([html.Div(id="output_monitoring")])],
            type="circle",
            fullscreen=False )])


    elif tab == 'tab-3':

        return  html.Div([
                html.Div([
                html.H4('What if playground', className="display-10"),
                html.Hr(),
                html.Div([
                html.P("Select method:"),
                dcc.Dropdown(
                    id="drop_whatif",
                    options=['Single Prediction', 'Range'],
                    clearable=False,)]),
                html.Div(children="", id="whatif_method")],style=SIDEBAR_STYLE),
                dcc.Store(id='offcanvas_type', data='whatif'),
                Offcanvas
                ])


@app.callback(
    Output("output", "children"),
    Input("dropdown", "value"))

def disp_context(sr):
    if sr == 'Permutation Importance':
        weights = Permutation_Importance
        fig = px.bar(weights, x=weights.weight, y=weights.feature)
        fig.update_traces(hovertemplate=None)
        fig.update_layout(template="plotly_white",hovermode="x",title = sr, yaxis=dict(autorange="reversed"))

        return html.Div([html.Div(children=
                      dcc.Graph(figure = fig), style=CONTENT_STYLE),
              dcc.Store(id='offcanvas_type', data='perm_importance'),
              Offcanvas
              ])

    elif sr == 'SHAP':
        return  html.Div([
                html.P("Select graph type:"),
                html.Div([

                dbc.RadioItems(options=[
                    {"label": "Summary plot", "value": "Summary plot"},
                    {"label": "Single prediction", "value": "Single prediction"}],
                    value =  'Summary plot',
                    id='SHAP-type',
                    inline=False)], style={
                        "position": "fixed",
                        "top": 205,
                        "left": 0,
                        "bottom": 0,
                        "width": "16rem",
                        "padding": "2rem 1rem",

                    }),
                        html.Div(children="",id="SHAP_graph"),
                        html.Div([dbc.Button("Display",
                                         id="display_SHAP",
                                         color="primary",
                                         style={"margin": "5px"},
                                         n_clicks_timestamp='0',
                                         type = "reset"

                                         )],style ={ "position": "fixed",
                                          "top": 300,
                                          "left": 20,
                                          "bottom": 0,

                                          }),
                    html.Div([dcc.Loading(
                        id="loading_shap",
                        children=[html.Div([html.Div(id="SHAP_graph")])],
                        type="circle",
                        fullscreen=False )],style =CONTENT_STYLE),
                    dcc.Store(id='offcanvas_type', data='shap'),
                    Offcanvas
                    ])





    elif sr == 'Partial Dependence Plots':
        return html.Div([
                html.Div([
                html.P("Select parameters to observe:"),
                html.Div([
                dcc.Dropdown(
                    options=list(X_valid.columns),
                    clearable=False,
                    value ='PCF',
                    id='xaxis-type'
                    )], style={'width': '30%', 'display': 'inline-block', 'margin-right' : '2rem'}),

                html.Div([
                dcc.Dropdown(
                    options=list(X_valid.columns),
                    clearable=False,
                    value ='PCF',
                    id='yaxis-type'
                    )], style={'width': '30%', 'display': 'inline-block','margin-left' : '2rem'}),
                html.Div([dbc.Button("Display",
                                     id="display_pdp",
                                     color="primary",
                                     style={
                                         'position': 'fixed',
                                         'top': '120px',
                                         'left': '1250px'},
                                     n_clicks_timestamp='0',
                                     type = "reset"

                                     )]),
                html.P("(Note: if both parameters match, the isolate graph will be displayed instead of interact)"),
                html.Hr(),
                html.Div(children="",id="pdp_graph"),
                html.Div([dcc.Loading(
                    id="loading",
                    children=[html.Div([html.Div(id="pdp_graph")])],
                    type="circle",
                    fullscreen=True )]),
                ], style =CONTENT_STYLE),
                dcc.Store(id='offcanvas_type', data='pdp'),
                Offcanvas
                ])



@app.callback(
    Output("pdp_graph", "children"),
    Input("display_pdp", "n_clicks_timestamp"),
    State("xaxis-type", "value"),
    State("yaxis-type", "value"))

def disp_pdp_graph(display_pdp,x_ax,y_ax):


    if display_pdp != '0':

        if x_ax == y_ax:
            pdp_isolate = PDP_isolate[x_ax]

            fig = go.Figure(data=go.Scatter(x=pdp_isolate.grid , y= (pdp_isolate.dist - min(pdp_isolate.dist))))
            fig.update_traces(hoverinfo='skip')
            fig.update_layout(template="plotly_white",title='PDP isolate for '+str(x_ax), autosize=False,
                      width=1100, height=500,
                      margin=dict(l=65, r=50, b=65, t=90),
                      xaxis_title=x_ax,
                      yaxis_title="Changes in GKPI")
            return html.Div([dcc.Graph(figure=fig),
                             # dcc.Store(id='offcanvas_type', data='isolate')
                            ])

        else:

            for k in list(PDP_interact.keys()):
                if x_ax in k and y_ax in k:
                    pdp_interact = PDP_interact[k]

            fig = go.Figure()
            temp =  pdp_interact.pdp.iloc[:,2]
            z_grid = temp.values.reshape(10,10)
            fig = go.Figure(data=[go.Surface(x = pdp_interact.pdp[pdp_interact.features[1]].unique(),
                                             y = pdp_interact.pdp[pdp_interact.features[0]].unique(),
                                             z=z_grid)])
            fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True),showscale = False,hoverinfo='skip')
            fig.update_layout(template="plotly_white", autosize=False,
                      scene = dict(
                          xaxis_title=pdp_interact.features[1],
                          yaxis_title=pdp_interact.features[0],
                          zaxis_title="GKPI"),
                      scene_camera_eye=dict(x=1.87, y=-0.88, z=0.64),
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90),
                      )

            fig1 = go.Figure()
            fig1  = go.Figure(data=[go.Contour(x = pdp_interact.pdp[pdp_interact.features[1]].unique(),
                                             y = pdp_interact.pdp[pdp_interact.features[0]].unique(),
                                             z=z_grid,line_width=1,
                                    line_color = "white",contours=dict(
                                        coloring ='fill',showlabels = True,
                                        labelfont = dict(size = 12,color = 'white')),
                                        colorbar = dict(separatethousands = False,xanchor = "left", title='GKPI'),
                                        hoverinfo='skip'
                                    )])
            fig1.update_layout(template="plotly_white", autosize=False,
                      xaxis_title=pdp_interact.features[1],
                      yaxis_title=pdp_interact.features[0],
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90),
                      )


            return html.Div([html.Div([dcc.Graph(figure=fig)],style = {
"display": "inline-block"}
                            ),
                              html.Div([dcc.Graph(figure=fig1)],
                                        style = {
 "display": "inline-block"}
                             ),
                                # dcc.Store(id='offcanvas_type', data='interact')
                               ])

@app.callback(
    Output("SHAP_graph", "children"),
    Input("display_SHAP", "n_clicks_timestamp"),
    State("SHAP-type", "value"))

def disp_SHAP(display_SHAP,SHAP_type):


    if display_SHAP != "0":

        if SHAP_type == 'Summary plot':

           # shap_values = SHAP["shap_values"]
           # features = X_valid.iloc[0:5000]

           # scaler = MinMaxScaler()
           # for i in list(features.columns):
           #     features[i] = np.log(scaler.fit_transform(np.array([features[i]]).T))

           # feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
           # feature_order = feature_order[-min(7, len(feature_order)):]

           # fig = go.Figure()
           # for pos, i in enumerate(feature_order):
           #         shaps = shap_values[:, i]
           #         values = features.values[:,i]
           #         inds = np.arange(len(shaps))
           #         np.random.shuffle(inds)
           #         if values is not None:
           #             values = values[inds]
           #         shaps = shaps[inds]

           #         N = len(shaps)

           #         nbins = 100
           #         row_height = 0.4
           #         quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
           #         inds = np.argsort(quant + np.random.randn(N) * 1e-6)
           #         layer = 0
           #         last_bin = -1
           #         ys = np.zeros(N)
           #         for ind in inds:
           #             if quant[ind] != last_bin:
           #                 layer = 0
           #             ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
           #             layer += 1
           #             last_bin = quant[ind]
           #         ys *= 0.9 * (row_height / np.max(ys + 1))
           #         nan_mask = np.isnan(values)

           #         fig.add_trace(go.Scatter(name= features.columns[i],
           #             x = shaps[np.invert(nan_mask)],
           #                                  y = pos + ys[np.invert(nan_mask)],
           #                               mode='markers',
           #                               hovertemplate = features.columns[i]

           #                               ))

           # fig.update_traces(marker=dict(
           #     size=5,
           #     color=values,
           #     colorscale='bluered',
           #     showscale=True,
           #     colorbar_title = "Values"),
           #     showlegend = False,

           # )
           # fig.update_layout(template="plotly_white", width=1100, height=550, xaxis_title="SHAP value")
           # return html.Div([dcc.Graph(figure=fig)])

           return html.Div([html.Img(src='/assets/shap.png',style = CONTENT_STYLE),
                            dcc.Store(id='offcanvas_type', data='shap_sum')
                            ])
        else:
            data = X_valid.iloc[1000:1001]
            output = y_valid.iloc[1000:1001]
            shap_values = SHAP["shap_values_single"]
            prediction = SHAP["prediction"]
            pred_err = abs(round(output.values[0] - prediction[0],3))
            if pred_err > 0.1:
                err_color = "danger"
            elif pred_err <0.02:
                err_color = "success"
            else:
                err_color = "warning"

            x_vals = ["Base value ","Predicted value"]
            y_vals = [SHAP["explainer"].expected_value,0]
            measure_type = ["total","relative"]
            bars_text = [str(round(SHAP["explainer"].expected_value,3)),"Output = " + str(round(prediction[0],3))]
            features = list(data.columns)
            for k in range(0,len(data.columns)):
                x_vals.insert(k+1,features[k]+" = " + str(data.iat[0,k]))
                y_vals.insert(k+1,shap_values[0,k])
                measure_type.insert(k+1,'relative')
                bars_text.insert(k+1, str(round(shap_values[0,k],3)))

            fig = go.Figure(go.Waterfall(
                  orientation = "v",
                  measure = measure_type,
                  x = x_vals,
                  textposition = "outside",
                  text = bars_text,
                  y = y_vals,
                 connector = {"line":{"color":"rgb(63, 63, 63)"}},
                 decreasing = {"marker":{"color":'#2E91E5'}},
                 increasing = {"marker":{"color":'#ff0d57'}},
                 showlegend = False))
            fig.update_layout(template="plotly_white",yaxis_title = "Deviation of base value", title="SHAP single prediction", width=1150, height=500)
            fig.add_trace(go.Scatter(x = ["Predicted value"], y = [round(output.values[0],3)],
                                     mode="markers+text",
                                     text = ["True Value = " +  str(round(output.values[0],3))],
                                     showlegend = False,
                                     hoverinfo = 'none',
                                      textposition = "top center"
            ))

            return html.Div([html.Div([
                    dbc.Alert("Prediction error is "+str(pred_err),
                          is_open=True, color=err_color)]),
                    html.Div([dcc.Graph(figure=fig)]),
                    dcc.Store(id='offcanvas_type', data='shap_single')])

@app.callback(
    Output("KPI_graph", "children"),
    Input("cell_var", "value"))

def disp_KPIplot(cell_id):

    if cell_id:

        select_cell = monitoring.loc[(monitoring['Cell ID'] == cell_id)]
        KPI = select_cell.KPI[0:90].values
        time = select_cell.sim_time[0:90].values
        Interference  = select_cell['Average Interference Created'][0:90].values
        RSRP =select_cell['cell RSRP'][0:90].values

        fig = make_subplots(rows=3, cols=1,
                                subplot_titles=("Cell RSRP", "Average Interference Created", "GKPI"))
        cell_RSRP = go.Scatter(x = time,
                                  y = RSRP,mode='lines+markers',
                                  showlegend=False)
        averageInterferenceCreated = go.Scatter(x = time,
                                  y = Interference,mode='lines+markers',
                                  showlegend=False)
        KPI = go.Scatter(x = time,
                                  y = KPI,mode='lines+markers',
                                  showlegend=False)

        fig.append_trace(cell_RSRP, 1, 1)
        fig.append_trace(averageInterferenceCreated, 2, 1)
        fig.append_trace(KPI, 3, 1)
        fig.update_layout(width=1200, height = 1000, showlegend=False,
                              title_text="Cell ID " + str(cell_id) + " KPI to time sampling")
        fig.update_xaxes(title_text="Time,ms")
        fig.update_yaxes(title_text="KPI")
        return html.Div([dcc.Graph(figure=fig)])


@app.callback(
    Output("whatif_method", "children"),
    Input("drop_whatif", "value"))

def what_if(method):

    if method == "Single Prediction":
        disp = []
        disp.append( html.Div([
        html.P("Select value of parameters:")],style = {"margin-top": "1rem"}))
        for var in list(X_valid.columns):
            min_val = round(min(X_valid[var]),2)
            max_val = round(max(X_valid[var]),2)
            step = round((max_val - min_val)/100,2)

            disp.append(html.Div([html.H6(var),
                 dcc.Slider(min_val, max_val,step,marks=None, id = var, value = min_val,
                            tooltip={"placement": "bottom", "always_visible": True})],
                 style = {"margin-top": "0.8rem"}),)
        disp.append(html.Div([html.Div([
                     dbc.RadioItems(options=[
                          {"label": "SHAP", "value": "shap"},
                          {"label": "Trust RETRO-VIZ", "value": "trust"}],
                          value =  'SHAP',
                          id='wtahif_type',
                          inline=False)],
                            style={"margin-top": "1rem"}
                          ),

            html.Div([dbc.Button("Display",
                                  id="disp_whatif",
                                  color="primary",
                                  style={"margin-top": "1rem"},
                                  n_clicks_timestamp='0',
                                  type = "reset")]),

            html.Div(children="",id="out_whatif"),
            dcc.Loading(id="loading", children=[html.Div([html.Div(id="pdp_graph")])],
                    type="circle",
                    fullscreen=False)]))

        return disp


    elif method == "Range":
        return html.Div([
            html.Div([
            html.P("Select fixed parameter:"),
            dcc.Dropdown(
                options=list(X_valid.columns),
                clearable=False,
                id='Range-type',
                value = "PCF")], ),
            html.Div(children="",id="output-range")])

@app.callback(
    Output("out_whatif", "children"),
    Input("disp_whatif", "n_clicks_timestamp"),
    State("wtahif_type", "value"),
    state_list)

def WhatIf(display_whatif,plot_type,*states):


    if display_whatif != '0':

        data = pd.DataFrame(np.reshape(np.array(states),(1,len(states))))


        if plot_type == "shap":

            prediction = model.predict(data)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)

            x_vals = ["Base value ","Predicted value"]
            y_vals = [explainer.expected_value,0]
            measure_type = ["total","relative"]
            bars_text = [str(round(explainer.expected_value,3)),"Output = " + str(round(prediction[0],3))]
            features = list(X_valid.columns)
            for k in range(0,len(X_valid.columns)):
                x_vals.insert(k+1,features[k]+" = " + str(data.iat[0,k]))
                y_vals.insert(k+1,shap_values[0,k])
                measure_type.insert(k+1,'relative')
                bars_text.insert(k+1, str(round(shap_values[0,k],3)))

            fig = go.Figure(go.Waterfall(
                  orientation = "v",
                  measure = measure_type,
                  x = x_vals,
                  textposition = "outside",
                  text = bars_text,
                  y = y_vals,
                 connector = {"line":{"color":"rgb(63, 63, 63)"}},
                 decreasing = {"marker":{"color":'#2E91E5'}},
                 increasing = {"marker":{"color":'#ff0d57'}},
                 showlegend = False))
            fig.update_layout(template="plotly_white", width=1150, height=500)

            return html.Div([dcc.Graph(figure=fig),
                             dcc.Store(id='offcanvas_type', data='whatif-shap')
                             ],style = {"top":"3rem",
                                                            "margin-left": "18rem",
                                                            "margin-right": "2rem",
                                                            "padding": "1rem",
                                                            "position": "absolute"})

        elif plot_type == "trust":

            train_pred = model.predict(X_train)

            X_val = data.values[0, :].reshape(1,-1)
            pred = model.predict(X_val)

            rs = RetroScore(k=5)
            retro_score, retro_score_unn, nbs_x, nbs_y = run_retro_score(rs, X_train.values, y_train.values, X_val, pred, train_pred)

            cols = list(X_valid.columns)

            cells = dict(values=[])
            for k in range(0,len(cols)):
                cells["values"].append(nbs_x[0][:,k])
            cells["values"].append(nbs_y[0].round(3))
            cols.insert(len(cols),"KPI")
            fig = go.Figure(data=[go.Table(header=dict(values = cols),
                              cells=cells)
                                  ])
            fig.update_layout(template="plotly_white", yaxis_title = "Deviation of base value", width=1200, height=500)

            if retro_score < 0.5:
                color = "danger"
            elif retro_score > 0.85:
                color = "success"
            else:
                color = "warning"

            if retro_score == 0:
                return  html.Div([dcc.Graph(figure=fig),
                                  dcc.Store(id='offcanvas_type', data='whatif-trust'),
                                  dbc.Alert("Trust score is "+str(round(retro_score[0],3)),
                                        is_open=True, color=color)],style = {"top":"3rem",
                                                                "margin-left": "15rem",
                                                                "margin-right": "2rem",
                                                                "padding": "1rem",
                                                                "position": "absolute"})
            return  html.Div([dcc.Graph(figure=fig),
                              dcc.Store(id='offcanvas_type', data='whatif-trust'),
                              dbc.Alert("Trust score is "+str(round(retro_score[0][0],3)),
                                    is_open=True, color=color)],style = {"top":"3rem",
                                                            "margin-left": "15rem",
                                                            "margin-right": "2rem",
                                                            "padding": "1rem",
                                                            "position": "absolute"})

@app.callback(
    Output("output_monitoring", "children"),
    Input("dropdown2", "value"))

def disp_monitoring(mt):
    if mt == 'KPI':

        return html.Div([html.Div([
                dcc.Dropdown(
                    id="cell_var",
                    searchable=True,
                    options=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                    clearable=False,
                    placeholder= "Select Cell ID:"
                    ),
          ],style = {
              "width": "14rem",
              "margin-top":"8rem",
              "margin-left":"1rem",
              "position": "absolute"}),
        html.Div(children="",id="KPI_graph",style =CONTENT_STYLE)])


    elif mt == 'Trust':
        y_max = pd.Series.to_list(score.y_max)
        y_min = pd.Series.to_list(score.y_min)
        x = pd.Series.to_list(score.bin)
        fig = go.Figure([
            go.Scatter(
                x=x,
                y=score.score,
                line=dict(color='rgb(0,100,80)'),
                mode='lines+markers',
                showlegend=False,
            ),
            go.Scatter(
                x=x+x[::-1],
                y=y_min+y_max[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        ])

        fig.update_layout(template="plotly_white",title="Maximum error vs. average RETRO-score",
                  autosize=False,
                  width=1100, height=500,
                  margin=dict(l=65, r=50, b=65, t=90),
                  xaxis_title="Maximum error",
                  yaxis_title="Average trust score")
        return html.Div(children=[
                      dcc.Graph(figure = fig)], style=CONTENT_STYLE)


    elif mt == 'Metric distributions':
        fig = make_subplots(rows=1, cols=3,
                                subplot_titles=("Cell RSRP", "Average Interference Created", "GKPI"))
        cell_RSRP = go.Histogram( x = monitoring['cell RSRP'],
                                    histnorm = "probability",
                                   marker_color='#EB89B5')
        averageInterferenceCreated = go.Histogram( x = monitoring['Average Interference Created'],
                                   histnorm = "probability",
                                  marker_color='#62639b')
        KPI = go.Histogram( x = monitoring.KPI,
                            histnorm = "probability",
                           marker_color='#330C73')

        fig.append_trace(cell_RSRP, 1, 1)
        fig.append_trace(averageInterferenceCreated, 1, 2)
        fig.append_trace(KPI, 1, 3)
        fig.update_layout(width=1200, showlegend=False,
                              title_text="Distributions of the metrics")
        fig.update_xaxes(title_text="dBm", row=1, col=1)
        fig.update_xaxes(title_text="dBm", row=1, col=2)
        fig.update_xaxes(title_text="normilized", row=1, col=3)
        return html.Div(children=[dcc.Graph(figure = fig)], style=CONTENT_STYLE)


@app.callback(
    Output("output-range", "children"),
    Input("Range-type", "value"))

def dispRange(rtype):

    min_val = round(min(X_valid[rtype]),2)
    max_val = round(max(X_valid[rtype]),2)
    step = round((max_val - min_val)/100,2)

    range_slider = dcc.RangeSlider(min_val, max_val, step, allowCross=False, id="range_state", value = [min_val, max_val],
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks=None)
    features_use = list(X_valid.columns)[:]
    features_use.remove(rtype)

    return html.Div([html.Div(children=[range_slider], style = {"margin-top":"1rem","margin-bottom":"1rem"}),
                     html.Div([
                             html.P("Select parameters to plot:"),
                             html.Div([
                             dcc.Dropdown(
                                 options=features_use,
                                 clearable=False,
                                 id='xaxis-range'
                                 )],style = {'margin-top':'0.8rem'}),
                             html.Div([
                             dcc.Dropdown(
                                 options=features_use,
                                 clearable=False,
                                 id='yaxis-range'
                                 )],style = {'margin-top':'0.8rem'})], style = {'margin-top':'0.8rem'}),
                     html.Div([dbc.Button("Display",
                                 id="range_slider",
                                 color="primary",
                                 style={"margin-left": "7px",
                                        "margin-top": "10px"},
                                 n_clicks_timestamp='0',
                                 type = "reset")]),
                     html.Div(children="",id="PDP_range",style =CONTENT_STYLE)]),html.Div([dcc.Loading(
                         id="loading",
                         children=[html.Div([html.Div(id="PDP_range")])],
                         type="circle",
                         fullscreen=True),
                         dcc.Store(id='offcanvas_type', data='whatif-range')])

@app.callback(
    Output("PDP_range", "children"),
    Input("range_slider", "n_clicks_timestamp"),
    State("range_state", "value"),
    State("Range-type", "value"),
    State("xaxis-range", "value"),
    State("yaxis-range", "value"))


def PDPrange(slider,state,Rtype,x_ax,y_ax):

    if slider != '0':

        if x_ax == y_ax:

            return  html.Div([dbc.Alert("The same feature is selected. The parameters for the ploting should be different!",color="danger")])
        elif state is None:
            return  html.Div([dbc.Alert("The value of the fixed feature should not be default")])
        else:


            X_valid_range = X_valid[X_valid[Rtype].between(state[0], state[1], inclusive='right')]
            dependence  =  pdp.pdp_interact(model=model, dataset=X_valid_range, model_features=list(X_valid.columns), features=[x_ax,y_ax])
            fig = go.Figure()
            temp = dependence.pdp.iloc[:,2]
            z_grid = temp.values.reshape(10,10)

            fig = go.Figure()
            fig  = go.Figure(data=[go.Contour(x = dependence.pdp[dependence.features[1]].unique(),
                                                  y = dependence.pdp[dependence.features[0]].unique(),
                                                  z=z_grid,line_width=1,
                                         line_color = "white",contours=dict(
                                             coloring ='fill',showlabels = True,
                                             labelfont = dict(size = 12,color = 'white')),
                                             colorbar = dict(separatethousands = False,xanchor = "left",  title='GKPI')
                                         )])
            fig.update_layout(template="plotly_white", autosize=False,
                              xaxis_title=y_ax,
                              yaxis_title=x_ax,
                           width=550, height=550,
                           title="Fixed feature - "+Rtype+ " in range [" +str(state[0])+":"+str(state[1])+"]" ,
                           margin=dict(l=65, r=50, b=65, t=90),
                           ),

            for k in list(PDP_interact.keys()):
                if x_ax in k and y_ax in k:
                    pdp_interact = PDP_interact[k]

            fig1 = go.Figure()
            temp1 =  pdp_interact.pdp.iloc[:,2]
            z_grid1 = temp1.values.reshape(10,10)
            fig1 = go.Figure()
            fig1  = go.Figure(data=[go.Contour(x = pdp_interact.pdp[pdp_interact.features[1]].unique(),
                                                  y = pdp_interact.pdp[pdp_interact.features[0]].unique(),
                                                  z=z_grid1,line_width=1,
                                         line_color = "white",contours=dict(
                                             coloring ='fill',showlabels = True,
                                             labelfont = dict(size = 12,color = 'white')),
                                             colorbar = dict(separatethousands = False,xanchor = "left",  title='GKPI')
                                         )])
            fig1.update_layout(template="plotly_white", autosize=False,
                              xaxis_title=y_ax,
                              yaxis_title=x_ax,
                           width=550, height=550,
                           title="Initial values" ,
                           margin=dict(l=65, r=50, b=65, t=90),
                           ),

            return html.Div([html.Div([dcc.Graph(figure=fig)], style = {
                "top": 55,
                "left": 300,
                "position": "absolute"}),
                html.Div([dcc.Graph(figure=fig1)], style = {
                    "top": 55,
                    "left": 900,
                    "position": "absolute"})
                ])


@app.callback(
    Output("offcanvas", "children"),
    Input("offcanvas_button", "n_clicks_timestamp"),
    State("offcanvas_type", "data"),
    )
def offcanvas_set(click, offc_type):
    if click != "0":

        return html.Div([
                      dbc.Offcanvas(
                           html.P([text[offc_type],html.Br(),'',html.Br(),source[offc_type]]),
                           id="offcanvas_set",
                           scrollable=True,
                           title=titles[offc_type],
                           is_open=False,
                       ),
                      ],style = OFFCANV_STYLE)

@app.callback(
    Output("offcanvas_set", "is_open"),
    Input("offcanvas_button", "n_clicks"),
    State("offcanvas_set", "is_open"),
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open
# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)

    # (host="10.0.0.143",
    #                port="8050",
    #                proxy= None,
    #                debug=False)
