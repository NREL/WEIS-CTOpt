'''This is the page for visualize the optimization results'''

'''
For understanding:
Callback function - Add controls to build the interaction. Automatically run this function whenever changes detected from either Input or State. Update the output.
'''

# Import Packages
import dash_bootstrap_components as dbc
from dash import html, register_page, callback, Input, Output, dcc, State, no_update, ctx
import numpy as np
import os
import time
import pandas as pd
from PIL import Image
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from dash.exceptions import PreventUpdate
from weis.visualization.utils import read_cm, load_OMsql, parse_contents, find_file_path_from_tree, find_iterations, empty_figure, toggle, read_per_iteration, get_timeseries_data, update_yaml, generate_raft_img

register_page(
    __name__,
    name='Optimize',
    top_nav=True,
    path='/optimize'
)

pio.templates.default = "ggplot2"

#################################################################
#   Read Optimization related variables/data from yaml file
#################################################################

def read_opt_vars_per_type(input_dict):
    opt_options = {}
    opt_type_reference = {1: 'RAFT', 3: 'OpenFAST'}     # TODO: Expand other types of optimizations

    opt_options['root_file_path'] = '/'.join(input_dict['userOptions']['output_folder'].split('/')[:-1])           # Remove the last output folder name for future path join
    opt_options['log_file_path'] = os.path.join(opt_options['root_file_path'], '/'.join(k for k in next(find_file_path_from_tree(input_dict['outputDirStructure'], input_dict['userOptions']['sql_recorder_file'])) if k not in ['dirs', 'files']))
    
    var_opt = input_dict['userPreferences']['optimization']
    opt_options['conv_y'] = var_opt['convergence']['channels']

    opt_options['opt_type'] = opt_type_reference[input_dict['userOptions']['optimization']['type']]

    if opt_options['opt_type'] == 'RAFT':
        opt_options['raft_design_dir'] = '/'.join(opt_options['log_file_path'].split('/')[:-1]) + '/raft_designs'
    
    elif opt_options['opt_type'] == 'OpenFAST':
        stats_paths = []
        for paths in find_file_path_from_tree(input_dict['outputDirStructure'], 'summary_stats.p'):
            stats_paths.append(os.path.join(opt_options['root_file_path'], '/'.join(k for k in paths if k not in ['dirs', 'files'])))
        
        iterations = []
        for iteration_nums in find_iterations(input_dict['outputDirStructure']):
            iterations.append(iteration_nums)

        opt_options['stats_path'] = stats_paths
        opt_options['iterations'] = iterations
        # Case Matrix could be named either 'case_matrix.yaml' or 'case_matrix_combined.yaml'
        try:
            opt_options['case_matrix'] = os.path.join(opt_options['root_file_path'], '/'.join(k for k in next(find_file_path_from_tree(input_dict['outputDirStructure'], 'case_matrix.yaml')) if k not in ['dirs', 'files']))
        except:
            opt_options['case_matrix'] = os.path.join(opt_options['root_file_path'], '/'.join(k for k in next(find_file_path_from_tree(input_dict['outputDirStructure'], 'case_matrix_combined.yaml')) if k not in ['dirs', 'files']))
        opt_options['x_stat'] = var_opt['dlc']['xaxis_stat']
        opt_options['y_stat'] = var_opt['dlc']['yaxis_stat']
        opt_options['x'] = var_opt['dlc']['xaxis']
        opt_options['y'] = var_opt['dlc']['yaxis']
        opt_options['y_time'] = var_opt['timeseries']['channels']
        

    return opt_options


@callback(Output('var-opt', 'data'),
          Input('input-dict', 'data'))
def read_variables(input_dict):
    if input_dict is None or input_dict == {}:
        raise PreventUpdate
    
    if input_dict['userOptions']['optimization']['status'] == True:
        opt_options = read_opt_vars_per_type(input_dict)
    
    else:
        # TODO: Handle if it doesn't have optimization
        opt_options = 'No Optimization..'
    
    print("Parse variables from optimization..\n", opt_options)

    return opt_options


###############################################
#   Data Preprocessing Status
###############################################

@callback(# Check loading status
          Output('load-warning', 'is_open'),
          Output('load-warning', 'header'),
          Output('load-warning', 'icon'),
          Output('load-warning-text', 'children'),
          # Data Retrieval
          Output('log-data', 'data'),
          Output('df', 'data'),
          Output('prep-data', 'data'),
          # Either triggered by any 'load' button
          Input('load-sql', 'n_clicks'),
          Input('load-raft', 'n_clicks'),
          # Form
          State('log-sql-path', 'value'),
          State('raft-dir-path', 'value'),
          State('log-data', 'data'),
          State('prep-data', 'data'))
def preprocess_data(nClickSQL, nClickRAFT, log_sql_path, raft_dir_path, log_data, prep_data):
    if nClickSQL not in [None, 0] and ctx.triggered_id == 'load-sql':
        prep_data['log_flag'] = True
        return load_sql(log_sql_path) + (prep_data,)
    
    elif nClickRAFT not in [None, 0] and ctx.triggered_id == 'load-raft':
        prep_data['raft_flag'] = True
        return load_raft(raft_dir_path, log_data) + (no_update, no_update, prep_data,)
    
    else:
        return False, "Warning", "danger", "", no_update, no_update, no_update
        


def load_sql(log_sql_path):
    try:
        time.sleep(.1)
        log_data = load_OMsql(log_sql_path)
        df = parse_contents(log_data)
        # df.to_csv('log_opt.csv', index=False)     # For checking..

    except Exception as e:
        return True, "Warning", "danger", f"An error occurred: {e}", {}, {}
    
    return True, "Success", "success", "Loaded Successfully", log_data, df.to_dict('records')


def load_raft(raft_dir_path, log_data):
    try:
        time.sleep(.1)
        # Generate RAFT Output Files
        plot_dir = os.path.join(raft_dir_path,'..','raft_plots')
        if not os.path.isdir(plot_dir):
            generate_raft_img(raft_dir_path, plot_dir, log_data)

    except Exception as e:
        return True, "Warning", "danger", f"An error occurred: {e}"
    
    return True, "Success", "success", "Loaded Successfully"


###############################################
#   Basic layout definition
###############################################

@callback(Output('data-layout', 'children'),
          Input('var-opt', 'data'))
def define_preprocess_layout(opt_options):

    opt_preprocess_layout = dbc.Card([
                                dbc.CardHeader('Data Preprocessing'),
                                dbc.CardBody([
                                    dbc.Row([
                                        # Prep for SQL data
                                        dbc.Label('SQL Data', width=1),
                                        dbc.Col(dbc.Input(id='log-sql-path', type='text', placeholder=opt_options['log_file_path'], value=opt_options['log_file_path'], disabled=True), width=6), 
                                        dbc.Col(dbc.Button('Load', id='load-sql', n_clicks=0, color='primary'), width='auto'),
                                    ], className="mb-3"),
                                    dbc.Row([
                                        # Prep for RAFT output files if opt type if RAFT
                                        dbc.Label('RAFT Designs', width=1),
                                        dbc.Col(dbc.Input(id='raft-dir-path', type='text', disabled=True), width=6),
                                        dbc.Col(dbc.Button('Load', id='load-raft', n_clicks=0, color='primary'), width='auto'),
                                    ], className="mb-3", style={} if opt_options['opt_type'] == 'RAFT' else {'display': 'none'})
                                ])
                            ], className='card')
    
    return opt_preprocess_layout

@callback(Output('raft-dir-path', 'placeholder'),
          Output('raft-dir-path', 'value'),
          Input('var-opt', 'data'))
def complete_raft_sublayout(opt_options):
    if opt_options['opt_type'] != 'RAFT':
        raise PreventUpdate
    
    raft_design_dir = opt_options['raft_design_dir']
    return raft_design_dir, raft_design_dir


def define_convergence_layout():
    # Layout for visualizing Conv-trend data
    convergence_layout = dbc.Collapse(
                            dbc.Card([
                                dbc.CardHeader('Convergence Trend from Optimization'),
                                dbc.CardBody([
                                    dcc.Loading(
                                        html.Div([
                                            html.H6('Y-channel:'),
                                            dcc.Dropdown(id='conv-signaly', multi=True, style={'color': 'black'}),      # Get 'conv-signaly' channels from user. Related function: update_graphs()
                                            dcc.Graph(id='conv-trend', figure=empty_figure()),                      # Initialize with empty figure and update with 'update-graphs() function'. Related function: update_graphs()
                                        ])
                                    )
                                ])
                            ], className='card'),
                            id = 'collapse-left',
                            is_open=False)
    
    return convergence_layout


def define_iteration_with_dlc_layout():

    # OpenFAST DLC: needs panel + dlc output
    # RAFT: needs dlc output
    panel_layout = html.Div(id='dlc-panel',
                            children=[html.H5("X Channel Statistics"),
                                        html.Div([dbc.RadioItems(
                                            id='x-stat-option',
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {'label': 'min', 'value': 'min'},
                                                {'label': 'max', 'value': 'max'},
                                                {'label': 'std', 'value': 'std'},
                                                {'label':  'mean', 'value': 'mean'},
                                                {'label': 'median', 'value': 'median'},
                                                {'label': 'abs', 'value': 'abs'},
                                                {'label': 'integrated', 'value': 'integrated'}],
                                        )], className='radio-group'),
                                        html.H5("Y Channel Statistics"),
                                        html.Div([dbc.RadioItems(
                                            id='y-stat-option',
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {'label': 'min', 'value': 'min'},
                                                {'label': 'max', 'value': 'max'},
                                                {'label': 'std', 'value': 'std'},
                                                {'label':  'mean', 'value': 'mean'},
                                                {'label': 'median', 'value': 'median'},
                                                {'label': 'abs', 'value': 'abs'},
                                                {'label': 'integrated', 'value': 'integrated'}],
                                        )], className='radio-group'),
                                        html.H5("X Channel"),
                                        dcc.Dropdown(id='x-channel'),
                                        html.H5("Y Channel"),
                                        dcc.Dropdown(id='y-channel', multi=True)],
                            style={'display': 'none'}       # Default: no visibility (scroll up)
                    )
    
    graph_layout = html.Div(dcc.Graph(id='dlc-output', figure=empty_figure()))                          # Related functions: update_dlc_plot()

    # Layout for visualizing Specific Iteration data - hidden in default
    iteration_with_dlc_layout = dbc.Collapse(
                                    dbc.Card([
                                            dbc.CardHeader(id='dlc-output-iteration'),      # Related function: update_dlc_outputs()
                                            dbc.CardBody([
                                                # dcc.Loading(html.Div(id='dlc-iteration-data'))   # Related function: update_dlc_outputs() -- nonexistent object errors
                                                panel_layout,
                                                graph_layout
                                            ])], className='card'),
                                    id = 'collapse-right',
                                    is_open=False)

    return iteration_with_dlc_layout


# We are using card container where we define sublayout with rows and cols.
def layout():

    load_msg = dbc.Toast(
                        html.P(id='load-warning-text'),
                        id="load-warning",
                        is_open=False,
                        dismissable=True,
                        style={"position": "fixed", "top": 66, "right": 10, "width": 350}
                    )
    
    outlier_div = html.Div([
                    html.H5("Channel to visualize timeseries data"),
                    dcc.Dropdown(id='time-signaly', multi=True),
                    dcc.Graph(id='time-graph', figure=empty_figure())
                ])

    layout = html.Div([
                # Optimization related Data fetched from input-dict
                dcc.Store(id='var-opt', data={}),
                dcc.Store(id='log-data', data={}),
                dcc.Store(id='df', data={}),
                dcc.Store(id='prep-data', data={'log_flag': False, 'raft_flag': False}),
                # Layout
                html.Div(id='data-layout'),
                dbc.Row([
                    dbc.Col(define_convergence_layout(), width=6),
                    dbc.Col(define_iteration_with_dlc_layout(), width=6)
                ], className='g-0'),         # No gutters where horizontal spacing is added between the columns by default

                # Modal Window layout for visualizing Outlier timeseries data
                dbc.Modal([
                    dbc.ModalHeader(dbc.ModalTitle(html.Div(id='outlier-header'))),                                 # Related function: display_outlier()
                    dbc.ModalBody(outlier_div)],                                                         # Related function: display_outlier()
                    id='outlier-div',
                    size='xl',
                    is_open=False),

                # Warning message - check the status of data preprocessing
                dcc.Loading(load_msg)
            ])

    return layout



###################################################################
#   Main Left Layout: Convergence trend data related functions
###################################################################

@callback(Output('collapse-left', 'is_open'),
          Output('conv-signaly', 'options'), 
          Output('conv-signaly', 'value'), 
          Input('prep-data', 'data'),
          State('var-opt', 'data'),
          State('df', 'data'))
def toggle_conv_layout(prep_data, opt_options, df_dict):
    '''
    If iteration has been clicked, open the card layout on right side.
    '''
    # Make sure data preprocessing is ready before data analysis
    if df_dict=={}:
        raise PreventUpdate
    
    if opt_options['opt_type'] == 'RAFT' and (False in [prep_data['log_flag'], prep_data['raft_flag']]):
        raise PreventUpdate

    elif opt_options['opt_type'] == 'OpenFAST' and prep_data['log_flag'] is False:
        raise PreventUpdate
    
    print('toggle conv layout with', opt_options['conv_y'])
    df = pd.DataFrame.from_records(df_dict)
    df = df.applymap(lambda x: np.array(x))
    
    return True, sorted(df.keys()), opt_options['conv_y']


def get_trace(df, label):
    '''
    Add the line graph (trace) for each channel (label)
    ______________
    where
    len(df[label])       # The number of rows
    df[label][0]         # First cell. Size of the list in each cell
    df[label][0].ndim    # Dimension
    '''
    assert isinstance(df[label][0], np.ndarray) == True
    trace_list = []
    
    # Need to parse the data depending on the dimension of values
    if df[label][0].ndim == 0:      # For single value
        # print('Single value')
        trace_list.append(go.Scatter(y = [df[label][i] for i in range(len(df[label]))], mode = 'lines+markers', name = label, showlegend=False))
    
    elif df[label][0].ndim == 1:    # For 1d-array
        # print('1D-array')
        for i in range(df[label][0].size):
            trace_list.append(go.Scatter(y = df[label].str[i], mode = 'lines+markers', name = label+'_'+str(i), showlegend=False))        # Works perfectly fine with 'visualization_demo/log_opt.sql'

    # TODO: how to viz 2d/3d-array cells?
    elif df[label][0].ndim == 2:    # For 2d-array
        print('2D-array')
        print('we cannot visualize arrays with more than one dimension')

    else:
        print('Need to add function..')
        print('we cannot visualize arrays with more than one dimension')
    

    return trace_list



@callback(Output('conv-trend', 'figure'),
          Input('conv-signaly', 'value'),
          State('df', 'data'))
def update_graphs(signaly, df_dict):
    '''
    Draw figures showing convergence trend with selected channels
    '''
    if signaly is None:
        raise PreventUpdate
    
    df = pd.DataFrame.from_records(df_dict)
    df = df.applymap(lambda x: np.array(x))

    # Add subplots for multiple y-channels vertically
    fig = make_subplots(
        rows = len(signaly),
        cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.05)

    for row_idx, label in enumerate(signaly):
        trace_list = get_trace(df, label)
        for trace in trace_list:
            fig.add_trace(trace, row=row_idx+1, col=1)
        fig.update_yaxes(title_text=label, row=row_idx+1, col=1)
    
    fig.update_layout(
        height=250 * len(signaly),
        hovermode='x unified',
        title_x=0.5)

    fig.update_traces(xaxis='x'+str(len(signaly)))   # Spike line hover extended to all subplots

    fig.update_xaxes(
        spikemode='across+marker',
        spikesnap='cursor',
        title_text='Iteration')

    return fig


###############################################################################
# Main Right Layout: DLC related functions for OpenFAST // Plot GIF for RAFT
###############################################################################

@callback(Output('x-stat-option', 'value'),
          Output('y-stat-option', 'value'),
          Output('x-channel', 'options'),
          Output('x-channel', 'value'),
          Output('y-channel', 'options'),
          Output('y-channel', 'value'),
          Input('var-opt', 'data'))
def initialize_panel_values(opt_options):
    if opt_options['opt_type'] == 'RAFT':
        raise PreventUpdate
    
    global cm
    case_matrix_path = opt_options['case_matrix']
    cm = read_cm(case_matrix_path)

    stats, _ = read_per_iteration(0, opt_options['stats_path'])     # Arbitrary index 0 (as it starts with zero)
    multi_indices = sorted(stats.reset_index().keys())
    channel_options = sorted(set([multi_key[0] for idx, multi_key in enumerate(multi_indices)]))

    return opt_options['x_stat'], opt_options['y_stat'], channel_options, opt_options['x'], channel_options, opt_options['y']


@callback(Output('collapse-right', 'is_open'),
          Output('dlc-panel', 'style'),
          Output('dlc-output-iteration', 'children'),
          Output('dlc-output', 'figure'),
          Input('conv-trend', 'clickData'),
          State('prep-data', 'data'),
          State('var-opt', 'data'),
          Input('x-stat-option', 'value'),
          Input('y-stat-option', 'value'),
          Input('x-channel', 'value'),
          Input('y-channel', 'value'))
def toggle_iteration_with_dlc_layout(clickData, prep_data, opt_options, x_chan_option, y_chan_option, x_channel, y_channel):
    '''
    If iteration has been clicked, open the card layout on right side.

    Once iteration has been clicked from the left convergence graph, analyze:
    1) What # of iteration has been clicked
    2) Corresponding iteration related optimization output files
    '''
    if clickData is None:
        raise PreventUpdate
    
    global iteration, stats, iteration_path
    iteration = clickData['points'][0]['x']

    # RAFT Optimization
    if prep_data['raft_flag'] == True:
        title_phrase = f'RAFT Optimization Iteration {iteration}'
        fig = update_raft_outputs(opt_options)
        return True, {'display': 'none'}, title_phrase, fig

    # OpenFAST Optimization
    else:
        title_phrase = f'OpenFAST Optimization Iteration {iteration}'
        stats, iteration_path = read_per_iteration(iteration, opt_options['stats_path'])

        fig = update_dlc_plot(x_chan_option, y_chan_option, x_channel, y_channel)
        return True, {'display': 'block'}, title_phrase, fig


def update_raft_outputs(opt_options):

    # TODO: Make it animation? Reference that works
    # import plotly.express as px
    # df = px.data.gapminder()
    # fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
    #         size="pop", color="continent", hover_name="country",
    #         log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
    
    # Read from matplotlib image
    # Create figure
    fig = go.Figure()
    png_per_iteration = Image.open(f'{opt_options["raft_design_dir"]}/../raft_plots/ptfm_{iteration}.png')
    img_width, img_height = png_per_iteration.size

    # Constants
    scale_factor = 0.8

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source = png_per_iteration)
    )

    # Configure other layout
    fig.update_layout(
        # width=img_width * scale_factor,
        # height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(255, 255, 255, 255)",
        plot_bgcolor="rgba(255, 255, 255, 255)"
    )

    return fig


def update_dlc_plot(x_chan_option, y_chan_option, x_channel, y_channel):
    '''
    Once required channels and stats options have been selected, draw figures that demonstrate DLC analysis.
    It will show default figure with default settings.
    '''
    fig = plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channel)

    return fig


def plot_dlc(cm, stats, x_chan_option, y_chan_option, x_channel, y_channels):
    '''
    Function from:
    https://github.com/WISDEM/WEIS/blob/main/examples/16_postprocessing/rev_DLCs_WEIS.ipynb

    Plot user specified stats option for each DLC over user specified channels
    '''
    dlc_inds = {}

    # TODO: The format of case matrix has been updated..
    try:
        dlcs = cm[('DLC', 'Label')].unique()
        for dlc in dlcs:
            dlc_inds[dlc] = cm[('DLC', 'Label')] == dlc     # dlcs- key: dlc / value: boolean array
    except:
        dlcs = cm['DLC'].unique()
        for dlc in dlcs:
            dlc_inds[dlc] = cm['DLC'] == dlc     # dlcs- key: dlc / value: boolean array
    
    # Add subplots for multiple y-channels vertically
    fig = make_subplots(
        rows = len(y_channels),
        cols = 1,
        shared_xaxes=True,
        vertical_spacing=0.05)

    # Add traces
    for row_idx, y_channel in enumerate(y_channels):
        for dlc, boolean_dlc in dlc_inds.items():
            x = stats.reset_index()[x_channel][x_chan_option].to_numpy()[boolean_dlc]
            y = stats.reset_index()[y_channel][y_chan_option].to_numpy()[boolean_dlc]
            trace = go.Scatter(x=x, y=y, mode='markers', name='dlc_'+str(dlc))
            fig.add_trace(trace, row=row_idx+1, col=1)
        fig.update_yaxes(title_text=f'{y_chan_option.capitalize()} {y_channel}', row=row_idx+1, col=1)

    fig.update_layout(
        height=300 * len(y_channels),
        title_text='DLC Analysis')
    
    fig.update_xaxes(title_text=f'{x_chan_option.capitalize()} {x_channel}', row=len(y_channels), col=1)
    
    return fig


###############################################
# Outlier related functions
###############################################

@callback(Output('outlier-div', 'is_open'),
          Input('dlc-output', 'clickData'),
          State('outlier-div', 'is_open'))
def toggle_outlier_timeseries_layout(clickData, is_open):
    '''
    Once user assumes a point as outlier and click that point, open the modal window showing the corresponding time series data.
    '''
    if clickData is None:
        raise PreventUpdate
    
    return toggle(clickData, is_open)


@callback(Output('outlier-header', 'children'),
          Output('time-signaly', 'options'),
          Output('time-signaly', 'value'),
          Input('dlc-output', 'clickData'),
          Input('var-opt', 'data'))
def display_outlier(clickData, opt_options):
    '''
    Once outlier has been clicked, show corresponding optimization run.
    '''
    if clickData is None or opt_options is None:
        raise PreventUpdate
    
    print("clickData\n", clickData)
    of_run_num = clickData['points'][0]['pointIndex']
    print("corresponding openfast run: ", of_run_num)

    global filename, timeseries_data
    filename, timeseries_data = get_timeseries_data(of_run_num, stats, iteration_path)
    print(timeseries_data)

    return filename, sorted(timeseries_data.keys()), opt_options['y_time']


@callback(Output('time-graph', 'figure'),
          Input('time-signaly', 'value'))
def update_timegraphs(signaly):
    '''
    Function to visualize the time series data graph
    '''
    if signaly is None:
        raise PreventUpdate

    # 1) Single plot
    # fig = make_subplots(rows = 1, cols = 1)
    # for col_idx, label in enumerate(signaly):
    #     fig.append_trace(go.Scatter(
    #         x = timeseries_data['Time'],
    #         y = timeseries_data[label],
    #         mode = 'lines',
    #         name = label),
    #         row = 1,
    #         col = 1)
    
    # 2) Multiple subplots
    fig = make_subplots(rows = len(signaly), cols = 1, shared_xaxes=True, vertical_spacing=0.05)
    for row_idx, label in enumerate(signaly):
        fig.append_trace(go.Scatter(
            x = timeseries_data['Time'],
            y = timeseries_data[label],
            mode = 'lines',
            name = label),
            row = row_idx + 1,
            col = 1)
        fig.update_yaxes(title_text=label, row=row_idx+1, col=1)
    
    fig.update_layout(
        height=200 * len(signaly),
        title=f"{filename}",
        title_x=0.5)
    
    # Define the graph layout where it includes the rendered figure
    fig.update_xaxes(title_text='Time', row=len(signaly), col=1)

    
    return fig


###############################################
#   Automatic Save configurations
###############################################

@callback(Output('var-opt', 'data', allow_duplicate=True),
          State('var-opt', 'data'),
          Input('input-dict', 'data'),
          Input('conv-signaly', 'value'),
          Input('x-stat-option', 'value'),
          Input('y-stat-option', 'value'),
          Input('x-channel', 'value'),
          Input('y-channel', 'value'),
          Input('time-signaly', 'value'),
          State('outlier-div', 'is_open'),
          prevent_initial_call=True)
def save_optimization(opt_options, input_dict, signaly, x_chan_option, y_chan_option, x_channel, y_channel, time_signaly, is_open_outlier):

    if opt_options['opt_type'] == 'OpenFAST' and is_open_outlier is False:
        raise PreventUpdate

    print('Automatic save with ', signaly, x_chan_option, y_chan_option, x_channel, y_channel, time_signaly)     # When time_signaly changed

    input_dict['userPreferences']['optimization']['convergence']['channels'] = signaly

    if opt_options['opt_type'] == 'OpenFAST':
        input_dict['userPreferences']['optimization']['dlc']['xaxis'] = x_channel
        input_dict['userPreferences']['optimization']['dlc']['yaxis'] = y_channel
        input_dict['userPreferences']['optimization']['dlc']['xaxis_stat'] = x_chan_option
        input_dict['userPreferences']['optimization']['dlc']['yaxis_stat'] = y_chan_option
        input_dict['userPreferences']['optimization']['timeseries']['channels'] = time_signaly
    
    opt_options['conv_y'] = signaly
    opt_options['x_stat'] = x_chan_option
    opt_options['y_stat'] = y_chan_option
    opt_options['x'] = x_channel
    opt_options['y'] = y_channel
    opt_options['y_time'] = time_signaly

    update_yaml(input_dict, input_dict['yamlPath'])
    
    return opt_options