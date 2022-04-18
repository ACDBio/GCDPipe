import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, dash_table
import base64
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
#from sklearn.model_selection import GridSearchCV
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import mannwhitneyu
import plotly.figure_factory as ff

#color_scheme = {
#    'twocat_1':"#07BEB8",
#    'twocat_2':"#8F3985"
#    }

#color_scheme = {
#    'twocat_1':"#8F3985",
#    'twocat_2':"#51575A"
#    }  

#color_scheme = {
#    'twocat_1':"#8F3985",
#    'twocat_2':"#709191"
#    }  

color_scheme = {
    'twocat_1':"#8F3985",
    'twocat_2':"#ADB1BA"
    }



app = Dash()
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    dcc.Store(id='gene_data'),
    dcc.Store(id='feature_data'),
    dcc.Store(id='drug_data'),
    dcc.Store(id='drug_list'),
    dcc.Markdown(children='''# GCDPipe: A Random Forest-based gene classification and expression profile ranking pipeline with optional drug prioritization from GWAS genetic fine-mapping results'''),
    dcc.Markdown(children='''The pipeline will use GWAS genetic fine-mapping data and expression profiles across cell types/tissues or other categories of interest to train a Random Forest classifier to distinguish the genes belonging to the risk class. Expression profiles will be ranked by correlation between their SHAP and gene expression values. Optionally, drugs can be ranked by maximal probabilities of their targets to be assigned to a risk class.'''),
    dcc.Markdown(children='''A list of drugs can also be provided to compare probabilities of their targets to be assigned to a risk class with that for other genes as well as to compare maximal risk probabilities of their targets with those for other drugs.'''),
    dcc.Markdown(children='''### Required input files:'''),
    dcc.Markdown(children='''1. a .csv file with columns named "pipe_genesymbol" and "is_True", in which the genes derived from GWAS genetic fine-mapping procedure are listed and their risk category (1 or 0) is provided (a True-False set for classifier training). Example: ...'''),
    dcc.Markdown(children='''2. a .csv file with with columns "pipe_genesymbol" and other columns with custom names - gene expression profiles across tissues/cell types of interest to train the classifier on. Example: ...'''),
    dcc.Markdown(children='''### Optional input files (for drug-gene interaction analysis):'''),
    dcc.Markdown(children='''3. a .csv file with columns named "DRUGBANK_ID" and "pipe_genesymbol", in which drugs and their gene targets are provided. Example: ... (is assembled from DrugCentral and DGIDB data and can be used as a default input)'''),
    dcc.Markdown(children='''4. a .csv file with a column named "DRUGBANK_ID" with drugs belinging to the category of interest. Example: ...'''),
    dcc.Markdown(children='''Note that the provided hyperparameter search space can be changed using the sliders.'''),
    dcc.Markdown(children='''See ...github... for further explanations.'''),

    dcc.Markdown(children='''# Input dataset upload'''),
    dcc.Upload(
        id='upload-genes',
        children=html.Div([
            'Gene data: Drag and Drop or ',
            html.A('Select a File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    dcc.Loading(
                    id="genes-loading",
                    children=[html.Div([html.Div(id='upload-genes-display')])],
                    type="cube",
                ),

    dcc.Upload(
        id='upload-features',
        children=html.Div([
            'Feature data (expression profiles): Drag and Drop or ',
            html.A('Select a File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    dcc.Loading(
                    id="features-loading",
                    children=[html.Div([html.Div(id='upload-features-display')])],
                    type="cube",
                ),

    dcc.Checklist(options=[
       {'label': 'Drug-risk gene interaction assessment', 'value': 'drug_analysis_True'}], 
       id='upload-drugs-checklist'),
    html.Div(id='upload-drugs-container'),
    dcc.Markdown(children='''# Training settings'''),
    dcc.Markdown(children=''' \n Number of estimators:'''),
    dcc.Input(id="input-n_estimators", type="number", placeholder="Enter the number...", style={'marginRight':'100px'},  value=3000),
    dcc.Markdown(children=''' \n Testing gene set/training gene set ratio:'''),
    dcc.RangeSlider(0, 1, 0.1, value=[0.3], id='input-testing_gs_ratio',  tooltip={"placement": "bottom", "always_visible": True}, marks=None),
    dcc.Markdown(children=''' \n Max tree depth:'''),
    dcc.RangeSlider(1, 100, 1, value=[1], id='input-max_depth',  tooltip={"placement": "bottom", "always_visible": True}, marks=None),
    dcc.Markdown(children=''' \n Number of samples per leaf:'''),
    dcc.RangeSlider(1, 50, 1, value=[5,5,5,5,5,5,5,5,5,5], id='input-n_samples_per_leaf',  tooltip={"placement": "bottom", "always_visible": True}), #[5,10,15,20,25,30] was used in the original study
    dcc.Markdown(children=''' \n Min number of samples required to split an internal node:'''),
    dcc.RangeSlider(1, 50, 1, value=[5,5,5,5,5,5,5,5,5,5], id='input-min_samples_split',  tooltip={"placement": "bottom", "always_visible": True}), #[2,5,10,20,30,40] was used in the original study
    
    html.Div([html.Button("Launch the pipeline", id="start_training", style={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem"}),
    dcc.Loading(
                    id="training-loading",
                    children=[html.Div([html.Div(id="training-output")])],
                    type="cube",
                )
    ]
    ),
])


def parse_content(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df.to_json(orient='split')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])



@app.callback(Output('gene_data', 'data'),
              Input('upload-genes', 'contents'),
              State('upload-genes', 'filename'),
              State('upload-genes', 'last_modified'))
def update_output(contents, name, date):
    if contents is not None:
        data_content=parse_content(contents, name, date)
        return data_content


@app.callback(Output('upload-genes-display', 'children'),
              Input('gene_data', 'data'))
def update_output(data):
    if data is not None:
        df=pd.read_json(data, orient='split')
        df=df.loc[0:2,:]
        return [


            html.Div('First rows read:'),
            dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
         'minWidth': '10px', 'width': '100px', 'maxWidth': '500px',
         'overflow': 'hidden',
         'textOverflow': 'ellipsis'
    },
    fill_width=False
        )]
    else:
        return []

        
@app.callback(Output('upload-features-display', 'children'),
              Input('feature_data', 'data'))
def update_output(data):
    if data is not None:
        df=pd.read_json(data, orient='split')
        df=df.loc[0:2,:]
        return [


            html.Div('First rows read:'),
            dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            fill_width=False
        )]
    else:
        return []


@app.callback(Output('upload-drugs-display', 'children'),
              Input('drug_data', 'data'))
def update_output(data):
    if data is not None:
        df=pd.read_json(data, orient='split')
        df=df.loc[0:2,:]
        return [


            html.Div('First rows read:'),
            dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            fill_width=False
        )]
    else:
        return []

@app.callback(Output('upload-drug-list-display', 'children'),
              Input('drug_list', 'data'))
def update_output(data):
    if data is not None:
        df=pd.read_json(data, orient='split')
        df=df.loc[0:2,:]
        return [


            html.Div('First rows read:'),
            dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                'whiteSpace': 'normal'
            },
            fill_width=False
        )]
    else:
        return []







@app.callback(Output('feature_data', 'data'),
              Input('upload-features', 'contents'),
              State('upload-features', 'filename'),
              State('upload-features', 'last_modified'))
def update_output(contents, name, date):
    if contents is not None:
        data_content=parse_content(contents, name, date)
        return data_content

@app.callback(Output('drug_data', 'data'),
              Input('upload-drugs', 'contents'),
              State('upload-drugs', 'filename'),
              State('upload-drugs', 'last_modified'))
def update_output(contents, name, date):
    if contents is not None:
        data_content=parse_content(contents, name, date)
        return data_content

@app.callback(Output('drug_list', 'data'),
              Input('upload-drug-list', 'contents'),
              State('upload-drug-list', 'filename'),
              State('upload-drug-list', 'last_modified'))
def update_output(contents, name, date):
    if contents is not None:
        data_content=parse_content(contents, name, date)
        return data_content


@app.callback(Output('upload-drugs-container', 'children'),
              Input('upload-drugs-checklist', 'value'))
def add_drugs_upload(drug_option_value):

    if (drug_option_value is not None) and (drug_option_value!=[]):
        drug_option_value=drug_option_value[0]


    if drug_option_value=='drug_analysis_True':
        return [
                dcc.Upload(
        id='upload-drugs',
        children=html.Div([
            'Drug-target interaction data (binary): Drag and Drop or ',
            html.A('Select a File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='upload-drugs-display'),
    dcc.Checklist(options=[
       {'label': 'Compare drug target gene probabilities to be assigned to the risk class for the selected drugs with that for other genes', 'value': 'drug_target_comparison_True'},
       {'label': 'Compare drug selection scores with other drug scores', 'value': 'drug_comparison_True'}], 
       id='drug-analysis-options-checklist'), 
    html.Div(id='drug-target-enrichment-options-container')   
        ]
    
@app.callback(Output('drug-target-enrichment-options-container', 'children'),
              Input('drug-analysis-options-checklist', 'value'))
def add_drug_list_upload(drug_analysis_options_value):

    if drug_analysis_options_value is not None:
    
        if ('drug_target_comparison_True' in drug_analysis_options_value) or ('drug_comparison_True' in drug_analysis_options_value):
            return [
                dcc.Upload(
            id='upload-drug-list',
            children=html.Div([
                'Target drug category Drugbank IDs: Drag and Drop or ',
                html.A('Select a File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='upload-drug-list-display'),
            ]
    else:
        return []
    
@app.callback(Output("training-output", 'children'),
              Input('start_training','n_clicks'),
              Input('gene_data', 'data'),
              Input('feature_data', 'data'),
              Input('input-n_estimators','value'),
              Input('input-testing_gs_ratio', 'value'),
              Input('input-max_depth','value'),
              Input('input-n_samples_per_leaf','value'),
              Input('input-min_samples_split','value'),
              Input('upload-drugs-checklist','value'),
              Input('drug_data', 'data'),
              Input('drug-analysis-options-checklist', 'value'),
              Input('drug_list', 'data'))
def train_rf_classifier(n_clicks, gene_data, feature_data, n_estimators, test_ratio, max_depth, n_samples_leaf, min_samples_split, drug_analysis_requirement, d_data, drug_analysis_options, d_list):
    if n_clicks is not None:
        if n_clicks>0:
            if gene_data is None:
                return html.Div(['Gene data is missing...'])

            if feature_data is None:
                return html.Div(['Feature data (gene expression profiles) is missing...'])


            gene_df=pd.read_json(gene_data, orient='split')
            feature_df=pd.read_json(feature_data, orient='split')

            processed_df=pd.merge(feature_df, gene_df, on='pipe_genesymbol',how='left')
            processed_df.is_True=processed_df.is_True.fillna('No_data')
            classifier_building_df=processed_df[processed_df.is_True!='No_data']

            classifier_building_df.is_True=[1 if x==1.0 else 0 for x in classifier_building_df.is_True]
            print('True gene count:')
            print(classifier_building_df.is_True.sum())
            
            classifier_building_df=classifier_building_df.set_index('pipe_genesymbol')

            X_train, X_test, Y_train, Y_test = train_test_split(classifier_building_df.iloc[:,:-1],classifier_building_df.is_True, test_size=test_ratio[0], stratify=classifier_building_df.is_True)
            print('Training risk gene count:')
            print(Y_train.sum())
            print('Training risk gene count:')
            print(Y_test.sum())
            
            classifier_tuned_parameters_rf={'bootstrap': [True],
                                            'max_depth': np.unique(max_depth),
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf':np.unique(n_samples_leaf),
                                            'min_samples_split':np.unique(min_samples_split),
                                            'max_features':[None],
                                            'n_estimators': [n_estimators]}

            clf=GridSearchCVProgressBar(RandomForestClassifier(oob_score=True,bootstrap = True), classifier_tuned_parameters_rf, scoring='f1_weighted', verbose=1)
            print('Hyperparameter tuning for the rf model...')
            clf.fit(X_train, Y_train)
            best_fitting_score=clf.best_score_
            Y_true, Y_pred = Y_test, clf.best_estimator_.predict_proba(X_test)
            print('Y_true')
            print(Y_true)
            print('Y_pred')
            print(Y_pred)

            Y_pred=Y_pred[:,1]
            fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
            roc_auc=roc_auc_score(Y_true, Y_pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print("Threshold value is:", optimal_threshold)
            predicted_classes=[1 if x>optimal_threshold else 0 for x in Y_pred] #here might be reasonable to try just >
            cm = confusion_matrix(Y_true, predicted_classes, labels=[0, 1])
            cls_report=classification_report(Y_true, predicted_classes, output_dict=True)
            print(cls_report)
            #precision=cm[1,1]/(cm[0,1]+cm[1,1])
            #accuracy=(cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0])

            #print('Plotting the classifier training stats...')

            # Confusion matrix hitmap
            #plt.figure()
            #sns.heatmap(cm, annot=True, cmap="YlGnBu")
            #plt.title(f'Confusion matrix for the optimal ROC-AUC threshold.')
            #fig_cm_sns = plt.gcf()
            #plt.close()


            #ROC-AUC curve
            #plt.figure()
            #lw = 2
            #plt.plot(
            #        fpr,
            #        tpr,
            #        color="darkorange",
            #        lw=lw,
            #        label="ROC curve (area = %0.2f)" % roc_auc,
            #    )
            #plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel("False Positive Rate")
            #plt.ylabel("True Positive Rate")
            #plt.title("Receiver operating characteristic")
            #plt.legend(loc="lower right")
            fig_roc_sns = plt.gcf()
            plt.close()
            fig_roc = go.Figure()
            name = f"AUC={roc_auc:.2f}"
            fig_roc.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
            fig_roc.update_layout(
                    template='plotly_white',
                    title=name,
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    xaxis=dict(constrain='domain'),
                    width=500, height=500
                )


            print('Scoring the whole dataset...')
            processed_df=processed_df.set_index('pipe_genesymbol')
            processed_df=processed_df.iloc[:,:-1]
            predicted_probs=pd.DataFrame.from_dict({'pipe_genesymbol':processed_df.index.tolist()})
            predicted_probs['score']=clf.best_estimator_.predict_proba(processed_df)[:,1]

            input_risk_genes=list(classifier_building_df[classifier_building_df.is_True==1].index.tolist())
            predicted_probs['is_input_risk_gene']=[1 if gene in input_risk_genes else 0 for gene in predicted_probs['pipe_genesymbol']]


            predicted_probs=predicted_probs.sort_values(ascending=False, by=['score'])


            explainer = shap.TreeExplainer(clf.best_estimator_)
            shap_values = explainer.shap_values(classifier_building_df.iloc[:,:-1])     

            shap_riskclass_df=pd.DataFrame(shap_values[1])
            shap_riskclass_df.columns=classifier_building_df.iloc[:, 1:].columns
            corr_between_shap_and_exprs_values=shap_riskclass_df.corrwith(classifier_building_df.iloc[:, 1:].reset_index(drop=True),axis=0)
            riskclass_shap_featurevalue_correlation=corr_between_shap_and_exprs_values.sort_values(ascending=False)
            riskclass_shap_featurevalue_correlation=riskclass_shap_featurevalue_correlation.fillna(0)
            riskclass_shap_featurevalue_correlation=pd.DataFrame.from_dict({'expression_profile':riskclass_shap_featurevalue_correlation.index.tolist(),'importance_based_score':list(riskclass_shap_featurevalue_correlation)})
            res= [
                 dcc.Markdown(children='''# Classifier test performance'''),
                 dcc.Graph(figure=fig_roc),
                 dcc.Markdown(children='''## Classification report'''),
                 dcc.Markdown(children='''### Non-risk class: '''),
                 html.Div([f"Precision {round(cls_report['0']['precision'],3)}"]),
                 html.Div([f"Recall {round(cls_report['0']['recall'],3)}"]),
                 html.Div([f"F1 score {round(cls_report['0']['f1-score'],3)}"]),
                 html.Div([f"Support {cls_report['0']['support']}"]),

                 dcc.Markdown(children='''### Risk class: '''),
                 html.Div([f"Precision {round(cls_report['1']['precision'], 3)}"]),
                 html.Div([f"Recall {round(cls_report['1']['recall'], 3)}"]),
                 html.Div([f"F1 score {round(cls_report['1']['f1-score'], 3)}"]),
                 html.Div([f"Support {cls_report['1']['support']}"]),

                 dcc.Markdown(children='''### Global: '''),
                 html.Div([f"Accuracy {round(cls_report['accuracy'], 3)}"]),
                 html.Div([f"Macro average precision {round(cls_report['macro avg']['precision'], 3)}"]),
                 html.Div([f"Macro average recall {round(cls_report['macro avg']['recall'], 3)}"]),
                 html.Div([f"Macro average f1 score {round(cls_report['macro avg']['f1-score'], 3)}"]),
                 html.Div([f"Macro average support {cls_report['macro avg']['support']}"]),
                 
                 dcc.Store(id='gene_probabilities', data=predicted_probs.to_json(orient='split')),
                 dcc.Store(id='expression_profile_importances', data=riskclass_shap_featurevalue_correlation.to_json(orient='split')), 
                 
                 dcc.Markdown(children='''# Classification results'''),
                 dcc.Markdown(children='''### Gene probabilities to be assigned to the risk class: '''),
                 dash_table.DataTable(
                    id='gene_risk_probabilities',
                    data=predicted_probs.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in predicted_probs.columns],
                    style_table={'overflowX': 'auto'},
                    editable=False,
                    page_current= 0,
                    page_size= 10,
                    filter_action="native",
                    sort_action="native",
                    export_format="csv",
                    style_cell={
                    'minWidth': '10px', 'width': '100px', 'maxWidth': '500px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                    },
                    fill_width=False
                    ),
                dcc.Markdown(children='''### Importance-based expression profile scores: '''),
                dash_table.DataTable(
                    id='expression_profile_importances',
                    data=riskclass_shap_featurevalue_correlation.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in riskclass_shap_featurevalue_correlation.columns],
                    style_table={'overflowX': 'auto'},
                    editable=False,
                    page_current= 0,
                    page_size= 10,
                    filter_action="native",
                    sort_action="native",
                    export_format="csv",
                    style_cell={
                    'minWidth': '10px', 'width': '100px', 'maxWidth': '500px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                    },
                    fill_width=False
                    )]
            if drug_analysis_requirement is not None:
                if drug_analysis_requirement[0]=='drug_analysis_True':
                    res.append(dcc.Markdown(children='''# Drug ranking results'''))
                    res.append(dcc.Markdown(children='''### Drug scores: '''))
                    if d_data is None:
                        res.append(html.Div(['Drug data is missing...']))
                    else:
                        drugdata_df=pd.read_json(d_data, orient='split')
                        drugdata_df=pd.merge(drugdata_df, predicted_probs[['pipe_genesymbol','score']], on='pipe_genesymbol', how='left')
                        drugdata_df=drugdata_df.dropna()
                        drugdata_df_agg=drugdata_df.groupby('DRUGBANK_ID')[['score']].agg('max')
                        drugdata_df_agg=drugdata_df_agg.sort_values(ascending=False, by=['score'])
                        drugdata_df_agg=drugdata_df_agg.reset_index()
                        
                        res.append(dash_table.DataTable(
                            id='drug_scores',
                            data=drugdata_df_agg.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in drugdata_df_agg.columns],
                            style_table={'overflowX': 'auto'},
                            editable=False,
                            page_current= 0,
                            page_size= 10,
                            filter_action="native",
                            sort_action="native",
                            export_format="csv",
                            style_cell={
                            'minWidth': '10px', 'width': '100px', 'maxWidth': '500px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                            },
                            fill_width=False
                            ))
                            
            if ('drug_target_comparison_True' in drug_analysis_options) or ('drug_comparison_True' in drug_analysis_options): 
                res.append(dcc.Markdown(children='''### Drug selection comparison: '''))
                if d_list is None:
                    res.append(html.Div(['Drug selection list is missing...']))
                else:
                    d_list=pd.read_json(d_list, orient='split')
                    d_list['target_drug_category']='True'

                    if 'drug_target_comparison_True' in drug_analysis_options:

                        drugdata_df_forcomparison=pd.merge(drugdata_df, d_list, on='DRUGBANK_ID', how='left')
                        target_genes=np.unique(list(drugdata_df_forcomparison[drugdata_df_forcomparison.target_drug_category=='True']['pipe_genesymbol']))
                        target_probs_comparison=predicted_probs
                        target_probs_comparison['is_target_for_selected_drugs']=['True' if gene in target_genes else 'False' for gene in target_probs_comparison['pipe_genesymbol']]



                        res.append(dcc.Markdown(children='''#### Comparison between target gene probabilities to be assigned to the risk class for the selected drugs and other genes: '''))

                        #fig_target_comparison = px.histogram(target_probs_comparison, x='score', color='is_target_for_selected_drugs',
                        #                    marginal="box", # box, violin, rug
                        #                    hover_data=target_probs_comparison.columns,
                        #                    opacity=0.5,
                        #                    histnorm="probability density",
                        #                    barmode='overlay')

                        hist_data = [list(target_probs_comparison[target_probs_comparison.is_target_for_selected_drugs=='False']['score']),
                                    list(target_probs_comparison[target_probs_comparison.is_target_for_selected_drugs=='True']['score'])]
                        group_labels = ['Other genes', 'Target genes for selected drugs']
                        colors = [color_scheme['twocat_2'], color_scheme['twocat_1']]
                        fig_target_comparison = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=.05, show_rug=True) #bin_size=.2






                        U1, p = mannwhitneyu(list(target_probs_comparison[target_probs_comparison.is_target_for_selected_drugs=='True']['score']),
                        list(target_probs_comparison[target_probs_comparison.is_target_for_selected_drugs=='False']['score']))
                        p=round(p, 5)
                        name=f'P={p}, U={U1} (Mann-Whitney)'

                        #fig_target_comparison.update_traces(showlegend=False)
                        fig_target_comparison.update_layout(
                            template='plotly_white',
                            title=name,
                            xaxis_title='is drug selection target',
                            width=1000, height=700)

                        res.append(dcc.Graph(figure=fig_target_comparison))

                        fig_target_comparison_box = px.box(target_probs_comparison, x='is_target_for_selected_drugs', y='score', color='is_target_for_selected_drugs', points='all',
                            color_discrete_map={
                                    'True': color_scheme['twocat_1'],
                                    'False': color_scheme['twocat_2']
                                },
                            hover_data=['pipe_genesymbol', 'score']
                            )
                        fig_target_comparison_box.update_traces(showlegend=False)
                        fig_target_comparison_box.update_layout(
                            template='plotly_white',
                            xaxis_title='is drug selection target',
                            width=500, height=500)

                        res.append(dcc.Graph(figure=fig_target_comparison_box))



                    

                    if 'drug_comparison_True' in drug_analysis_options:
                        res.append(dcc.Markdown(children='''#### Comparison between aggregation-based drug scores for the selected drugs and other drugs: '''))

                        drugdata_df_agg_drug_comparison=pd.merge(drugdata_df_agg, d_list, on='DRUGBANK_ID', how='left')
                        drugdata_df_agg_drug_comparison.target_drug_category=drugdata_df_agg_drug_comparison.target_drug_category.fillna('False')

                        #fig_drug_comparison = px.histogram(drugdata_df_agg_drug_comparison, x='score', color='target_drug_category',
                        #                    marginal="box", # box, violin, rug
                        #                    hover_data=drugdata_df_agg_drug_comparison.columns,
                        #                    opacity=0.5,
                        #                    histnorm="probability density",
                        #                    barmode='overlay')


                        hist_data = [list(drugdata_df_agg_drug_comparison[drugdata_df_agg_drug_comparison.target_drug_category=='False']['score']),
                                    list(drugdata_df_agg_drug_comparison[drugdata_df_agg_drug_comparison.target_drug_category=='True']['score'])]
                        group_labels = ['Other drugs', 'Selected drugs']
                        colors = [color_scheme['twocat_2'], color_scheme['twocat_1']]

                        fig_drug_comparison = ff.create_distplot(hist_data, group_labels, colors=colors, bin_size=.05, show_rug=True) #bin_size=.2





                        U1, p = mannwhitneyu(list(drugdata_df_agg_drug_comparison[drugdata_df_agg_drug_comparison.target_drug_category=='True']['score']),
                        list(drugdata_df_agg_drug_comparison[drugdata_df_agg_drug_comparison.target_drug_category=='False']['score']))
                        p=round(p, 5)
                        name=f'P={p}, U={U1} (Mann-Whitney)'

                        #fig_drug_comparison.update_traces(showlegend=False)
                        fig_drug_comparison.update_layout(
                            template='plotly_white',
                            title=name,
                            xaxis_title='scores',
                            width=1000, height=700)

                        res.append(dcc.Graph(figure=fig_drug_comparison))
                    
                        fig_drug_comparison_box = px.box(drugdata_df_agg_drug_comparison, x='target_drug_category', y='score', color='target_drug_category', points='all',
                            color_discrete_map={
                                    'True': color_scheme['twocat_1'],
                                    'False': color_scheme['twocat_2']
                                },
                                hover_data=['DRUGBANK_ID','score']
                            )
                        fig_drug_comparison_box.update_traces(showlegend=False)
                        fig_drug_comparison_box.update_layout(
                            template='plotly_white',
                            xaxis_title='is selected drug',
                            width=500, height=500)

                        res.append(dcc.Graph(figure=fig_drug_comparison_box))



            return res



        


if __name__ == '__main__':
    app.run_server(debug=False)
