import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from tqdm.autonotebook import tqdm


#ChEMBL
def get_ChEMBL_indications_molecular_data(entity, entity_type='efo_term', get_phases_coded_inonehot=True, drop_ifna_incols=['canonical_SMILES'], save=True, savepath='./'):
    '''Extracts all molecules for specific indication defined by EFO or MESH id or term. Returns a dataframe with molecule data.'''
    print(f'Extracting ChEMBL compounds tested for {entity}.')
    drug_indication=new_client.drug_indication
    molecules=new_client.molecule
    if entity_type=='efo_term':
        tar_ind=drug_indication.filter(efo_term__icontains=entity)
    elif entity_type=='efo_id':
        tar_ind=drug_indication.filter(efo_id__iexact=entity)
    elif entity_type=='mesh_term':
        tar_ind=drug_indication.filter(mesh_heading__icontains=entity)
    elif entity_type=='mesh_id':
        tar_ind=drug_indication.filter(mesh_id__iexact=entity)
    else:
        print('Unexpected entity_type value. Use one of: efo_term, efo_id, mesh_term, mesh_id.')

    all_data=[]
    for data_entry in tqdm(tar_ind):
        data={}
        data['CHEMBL_ID']=[data_entry['molecule_chembl_id']]
        data[f'TARGET_INDICATION_{entity_type}']=[entity]
        data['TARGET_INDICATION_mesh_id']=[data_entry['mesh_id']]
        data['TARGET_INDICATION_mesh_heading']=[data_entry['mesh_heading']]
        data['max_phase_for_target_ind']=[data_entry['max_phase_for_ind']]
        data['efo_id']=[data_entry['efo_id']]
        data['efo_term']=[data_entry['efo_term']]
        
        tar_mol=molecules.filter(molecule_chembl_id__iexact=data_entry['molecule_chembl_id'])
        tar_mol=tar_mol[0]
        data['mol_name']=[tar_mol['pref_name']]
        data['initial_indication_class']=[tar_mol['indication_class']]
        data['molecule_type']=[tar_mol['molecule_type']]
        if tar_mol['molecule_type']=='Small molecule':
            try:
                data['canonical_SMILES']=[tar_mol['molecule_structures']['canonical_smiles']]
                data['standard_inchi_key']=[tar_mol['molecule_structures']['standard_inchi_key']]
            except:
                data['canonical_SMILES']=[None]
                data['standard_inchi_key']=[None]
                print(f"SMILES not retrieved for {tar_mol['pref_name']}.")

        else:
            data['canonical_SMILES']=['NotApplicable']
            data['standard_inchi_key']=['NotApplicable']

        
        data['therapeutic_flag']=[tar_mol['therapeutic_flag']]
        data['prodrug']=[tar_mol['prodrug']]

        data['withdrawn_flag']=[tar_mol['withdrawn_flag']]
        data['withdrawn_reason']=[tar_mol['withdrawn_reason']]
        data['withdrawn_class']=[tar_mol['withdrawn_class']]
        data['black_box_warning']=[tar_mol['black_box_warning']]
        data=pd.DataFrame.from_dict(data)
        all_data.append(data)
    resdf=pd.concat(all_data)
    resdf=resdf.reset_index(drop=True)
    if len(drop_ifna_incols)>0:
        resdf=resdf.dropna(subset=drop_ifna_incols)

    if get_phases_coded_inonehot==True:
        resdf=numeric_to_onehot_transitions(resdf)
    if save==True:
        savepath=savepath+entity+'.tsv'
        resdf.to_csv(savepath, sep='\t')
        print(f'ChEMBL data is saved to {savepath}.')    
    return resdf



def get_ChEMBL_indications_molecular_data_fordiseaselist(entity_list, entity_type='efo_term', save=True, savepath='./'):
    results=[]
    for entity in entity_list:
        print(f'Started processing condition {entity}...')
        result=get_ChEMBL_indications_molecular_data(entity, entity_type=entity_type, get_phases_coded_inonehot=True, drop_ifna_incols=['canonical_SMILES'], save=False, savepath='./')
        results.append(result)
    all_results=pd.concat(results)
    if save==True:
        savepath=savepath+entity+'.tsv'
        all_results.to_csv(savepath, sep='\t')
        print(f'ChEMBL data is saved to {savepath}.')    
    return all_results



def get_chembl_mols_by_indication(efo_term_includes):
    '''Sources molecules with specific efo indications.'''
    drug_indication=new_client.drug_indication
    molecules=new_client.molecule
    tar_ind=drug_indication.filter(efo_term__icontains=efo_term_includes)
    tar_mols=molecules.filter(molecule_chembl_id__in=[x['molecule_chembl_id'] for x in tar_ind])
    return tar_mols

def chembl_mol_xml_to_df(xml_data_list):
    '''Generates a dataframe from an xml molecular data sourced from ChEMBL.'''
    all_data=[]
    for molecule_data in tqdm(xml_data_list):
        moldict={}
        


        moldict['CHEMBL_ID']=[molecule_data['molecule_chembl_id']]
        moldict['name']=[molecule_data['pref_name']]
        moldict['indication_class']=[molecule_data['indication_class']]
        moldict['atc_codes']=['|'.join(molecule_data['atc_classifications'])]

        moldict['max_phase']=[molecule_data['max_phase']] #prodrug
        moldict['first_approval']=[molecule_data['first_approval']]


        moldict['molecule_type']=[molecule_data['molecule_type']]
        if moldict['molecule_type']==['Small molecule']:
            try:
                moldict['canonical_SMILES']=[molecule_data['molecule_structures']['canonical_smiles']]
                moldict['standard_inchi_key']=[molecule_data['molecule_structures']['standard_inchi_key']]
            except:
                moldict['canonical_SMILES']=[None]
                moldict['standard_inchi_key']=[None]
                print(f"SMILES not retrieved for {moldict['name'][0]}.")

        else:
            moldict['canonical_SMILES']=['NotApplicable']
            moldict['standard_inchi_key']=['NotApplicable']

        moldict['therapeutic_flag']=[molecule_data['therapeutic_flag']]
        moldict['prodrug']=[molecule_data['prodrug']]

        moldict['withdrawn_flag']=[molecule_data['withdrawn_flag']]
        moldict['withdrawn_reason']=[molecule_data['withdrawn_reason']]
        moldict['withdrawn_class']=[molecule_data['withdrawn_class']]
        moldict['black_box_warning']=[molecule_data['black_box_warning']]

        data=pd.DataFrame.from_dict(moldict)
        all_data.append(data)
    resdf=pd.concat(all_data)
    return resdf

def numeric_to_onehot_transitions(df, column_to_spread='max_phase_for_target_ind', compound_id_col='CHEMBL_ID', max_numeric_class=4, fuse=True):
    '''Generates a dataframe, in which clinical trial phases transitions are coded in one-hot manner.'''
    print('Coding the clinical trial phases in one-hot manner...')
    all_results=[]
    for idx in tqdm(range(len(df[column_to_spread]))):
        res={}
        cur_data=df.iloc[idx,:]
        res[compound_id_col]=[cur_data[compound_id_col]]
        max_val=cur_data[column_to_spread]
        for class_num in list(map(str, list(range(max_numeric_class)))):
            if int(class_num)<=max_val:
                res[f'clinical_studies_done|||PHASE_{class_num}']=[1]
                if int(class_num)>0:
                    res[f'clinical_studies_transitions|||PHASE_{int(class_num)-1}->{int(class_num)}']=[1]
            else:
                res[f'clinical_studies_done|||PHASE_{class_num}']=[0]
                if int(class_num)>0:
                    res[f'clinical_studies_transitions|||PHASE_{int(class_num)-1}->{int(class_num)}']=[0]
        res=pd.DataFrame.from_dict(res)
        all_results.append(res)
    all_results=pd.concat(all_results)
    if fuse==True:
        all_results=pd.merge(df, all_results, on=compound_id_col, how='left')
    return all_results




#DrugBank
def parse_drugbank_xml(path_to_drugbank_data='/home/biorp/Gitrepos/InterPred/full_database.xml', save=True, savepath='DRUGBANK_interactions_DB_full.tsv'):
    '''Drugbank database xml file parsing for getting drug-target interaction modes.'''
    with open(path_to_drugbank_data) as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = '{http://www.drugbank.ca}'
    #Drugbank ID parsing for drug-target interaction modes
    rows = list()
    for i, drug in enumerate(root):
        assert drug.tag == ns + 'drug'
        dbid = drug.findtext(ns + "drugbank-id[@primary='true']")
        drugname = drug.findtext(ns + "name")
        if len(drug.findall('{http://www.drugbank.ca}calculated-properties'))>0:
            drug_properties=list(drug.findall('{http://www.drugbank.ca}calculated-properties')[0])
            for x in drug_properties:
                if x.findall('{http://www.drugbank.ca}kind')[0].text=='IUPAC Name':
                    IUPAC=x.findall('{http://www.drugbank.ca}value')[0].text
                if x.findall('{http://www.drugbank.ca}kind')[0].text=='SMILES':
                    SMILES=x.findall('{http://www.drugbank.ca}value')[0].text
                if x.findall('{http://www.drugbank.ca}kind')[0].text=='InChI':
                    InChI=x.findall('{http://www.drugbank.ca}value')[0].text
                if x.findall('{http://www.drugbank.ca}kind')[0].text=='InChIKey':
                    InChIKey=x.findall('{http://www.drugbank.ca}value')[0].text
        else:
            IUPAC=None
            SMILES=None
            InChI=None
            InChIKey=None

        #print(drugname)
        #print(dbid)
        #print(drug.findall('{http://www.drugbank.ca}targets/{http://www.drugbank.ca}target'))
        for target in drug.findall('{http://www.drugbank.ca}targets/{http://www.drugbank.ca}target'):
            row={}
            targetname=target.findall('{http://www.drugbank.ca}id')[0].text
            #print(targetname)

            gene_data=target.findall('{http://www.drugbank.ca}polypeptide/{http://www.drugbank.ca}gene-name')
            if len(gene_data)>0:
                gene_symbol=gene_data[0].text
                prtn_name=target.findall('{http://www.drugbank.ca}polypeptide/{http://www.drugbank.ca}name')[0].text
                prtn_seq=target.findall('{http://www.drugbank.ca}polypeptide/{http://www.drugbank.ca}amino-acid-sequence')[0].text
            else:
                gene_symbol=None
                prtn_name=None
                prtn_seq=None
            actions=target.findall('{http://www.drugbank.ca}actions')
            for act in actions:
                if len(act)>0:
                    for ac in act:
                        row['DRUGBANK_ID']=dbid
                        row['DRUG_Common_name']=drugname
                        row['target']=targetname
                        row['action']=ac.text 
                        row['target_gene_symbol']=gene_symbol
                        row['target_protein_name']=prtn_name
                        row['target_protein_sequence']=prtn_seq
                        row['SMILES']=SMILES
                        row['IUPAC']=IUPAC
                        row['InChI']=InChI
                        row['InChIKey']=InChIKey
                else:
                    row['DRUGBANK_ID']=dbid
                    row['DRUG_Common_name']=drugname
                    row['target']=targetname
                    row['action']='unknown' 
                    row['target_gene_symbol']=gene_symbol
                    row['target_protein_name']=prtn_name
                    row['target_protein_sequence']=prtn_seq
                    row['SMILES']=SMILES
                    row['IUPAC']=IUPAC
                    row['InChI']=InChI
                    row['InChIKey']=InChIKey
            rows.append(row)  
    drugbank_df = pd.DataFrame.from_dict(rows)
    drugbank_df_full=drugbank_df
    drugbank_df_full=drugbank_df_full.drop_duplicates()
    drugbank_df_full=drugbank_df_full.reset_index(drop=True)
    drugbank_df_full=drugbank_df_full.fillna('NoData')
    drugbank_df_full.DRUG_Common_name=[str(x).lower() for x in drugbank_df_full.DRUG_Common_name]
    if save==True:
        drugbank_df_full.to_csv(savepath, sep='\t')
    return drugbank_df_full