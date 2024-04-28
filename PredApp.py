import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdchem, QED, rdmolops, rdMolDescriptors
import glob
import pandas as pd
from padelpy import padeldescriptor
from joblib import load
from sklearn.ensemble import RandomForestRegressor


import os

filepath = os.path.dirname(__file__)
xml = os.path.join(filepath, r'xml/*.xml')
title = os.path.join(filepath, 'title.txt')
image = os.path.join(filepath, 'hello.png')
about = os.path.join(filepath, 'About.txt')
model = os.path.join(filepath, 'model.txt')
data = os.path.join(filepath, 'data.txt')
dataimg  = os.path.join(filepath, 'dta.png')
citation = os.path.join(filepath, 'citation.txt')
citationlink = os.path.join(filepath, 'citationlink.txt')
author = os.path.join(filepath, 'authors.txt')
smiledraw = os.path.join(filepath, 'mol.png')
rmodel = os.path.join(filepath, 'model.joblib')
smilefile = os.path.join(filepath, 'molecule.smi')
fingerprint_output_file = os.path.join(filepath, "fingerprint.csv")
fingerprint_output_file_txt = os.path.join(filepath, "fingerprint.csv.log")
cfp = os.path.join(filepath, 'cfp.txt')
loadm = os.path.join(filepath, 'model.joblib')
xcol = os.path.join(filepath, 'col.csv')

st.sidebar.title('*Input SMILES*')

with open(title, 'r') as file:
    content = file.read()


st.title(f'  *{content}* :pill:   ')

tab0 ,tab1, tab2, tab3, tab4, tab5 = st.tabs(['Predict' , 'About', 'Dataset', 'Model', 'Citation', 'Authors'])


with tab0:
    st.write("""### Instructions""")
    st.write('- *Input **SMILES** in the Sidebar* ')
    st.write('- *Press Confirm Button To Generate Predicted **IC50** value*')
    st.write("""## Output""")
    SMILES_input = st.sidebar.text_input(' **Enter Your SMILES Below** ', 'c1ccccc1')
    button = st.sidebar.button('Confirm')
    def model_pred():
        st.write(f" *The SMILES you wanted to predict activity* : {SMILES_input} ")
        if SMILES_input:
            m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

            
            if m is None:
                # The SMILES is invalid
                
                
                st.write(" ### Your **SMILES** is not correct  ")
            else:
                xml_files = glob.glob(xml)
                xml_files.sort() 
                FP_list = ['AtomPairs2DCount',
                    'AtomPairs2D',
                    'EState',
                    'CDKextended',
                    'CDK',
                    'CDKgraphonly',
                    'KlekotaRothCount',
                    'KlekotaRoth',
                    'MACCS',
                    'PubChem',
                    'SubstructureCount',
                    'Substructure']
                
                global fp
                fp = dict(zip(FP_list, xml_files))
                df = pd.DataFrame({'SMILES': [SMILES_input], 'Name' : ['Molecule']} )
                df.to_csv(smilefile, sep='\t', index=False, header=False)
                with open(cfp, 'r') as file:
                    content = file.readline()
                fingerprint = content
                fingerprint_descriptortypes = fp[fingerprint]
                padeldescriptor(mol_dir= smilefile, 
                            d_file=fingerprint_output_file, #'Substructure.csv'
                            #descriptortypes='SubstructureFingerprint.xml', 
                            descriptortypes= fingerprint_descriptortypes,
                            detectaromaticity=True,
                            standardizenitro=True,
                            standardizetautomers=True,
                            threads=2,
                            removesalt=True,
                            log=True,
                            fingerprints=True)
                descriptors = pd.read_csv(fingerprint_output_file)
                X1 = pd.read_csv(xcol)
                R = X1.drop(['IC50'], axis=1)
                X = descriptors.drop(['Name'], axis=1)
                X = descriptors[R.columns]
                
                st.write(' ### Molecular Fingerprint Of Your Structure')
                st.write(X)
                st.write(X.shape)
                model = load(loadm)
                
                pred = model.predict(X)
            
                st.write(f" ### IC50 value : {str(pred)}")


                if SMILES_input:
                    m = Chem.MolFromSmiles(SMILES_input, sanitize=False)

                    
                    if m is None:
                        # The SMILES is invalid
                        
                        
                        st.write(" ### Your **SMILES** is not correct  ")
                    else:
                        try:
                               
                                # SMILES is valid, perform further processing
                                st.write('### Molecular Properties')

                                # Calculate Lipinski properties
                                m = Chem.MolFromSmiles(SMILES_input)
                                NHA = Lipinski.NumHAcceptors(m)
                                NHD = Lipinski.NumHDonors(m)
                                st.write(f'- **Number of Hydrogen Acceptor** : {NHA}')
                                st.write(f'- **Number of Hydrogen Donors** : {NHD}')
                                Molwt = Descriptors.ExactMolWt(m)
                                Molwt = "{:.2f}".format(Molwt)
                                st.write(f'- **Molecular Wieght** : {Molwt}')
                                logP = Crippen.MolLogP(m)
                                logP = "{:.2f}".format(logP)
                                st.write(f'- **LogP** : {logP}')
                                rb = Descriptors.NumRotatableBonds(m)
                                st.write(f'- **Number of Rotatable Bonds** : {rb}')
                                numatom = rdchem.Mol.GetNumAtoms(m)
                                st.write(f'- **Number of Atoms** : {numatom}')
                                mr = Crippen.MolMR(m)
                                mr = "{:.2f}".format(mr)
                                st.write(f'- **Molecular Refractivity** : {mr}')
                                tsam = QED.properties(m).PSA
                                st.write(f'- **Topology Polar Surface Area** : {tsam}')
                                fc = rdmolops.GetFormalCharge(m)
                                st.write(f'- **Formal Charge** : {fc}')
                                ha = rdchem.Mol.GetNumHeavyAtoms(m)
                                st.write(f'- **Number of Heavy Atoms** : {ha}')
                                nr = rdMolDescriptors.CalcNumRings(m)
                                st.write(f'- **Number of Rings** : {nr}')
                                Lipin = "Pass" if (float(Molwt) <= 500 and float(logP) <= 5 and int(NHD) <= 5 and int(NHA) <= 10 and int(rb) <= 5) else "Fail"
                                Ghose = "Pass" if (float(Molwt) >= 160 and float(Molwt) <= 480 and float(logP) >= -0.4 and float(logP) <= 5.6 and int(numatom) >= 20 and int(numatom) <= 70 and float(mr) >= 40 and float(mr) <= 130) else "Fail"
                                veber = "Pass" if (int(rb) <= 10 and float(tsam) <= 140) else "Fail"
                                Ruleof3 = "Pass" if (float(Molwt) <= 300 and float(logP) <= 3 and int(NHD) <= 3 and int(NHA) <= 3 and int(rb) <= 3) else "Fail"
                                Reos = "Pass" if (float(Molwt) >= 200 and float(Molwt) <= 500 and float(logP) >= -5 and float(logP) <= 5 and int(NHD) >= 0 and int(NHD) <= 5 and int(NHA) >= 0 and int(NHA) <= 10 and int(fc) >= -2 and int(fc) <= 2 and int(rb) >= 0 and int(rb) <= 8 and int(ha) >= 15 and int(ha) <= 50) else "Fail"
                                DrugLike = "Pass" if (float(Molwt) < 400 and int(nr) > 0 and int(rb) < 5 and int(NHD) <= 5 and int(NHA) <= 10 and float(logP) < 5) else "Fail"
                                st.write('### Filter Profile')
                                st.write(f"- **Lipinksi Filter** : {Lipin}")
                                st.write(f"- **Ghose Filter** : {Ghose}")
                                st.write(f"- **Veber Filter** : {veber}")
                                st.write(f"- **Ruleof3** : {Ruleof3}")
                                st.write(f"- **Reos Filter** : {Reos}")
                                st.write(f"- **DrugLike Filter** : {DrugLike}")
                                os.remove(smilefile)
                                os.remove(fingerprint_output_file)
                                os.remove(fingerprint_output_file_txt)

                                # Generate 2D coordinates for the molecule
                        
                            
                        except Exception as e:
                                # Handle exceptions during Lipinski property calculation or image saving
                                error_message = "***Error occured due to improper SMILES***"
                                st.write(error_message)
                else:
                # SMILES value is empty or doesn't exist
                    
                    
                    st.write(' ## SMILES not given')



    if button:
        model_pred()
    
with tab1:
    with open(about, 'r') as file:
        content = file.read()

    st.write(content)
    
with tab2:
    with open(data, 'r') as file:
        content = file.read()
    st.write(content)
    st.image(dataimg)
    

with tab3:
    with open(model, 'r') as file:
        content = file.read()
        

    st.write(content)
    st.image(image, caption='Concept of Random Forrest Regressor')

with tab4:
    with open(citation, 'r') as file:
        content = file.read()
    st.write(content)
    with open(citationlink, 'r') as file:
        content = file.read()
    st.link_button('Go To Article Page', url=f'{content}')

with tab5:
    with open(author, 'r') as file:
        content = file.read()
    
    
    st.write(content)
