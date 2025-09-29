import pandas as pd
import numpy as np
import os, json
from langchain.schema import Document

folder_path = r"YOUR PATH TO SYNTHIA DATA FILES"

#selecting most important datafiles and creating dataframes for each

encounters = pd.read_csv(os.path.join(folder_path, 'encounters.csv'))
patients = pd.read_csv(os.path.join(folder_path, 'patients.csv'))
conditions = pd.read_csv(os.path.join(folder_path, 'conditions.csv'))
medications = pd.read_csv(os.path.join(folder_path, 'medications.csv'))
observations = pd.read_csv(os.path.join(folder_path, 'observations.csv'))
procedures = pd.read_csv(os.path.join(folder_path, 'procedures.csv'))
allergies = pd.read_csv(os.path.join(folder_path, 'allergies.csv'))
immunizations = pd.read_csv(os.path.join(folder_path, 'immunizations.csv'))

# Rename columns for clarity
patients = patients.rename(columns={"Id": "patient_id", "BIRTHDATE": "birthdate", "SSN":"ssn","GENDER": "gender", "FIRST":"first", "LAST":"last"})
encounters = encounters.rename(columns={"Id": "encounter_id", "PATIENT": "patient_id", "START": "start", "STOP": "end", "ENCOUNTERCLASS": "class"})
conditions = conditions.rename(columns={"PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "DESCRIPTION": "condition"})
medications = medications.rename(columns={"PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "DESCRIPTION": "medication"})
observations = observations.rename(columns={"PATIENT": "patient_id", "ENCOUNTER": "encounter_id", "DESCRIPTION": "observation", "VALUE": "value", "UNITS": "units"})
procedures = procedures.rename(columns={"START":"start_date","STOP":"stop_date","PATIENT": "patient_id","ENCOUNTER": "encounter_id","DESCRIPTION":"procedure","CODE":"procedure_code"})
allergies=allergies.rename(columns={"START":"start_date","STOP":"stop_date","PATIENT": "patient_id","ENCOUNTER": "encounter_id","CATEGORY":"category","DESCRIPTION":"main_allergy",
                                    "DESCRIPTION1":"allergy1","DESCRIPTION2":"allergy2","SEVERITY1":"severity1","SEVERITY2":"severity2"})
immunizations = immunizations.rename(columns={"PATIENT":"patient_id", "ENCOUNTER":"encounter_id", "DESCRIPTION":"immunization_description"})

# Merging Patients and Encounters dataframes
encounters_merged = encounters.merge(patients[["patient_id","birthdate","ssn","gender","first","last"]], how='left', on='patient_id')

output_dir = r"YOUR OUTPUT DIRECTORY FOR PATIENT NOTES - patient notes"

def format_bullets(items):
    if not items:
        return "None"
    return "\n- " + "\n- ".join(items)

# Make folder if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_docs = []

for pat_id, patient_rows in encounters_merged.groupby("patient_id"):
    patient_info = patient_rows.iloc[0]
    full_name = f"{patient_info['first']} {patient_info['last']}"
    birthdate = patient_info['birthdate']
    gender = patient_info['gender']
    ssn = patient_info['ssn']

    encounter_texts = []

    for _, row in patient_rows.iterrows():
        enc_id = row["encounter_id"]
        enc_class = row["class"]
        enc_start = row["start"]
        enc_end = row["end"] if pd.notna(row["end"]) else "Ongoing"

        cond = conditions[conditions["encounter_id"] == enc_id]["condition"].dropna().tolist()
        meds = medications[medications["encounter_id"] == enc_id]["medication"].dropna().tolist()
        obser = observations[observations["encounter_id"] == enc_id][["observation", "value", "units"]].dropna(subset=["observation", "value"])
        proc = procedures[procedures["encounter_id"] == enc_id][["procedure", "procedure_code"]].dropna(subset=["procedure"])
        allg = allergies[allergies["encounter_id"] == enc_id][["main_allergy","allergy1", "allergy2", "severity1", "severity2"]].dropna(subset=["main_allergy"])
        immu = immunizations[immunizations["encounter_id"] == enc_id]["immunization_description"].dropna().tolist()

        obser_texts = [f"{o['observation']}: {o['value']} {o['units'] if pd.notna(o['units']) else ''}".strip() for _, o in obser.iterrows()]
        proc_texts = [f"{p['procedure']}: code {p['procedure_code'] if pd.notna(p['procedure_code']) else ''}".strip() for _, p in proc.iterrows()]

        allg_texts = []
        for _, a in allg.iterrows():
            if pd.notna(a["allergy1"]):
                allg_texts.append(f"{a['allergy1']}: {a['severity1'] if pd.notna(a['severity1']) else ''}")
            if pd.notna(a["allergy2"]):
                allg_texts.append(f"{a['allergy2']}: {a['severity2'] if pd.notna(a['severity2']) else ''}")

        encounter_texts.append(f"""\
            Patient: {full_name}, Gender: {gender}, DOB: {birthdate}, SSN: {ssn}
            
            Encounter: {enc_class}, from {enc_start} to {enc_end}
            
            Conditions: {format_bullets(cond)}
            Medications: {format_bullets(meds)}
            Observations: {format_bullets(obser_texts)}
            Procedures: {format_bullets(proc_texts)}
            Allergies: {format_bullets(allg_texts)}
            Immunization: {format_bullets(immu)}
            """)

        # Full document
        encounters_summary = "\n\n".join(encounter_texts)
    
        full_document = f"""Patient: {full_name}, Gender: {gender}, DOB: {birthdate}
            {"-"*60}
            {encounters_summary}
            """
    
        # Write file
        filename = f"{pat_id}_{patient_info['first']}_{patient_info['last']}.txt"
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(full_document)
    
        # Add to list for vectorstore
        all_docs.append(Document(
            page_content=full_document,
            metadata={"patient_name": full_name, "patient_id": pat_id}
        ))

