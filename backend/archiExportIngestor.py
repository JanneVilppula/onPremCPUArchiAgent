import pandas as pd
pd.set_option('display.max_columns', None)

elements_csv_path = 'elements.csv'
relations_csv_path = 'relations.csv'

try:
    elements_df = pd.read_csv(elements_csv_path)
    relations_df = pd.read_csv(relations_csv_path)
except FileNotFoundError as e:
    print(f"Error loading {e}")
    exit()

merged_relations_df = pd.merge(
    relations_df,
    elements_df,
    left_on='Source',
    right_on='ID',
    how='left',
    suffixes=('', '_Source')
)
merged_relations_df = pd.merge(
    merged_relations_df,
    elements_df,
    left_on='Target',
    right_on='ID',
    how='left',
    suffixes=('', '_Target')
)

final_relations_df = merged_relations_df[[
    'ID',
    'Type',
    'Documentation',
    'ID_Source',
    'Type_Source',
    'Name_Source',
    'Documentation_Source',
    'ID_Target',
    'Type_Target',
    'Name_Target',
    'Documentation_Target',
]]

facts = []

for index, row  in elements_df.iterrows():
    fact_description = f"The {row['Name']} ({row['Type']}) is described as: '{row['Documentation']}'"
    facts.append({
        'id': row['ID'],
        'type': 'element_description',
        'text': fact_description
    })

for index, row in final_relations_df.iterrows():
    relation_doc = row['Documentation'] if pd.notna(row['Documentation']) else ""
    fact_relation = (
        f"The {row['Name_Source']} ({row['Type_Source']}) has {row['Type']} with {row['Name_Target']} ({row['Type_Target']})"
        f"The relation is described as: '{relation_doc}'"
    )
    facts.append({
        'id': row['ID'],
        'type': 'relation_description',
        'text': fact_relation
    })

for index, row in elements_df.iterrows():
    element_documentation = row['Documentation'] if pd.notna(row['Documentation']) else ""
    related_info = []

    outgoing_relations = final_relations_df[final_relations_df['ID_Source'] == row['ID']]
    for index, o_row in outgoing_relations.iterrows():
        related_info.append(f"It {o_row['Type']} the {o_row['Name_Target']} ({o_row['Type_Target']}).")

    incoming_relations = final_relations_df[final_relations_df['ID_Target'] == row['ID']]
    for index, i_row in incoming_relations.iterrows():
        related_info.append(f"It is {i_row['Type']} by the {i_row['Name_Source']} ({i_row['Type_Source']}).")
    
    fact_complex = (
        f"The {row['Name']} ({row['Type']}) is described as: '{row['Documentation']}'. "
        f"{row['Name']} is also involved in connections: " + " ".join(related_info)
    )
    facts.append({
        'id': f"fact-element-{row['ID']}",
        'type': 'element-relation_description',
        'text': fact_complex
    })

print(f"\n--- Generated {len(facts)} Facts for Vector DB ---")
for fact in facts:
    print(f"ID: {fact['id']}\nText: {fact['text']}\nType: {fact['type']}\n---")