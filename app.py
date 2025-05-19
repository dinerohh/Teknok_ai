# import streamlit as st
#
# def process_prompt(prompt):
#     # Example: Simple processing (replace with your logic)
#     return f"Processed: {prompt.upper()}"
#
# st.title("We are Teknok")
#
# user_prompt = st.text_area(
#     "Enter your prompt here:",
#     placeholder="Type here...",
#     height=100,  # Adjust height if needed
#     key="prompt_input"
# )
# submit_button = st.button("Submit")
#
# if submit_button:
#     if user_prompt:
#         result = process_prompt(user_prompt)  # Process the prompt if input exists
#         #st.write("Result:", result)
#         #st.write("You entered:", user_prompt)  # Optional: Show the input back to the user
#     else:
#         st.error("Please type something!")  # Show error if input is empty
#


# import streamlit as st
# import pandas as pd
# import joblib
# from datetime import date
#
# # ——— 1) Load artifacts ———
# model = joblib.load('construction_cost_model.pkl')
# training_columns = joblib.load('training_columns.pkl')
# earliest_date = joblib.load('earliest_date.pkl').date()
#
# # ——— 2) Streamlit UI ———
# st.title("We are Teknok 🏗️")
# st.markdown("**Tell me about your material and date, and I'll predict its unit price!**")
#
# # 2a) Inputs matching your features
# material = st.selectbox(
#     "Material name",
#     ["Cement", "Iron Sheets", "Building stones"]
# )
#
# mat_type = st.text_input(
#     "Material type",
#     value="e.g. Portland cement"
# )
#
# retailer = st.text_input(
#     "Retailer",
#     value="e.g. Nairobi Building Supplies"
# )
#
# town = st.text_input(
#     "Town",
#     value="e.g. Nairobi"
# )
#
# # 2b) Date input with min/max guard
# today = date.today()
# min_date = earliest_date if earliest_date <= today else None
#
# collected_on = st.date_input(
#     "Date collected",
#     value=today,
#     min_value=min_date,
#     max_value=today
# )
#
# # ——— 3) Button to trigger prediction ———
# if st.button("Get Price Prediction"):
#     # Basic validation
#     if not (mat_type and retailer and town):
#         st.error("Please fill in all text fields!")
#     else:
#         # ——— 4) Pre‑process inputs ———
#         days_since = (collected_on - earliest_date).days
#         input_dict = {
#             'Days since': days_since,
#             'Material name': material,
#             'Material type': mat_type,
#             'Retailer': retailer,
#             'Town': town
#         }
#         df_input = pd.DataFrame([input_dict])
#
#         # One‑hot encode & align to training columns
#         df_encoded = pd.get_dummies(
#             df_input,
#             columns=['Material name','Material type','Retailer','Town'],
#             drop_first=True
#         )
#         # Add any missing dummy columns
#         for col in training_columns:
#             if col not in df_encoded.columns:
#                 df_encoded[col] = 0
#         # Ensure correct column order
#         df_encoded = df_encoded[training_columns]
#
#         # ——— 5) Predict! ———
#         pred_price = model.predict(df_encoded)[0]
#
#         # ——— 6) Display as a little chat ———
#         st.markdown(f"**You:** What's the unit price of **{material}** "
#                     f"in **{town}** from **{retailer}**, collected on **{collected_on}**?")
#         st.markdown(f"**Teknok:** I estimate the price per unit at **Ksh {pred_price:,.2f}**.")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, date
# from fuzzywuzzy import fuzz
#
# # ——— 1) Load artifacts ———
# try:
#     model = joblib.load('construction_cost_model.pkl')
#     training_columns = joblib.load('training_columns.pkl')
#     earliest_date = joblib.load('earliest_date.pkl').date()
# except FileNotFoundError as e:
#     st.error(f"Missing file: {e}. Please ensure model artifacts are in the directory.")
#     st.stop()
#
# # ——— 2) Load and clean data for combos ———
# try:
#     df_all = pd.read_csv('cleaned_data.csv')
#     df_all['Material name'] = df_all['Material name'].astype(str).str.strip()
#     df_all['Material type'] = df_all['Material type'].astype(str).str.strip().replace('nan', 'Unknown')
#     df_all['Retailer'] = df_all['Retailer'].astype(str).str.strip().replace('nan', 'Unknown')
#     df_all['Town'] = df_all['Town'].astype(str).str.strip().replace('nan', 'Unknown')
# except FileNotFoundError:
#     st.error("Missing 'cleaned_data.csv'. Please generate it from your training script.")
#     st.stop()
#
# # Unique values, sorted longest first
# materials = sorted(df_all['Material name'].unique(), key=len, reverse=True)
# types = sorted(df_all['Material type'].unique(), key=len, reverse=True)
# retailers = sorted(df_all['Retailer'].unique(), key=len, reverse=True)
# towns = sorted(df_all['Town'].unique(), key=len, reverse=True)
#
# # ——— 3) Streamlit UI ———
# st.title("Teknok 🏗️ Price Predictor")
# st.markdown("""
# Type a simple request like “cement” or “iron sheets in Nairobi” to get material price predictions.
# Examples: “cement simba”, “building stones in Machakos”, “iron sheets for 2025-06-01”.
# """)
#
# user_text = st.text_input(
#     "Your request:",
#     placeholder="e.g., cement in Nairobi",
#     key="user_input"
# )
#
# if st.button("Get Price Predictions"):
#     txt = user_text.strip().lower()
#     if not txt:
#         st.error("Please type a material or description (e.g., 'cement').")
#     else:
#         # ——— 4) Fuzzy matching for material ———
#         def find_best_material(text, options, threshold=80):
#             best_score = 0
#             best_match = None
#             for opt in options:
#                 score = fuzz.partial_ratio(text.lower(), opt.lower())
#                 if score > best_score and score >= threshold:
#                     best_score = score
#                     best_match = opt
#             return best_match
#
#         found_material = find_best_material(txt, materials)
#         found_type = next((t for t in types if t.lower() in txt), None)
#         found_retailer = next((r for r in retailers if r.lower() in txt), None)
#         found_town = next((w for w in towns if w.lower() in txt), None)
#
#         if not found_material:
#             st.error(
#                 f"No matching material found. Try one of: {', '.join(materials[:5])}{'…' if len(materials) > 5 else ''} "
#                 f"or check your spelling."
#             )
#         else:
#             # ——— 5) Extract date ———
#             import re
#             date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', txt)
#             if date_match:
#                 try:
#                     selected_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
#                 except ValueError:
#                     selected_date = date.today()
#                     st.warning("Invalid date format. Using today’s date.")
#             else:
#                 selected_date = date.today()
#                 st.info("No date specified. Using today’s date.")
#
#             days_since = (selected_date - earliest_date).days
#
#             # ——— 6) Filter combinations ———
#             combos = (
#                 df_all[df_all['Material name'] == found_material]
#                 [['Material name', 'Material type', 'Retailer', 'Town']]
#                 .drop_duplicates()
#                 .reset_index(drop=True)
#             )
#
#             # Apply filters cautiously
#             applied_filters = []
#             if found_type:
#                 if found_type in combos['Material type'].values:
#                     combos = combos[combos['Material type'] == found_type]
#                     applied_filters.append(f"type '{found_type}'")
#             if found_retailer:
#                 if found_retailer in combos['Retailer'].values:
#                     combos = combos[combos['Retailer'] == found_retailer]
#                     applied_filters.append(f"retailer '{found_retailer}'")
#             if found_town:
#                 if found_town in combos['Town'].values:
#                     combos = combos[combos['Town'] == found_town]
#                     applied_filters.append(f"town '{found_town}'")
#
#             # Fallback to all combos for the material
#             if combos.empty:
#                 combos = (
#                     df_all[df_all['Material name'] == found_material]
#                     [['Material name', 'Material type', 'Retailer', 'Town']]
#                     .drop_duplicates()
#                     .reset_index(drop=True)
#                 )
#                 if applied_filters:
#                     st.warning(
#                         f"The {', '.join(applied_filters)} didn’t match any known options for '{found_material}'. "
#                         f"Showing all available combinations."
#                     )
#
#             if combos.empty:
#                 st.error(f"No valid combinations found for '{found_material}'. Please check your data.")
#             else:
#                 # ——— 7) Predict prices ———
#                 rows = []
#                 for _, row in combos.iterrows():
#                     rows.append({
#                         'Days since': days_since,
#                         'Material name': row['Material name'],
#                         'Material type': row['Material type'],
#                         'Retailer': row['Retailer'],
#                         'Town': row['Town'],
#                     })
#                 df_input = pd.DataFrame(rows)
#
#                 df_enc = pd.get_dummies(
#                     df_input,
#                     columns=['Material name', 'Material type', 'Retailer', 'Town']
#                 )
#                 for c in training_columns:
#                     if c not in df_enc.columns:
#                         df_enc[c] = 0
#                 df_enc = df_enc[training_columns]
#
#                 raw_preds = model.predict(df_enc)
#                 preds = np.rint(raw_preds).astype(int)
#
#                 combos['Predicted Price per Unit (Ksh)'] = preds
#
#                 # ——— 8) Display results ———
#                 st.markdown(f"**Predicted Prices for {selected_date.strftime('%Y-%m-%d')}:**")
#                 st.dataframe(
#                     combos[['Material name', 'Material type', 'Retailer', 'Town', 'Predicted Price per Unit (Ksh)']]
#                     .sort_values('Predicted Price per Unit (Ksh)')
#                     .reset_index(drop=True),
#                     use_container_width=True
#                 )
#                 st.info("Prices are per unit (e.g., per bag for cement). Try adding type, retailer, or town to narrow results.")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date
from fuzzywuzzy import fuzz

# ——— 1) Load artifacts ———
try:
    model = joblib.load('construction_cost_model.pkl')
    training_columns = joblib.load('training_columns.pkl')
    earliest_date = joblib.load('earliest_date.pkl').date()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Please ensure model artifacts are in the directory.")
    st.stop()

# ——— 2) Load and clean data for combos ———
try:
    df_all = pd.read_csv('cleaned_data.csv')
    df_all['Material name'] = df_all['Material name'].astype(str).str.strip()
    df_all['Material type'] = df_all['Material type'].astype(str).str.strip().replace('nan', 'Unknown')
    df_all['Retailer'] = df_all['Retailer'].astype(str).str.strip().replace('nan', 'Unknown')
    df_all['Town'] = df_all['Town'].astype(str).str.strip().replace('nan', 'Unknown')
except FileNotFoundError:
    st.error("Missing 'cleaned_data.csv'. Please generate it from your training script.")
    st.stop()

# Unique values, sorted longest first
materials = sorted(df_all['Material name'].unique(), key=len, reverse=True)
types = sorted(df_all['Material type'].unique(), key=len, reverse=True)
retailers = sorted(df_all['Retailer'].unique(), key=len, reverse=True)
towns = sorted(df_all['Town'].unique(), key=len, reverse=True)

# ——— 3) Streamlit UI ———
st.title("Teknok 🏗️ Price Predictor")
st.markdown("""
Type a simple request like “cement” or “iron sheets in Nairobi” to get material price predictions.  
Examples: “cement simba”, “building stones in Machakos”, “iron sheets for 2025-06-01”.
""")

user_text = st.text_input(
    "Your request:",
    placeholder="e.g., cement in Nairobi",
    key="user_input"
)

if st.button("Get Price Predictions"):
    txt = user_text.strip().lower()
    if not txt:
        st.error("Please type a material or description (e.g., 'cement').")
    else:
        # ——— 4) Fuzzy matching for material ———
        def find_best_material(text, options, threshold=80):
            best_score = 0
            best_match = None
            for opt in options:
                score = fuzz.partial_ratio(text.lower(), opt.lower())
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = opt
            return best_match

        found_material = find_best_material(txt, materials)
        found_type = next((t for t in types if t.lower() in txt), None)
        found_retailer = next((r for r in retailers if r.lower() in txt), None)
        found_town = next((w for w in towns if w.lower() in txt), None)

        if not found_material:
            st.error(
                f"No matching material found. Try one of: {', '.join(materials[:5])}{'…' if len(materials) > 5 else ''} "
                f"or check your spelling."
            )
        else:
            # ——— 5) Extract date ———
            import re
            date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', txt)
            if date_match:
                try:
                    selected_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
                except ValueError:
                    selected_date = date.today()
                    st.warning("Invalid date format. Using today’s date.")
            else:
                selected_date = date.today()  # Silently use today’s date

            days_since = (selected_date - earliest_date).days

            # ——— 6) Filter combinations ———
            combos = (
                df_all[df_all['Material name'] == found_material]
                [['Material name', 'Material type', 'Retailer', 'Town']]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            # Apply filters cautiously
            applied_filters = []
            if found_type:
                if found_type in combos['Material type'].values:
                    combos = combos[combos['Material type'] == found_type]
                    applied_filters.append(f"type '{found_type}'")
            if found_retailer:
                if found_retailer in combos['Retailer'].values:
                    combos = combos[combos['Retailer'] == found_retailer]
                    applied_filters.append(f"retailer '{found_retailer}'")
            if found_town:
                if found_town in combos['Town'].values:
                    combos = combos[combos['Town'] == found_town]
                    applied_filters.append(f"town '{found_town}'")

            # Fallback to all combos
            if combos.empty:
                combos = (
                    df_all[df_all['Material name'] == found_material]
                    [['Material name', 'Material type', 'Retailer', 'Town']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                if applied_filters:
                    st.warning(
                        f"The {', '.join(applied_filters)} didn’t match any known options for '{found_material}'. "
                        f"Showing all available combinations."
                    )

            if combos.empty:
                st.error(f"No valid combinations found for '{found_material}'. Please check your data.")
            else:
                # ——— 7) Predict prices ———
                rows = []
                for _, row in combos.iterrows():
                    rows.append({
                        'Days since': days_since,
                        'Material name': row['Material name'],
                        'Material type': row['Material type'],
                        'Retailer': row['Retailer'],
                        'Town': row['Town'],
                    })
                df_input = pd.DataFrame(rows)

                df_enc = pd.get_dummies(
                    df_input,
                    columns=['Material name', 'Material type', 'Retailer', 'Town']
                )
                for c in training_columns:
                    if c not in df_enc.columns:
                        df_enc[c] = 0
                df_enc = df_enc[training_columns]

                raw_preds = model.predict(df_enc)
                preds = np.rint(raw_preds).astype(int)

                combos['Predicted Price per Unit (Ksh)'] = preds

                # ——— 8) Display results as table ———
                st.markdown(f"**Predicted Prices for {selected_date.strftime('%Y-%m-%d')}:**")
                st.dataframe(
                    combos[['Material name', 'Material type', 'Retailer', 'Town', 'Predicted Price per Unit (Ksh)']]
                    .sort_values('Predicted Price per Unit (Ksh)')
                    .reset_index(drop=True),
                    use_container_width=True
                )
                st.info("Prices are per unit (e.g., per bag for cement). Try adding type, retailer, or town to narrow results.")