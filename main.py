##Codice da utilizzare per la dashboard Streamlit. Si lancia con il notebook "Streamlit"


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

from prophet import Prophet       
from sklearn.cluster import KMeans

# Caricamento dei dataset
df_fuel = pd.read_csv('/content/drive/MyDrive/Tesi/EnergyDecription.csv')
df = pd.read_csv('/content/drive/MyDrive/Tesi/WorldConsumption_Prepdataset.csv')

# File JSON per salvare i preferiti
favorites_file = '/content/drive/MyDrive/Tesi/preferiti.json'

#---FUNZIONI
# Funzione per leggere i preferiti dal file JSON
def leggi_preferiti(favorites_file):
    try:
        with open(favorites_file, 'r') as json_file:
            preferiti = json.load(json_file)
        return preferiti["preferiti"]  # Restituisce solo la lista dei preferiti
    except FileNotFoundError:
        return []  # Se il file non esiste, ritorna una lista vuota

# Funzione per eliminare un preferito
def elimina_preferito(favorites_file, preferito_da_eliminare):
    preferiti = leggi_preferiti(favorites_file)
    # Rimuovi il preferito dalla lista
    preferiti = [preferito for preferito in preferiti if preferito['nome preferito'] != preferito_da_eliminare]
    # Scrivi la lista aggiornata nel file
    scrivi_preferiti(favorites_file, preferiti)

def scrivi_preferiti(favorites_file, preferiti):
    """
    Salva i preferiti in un file JSON, sovrascrivendo il contenuto esistente.

    :param file_path: Percorso del file JSON.
    :param preferiti: Lista di preferiti da salvare.
    """
    try:
        with open(favorites_file, 'w', encoding='utf-8') as json_file:
            json.dump({"preferiti": preferiti}, json_file, indent=4, ensure_ascii=False)
        print(f"Preferiti salvati con successo in {favorites_file}")
    except Exception as e:
        print(f"Errore nel salvataggio dei preferiti: {e}")



# Estrai la lista unica dei paesi dalla colonna 'country'
paesi_disponibili = df['country'].unique().tolist()

# Leggi i preferiti dal file JSON
preferiti = leggi_preferiti(favorites_file)

# Titolo della pagina
st.title("Dashboard Consumi Globali")


# Crea un menu a tendina per scegliere l'azione sui preferiti
azione_preferito = st.selectbox(
    "Scegli un'azione",
    ["Aggiungi un nuovo preferito", "Elimina un preferito", "Seleziona un preferito esistente","Inserisci dati manualmente"]
)


# cancella preferiti
if azione_preferito == "Elimina un preferito":
   # Visualizza i preferiti già presenti
  if len(preferiti) > 0:
      st.write("### I tuoi preferiti salvati:")
      for preferito in preferiti:
          st.write(f"**{preferito['nome preferito']}** - Paese: {', '.join(preferito['paese'])}, Anni: {preferito['anno da']} - {preferito['anno a']}, Energia: {preferito['energia']}")
  else:
      st.write("Non ci sono preferiti da cancellare.")
      
  if len(preferiti) > 0:
      preferiti_names = [preferito['nome preferito'] for preferito in preferiti]
      selected_preferito = st.selectbox("Scegli un preferito da cancellare", preferiti_names)

      if st.button("Elimina Preferito"):
          elimina_preferito(favorites_file, selected_preferito)
          st.success(f"Preferito '{selected_preferito}' eliminato con successo!")
          st.rerun()  # Ricarica la pagina per vedere i cambiamenti
  else:
      st.warning("Non ci sono preferiti salvati.")


# Crea un nuovo preferito se l'utente lo vuole
if azione_preferito== "Aggiungi un nuovo preferito":
    nome_preferito = st.text_input("Nome Preferito")
    paesi_input = st.multiselect("Seleziona il Paese", paesi_disponibili)    
    anno_da = st.number_input("Anno da", min_value=1900, max_value=2022)
    anno_a = st.number_input("Anno a", min_value=1900, max_value=2022)
    energia = st.selectbox("Tipo di Energia", ["biofuel", "coal", "gas", "hydro", "nuclear", "oil", "other_renewable", "solar", "wind"])
#si potrebbe pensare qui di far vedere le descrizioni delle energie contenute del dataset df_fuel
    if st.button("Salva Preferito"):
        if not paesi_input:
            st.warning("Devi selezionare almeno un paese.")      
        nuovo_preferito = {
            "nome preferito": nome_preferito,
            "paese": paesi_input,
            "anno da": anno_da,
            "anno a": anno_a,
            "energia": energia
        }
        
        # Aggiungi il nuovo preferito alla lista
        #preferiti["preferiti"].append(nuovo_preferito) CONTROLALRE QUALE è GIUSTO
        preferiti.append(nuovo_preferito)
        
        # Salva i preferiti aggiornati nel file
        scrivi_preferiti(favorites_file,preferiti)
        
        st.success(f"Preferito aggiunto: {nome_preferito}, Paese: {', '.join(paesi_input)}, Anni: {anno_da}-{anno_a}, Energia: {energia}")

        # Ricarica la pagina per vedere i cambiamenti
        st.rerun()

# Crea un menu a tendina per selezionare un preferito dalla lista

if azione_preferito == "Seleziona un preferito esistente":
  # Visualizza i preferiti già presenti
  if len(preferiti) > 0:
      st.write("### I tuoi preferiti salvati:")
      for preferito in preferiti:
          st.write(f"**{preferito['nome preferito']}** - Paese: {', '.join(preferito['paese'])}, Anni: {preferito['anno da']} - {preferito['anno a']}, Energia: {preferito['energia']}")
  else:
      st.write("Non ci sono preferiti salvati.")

  if len(preferiti) > 0:
      preferito_selezionato = st.selectbox("Scegli un preferito", [preferito["nome preferito"] for preferito in preferiti])

      if preferito_selezionato:
          selected_preference = next(preferito for preferito in preferiti if preferito["nome preferito"] == preferito_selezionato)
          
         
          # Estrai le variabili dal preferito selezionato
          selected_countries = selected_preference['paese']
          anno_da_selezionato = selected_preference['anno da']
          anno_a_selezionato = selected_preference['anno a']
          selected_fuel = selected_preference['energia']
          selected_years = (anno_da_selezionato, anno_a_selezionato)

          #stampa variabili per controllo
          #st.write(paesi_selezionati)
          #st.write(selected_years)
          #st.write(energia_selezionata)

          # Mostra i dettagli del preferito scelto
          paesi_str = ", ".join(selected_countries)
          st.write(f"**{selected_preference['nome preferito']}**: Paesi: {paesi_str}, Anni: {anno_da_selezionato}-{anno_a_selezionato}, Energia: {selected_fuel}")


          # Crea il dataset filtrato in base ai parametri del preferito
          filtered_df = df[(df['country'].isin(selected_countries)) & 
                         (df['fuel'] == selected_fuel) & 
                        (df['year'].between(selected_years[0], selected_years[1]))]
          st.dataframe(filtered_df)

  else:
      st.write("Non ci sono preferiti da selezionare.")


#SCELTE UTENTE
if azione_preferito == "Inserisci dati manualmente":
  # Selezione multipla di Paesi
  countries = df['country'].unique()
  selected_countries = st.multiselect("Seleziona uno o più Paesi", countries)
  #st.write(selected_countries)

  # Selezione del tipo di fonte energetica
  fuels = df['fuel'].unique()
  selected_fuel = st.selectbox("Seleziona il Tipo di Energia", fuels)

  # Visualizzazione delle informazioni sul carburante selezionato
  fuel_description = df_fuel[df_fuel['fuel'] == selected_fuel]['description'].values[0]
  st.write(f"### Descrizione di {selected_fuel.capitalize()}")
  st.write(fuel_description)

  # Filtro per intervallo di anni
  min_year = int(df['year'].min())  # Corretto: variabile min_year
  max_year = int(df['year'].max())
  selected_years = st.slider("Seleziona intervallo di anni", min_year, max_year, (min_year, max_year))



  # Filtro per intervallo di anni
  filtered_df = df[(df['country'].isin(selected_countries)) & 
                  (df['fuel'] == selected_fuel) & 
                  (df['year'].between(selected_years[0], selected_years[1]))]
  #stampo variabili per controllo
  #st.write(selected_countries)
  #st.write(f"Tipo di anno_da_selezionato: {type(selected_years)}")

if azione_preferito != "Elimina un preferito" and azione_preferito != "Aggiungi un nuovo preferito":
  # 1. Analisi della Crescita - Tasso di crescita della produzione energetica
  filtered_df_growth = filtered_df.groupby('country')['production'].agg(['min', 'max']).reset_index()
  filtered_df_growth['growth_rate'] = (filtered_df_growth['max'] - filtered_df_growth['min']) / filtered_df_growth['min'] * 100
  st.write("### Tasso di Crescita della Produzione Energetica")
  st.dataframe(filtered_df_growth[['country', 'growth_rate']])


  # 2. Redditività dell'Investimento - Confronto tra produzione e PIL (normalizzati)
  if 'normalized_production_per_gdp' in df.columns:
      # Filtraggio dei dati per Redditività dell'Investimento
      gdp_comparison = filtered_df.groupby('country').agg({'normalized_production': 'sum', 'normalized_gdp': 'sum', 'normalized_production_per_gdp': 'sum'}).reset_index()

      st.write("### Redditività dell'Investimento (Produzione / PIL Normalizzati)")
      st.dataframe(gdp_comparison[['country', 'normalized_production_per_gdp']])
  else:
      st.write("### Redditività dell'Investimento")
      st.warning("Non è possibile calcolare la Redditività dell'Investimento, poiché i dati normalizzati non sono disponibili.")

  # 3. Matrice di Correlazione - Analizzare la correlazione tra variabili
  correlation_df = df[(df['country'].isin(selected_countries)) & 
                      (df['fuel'] == selected_fuel) & 
                      (df['year'].between(selected_years[0], selected_years[1]))]

  # Verifica se le colonne esistono
  columns_to_check = ['production', 'gdp', 'population', 'per_capita']
  existing_columns = [col for col in columns_to_check if col in correlation_df.columns]

  # Se le colonne esistono, calcola la matrice di correlazione
  if len(existing_columns) == len(columns_to_check):
      correlation_df = correlation_df[existing_columns]
      corr_matrix = correlation_df.corr()

      st.write("### Matrice di Correlazione")
      
      # Creazione della figura per il grafico
      fig, ax = plt.subplots(figsize=(10, 8))  # Dimensioni personalizzabili
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
      
      # Visualizza il grafico
      st.pyplot(fig)
  else:
      st.write("### Matrice di Correlazione")
      st.warning("Alcune delle colonne richieste non sono disponibili nel dataset. Non è possibile calcolare la matrice di correlazione.")

  # 4. Ranking Paesi - Classificare per crescita nella produzione energetica
  ranking_df = filtered_df_growth.sort_values(by='growth_rate', ascending=False)
  st.write("### Ranking dei Paesi per Crescita nella Produzione Energetica")
  st.dataframe(ranking_df[['country', 'growth_rate']])

  # 5. Scenario Futuro - Previsione della Produzione Energetica
  st.write("### Scenario Futuro - Previsione della Produzione Energetica")
  country_for_prediction = st.selectbox('Seleziona il Paese per la previsione', selected_countries)

  # Filtrare i dati per il paese selezionato
  prediction_df = df[(df['country'] == country_for_prediction) & 
                    (df['fuel'] == selected_fuel) & 
                    (df['year'].between(selected_years[0], selected_years[1]))]

  # Previsione con regressione lineare
  X = prediction_df['year'].values.reshape(-1, 1)
  y = prediction_df['production'].values
  model = LinearRegression()
  model.fit(X, y)

  # Previsione per gli anni futuri
  future_years = np.array(range(selected_years[1] + 1, selected_years[1] + 6)).reshape(-1, 1)
  predictions = model.predict(future_years)

  # Visualizzazione del grafico delle previsioni
  fig_pred = px.line(x=future_years.flatten(), y=predictions, labels={'x': 'Anno', 'y': 'Produzione Energetica'},
                    title=f"Previsione della Produzione Energetica Futura per {country_for_prediction}")
  st.plotly_chart(fig_pred)

  # Grafico dell'andamento della produzione nel tempo
  fig = px.line(filtered_df, x='year', y='production', color='country',
                title=f"Andamento Produzione di {selected_fuel} in Paesi selezionati")
  st.plotly_chart(fig)

  # Grafico della produzione pro capite
  fig2 = px.line(filtered_df, x='year', y='per_capita', color='country',
                title=f"Produzione Pro Capite di {selected_fuel} in Paesi selezionati")
  st.plotly_chart(fig2)

  # Mappa della produzione energetica globale
  if 'iso_code' in df.columns:
      map_df = df[(df['fuel'] == selected_fuel) & df['year'].between(selected_years[0], selected_years[1])]
      fig_map = px.choropleth(map_df, locations='iso_code', color='production',
                              hover_name='country', title=f"Produzione di {selected_fuel} nel Mondo",
                              color_continuous_scale='viridis', projection='natural earth')
      st.plotly_chart(fig_map)

    # --- Grafico a barre della produzione totale per Paese
  fig_bar = px.bar(
      filtered_df.groupby('country', as_index=False).agg({'production': 'sum'}),
      x='country', 
      y='production',
      color='country',
      title="Produzione Totale per Paese"
  )
  st.plotly_chart(fig_bar)

  # --- Grafico a dispersione (scatter plot) Produzione vs PIL con dimensione = popolazione
  # Assicurati che la colonna 'gdp' e 'population' siano disponibili nel dataset filtrato
  if 'gdp' in filtered_df.columns and 'population' in filtered_df.columns:
      fig_scatter = px.scatter(
          filtered_df,
          x='gdp',
          y='production',
          size='population',
          color='country',
          title="Produzione vs PIL (dimensione = Popolazione)",
          hover_data=['country']
      )
      st.plotly_chart(fig_scatter)
  else:
      st.write("Colonne 'gdp' o 'population' non disponibili per il grafico scatter.")

  # --- Box Plot della distribuzione della produzione per Tipo di Energia
  fig_box = px.box(
      filtered_df,
      x='fuel',
      y='production',
      title="Distribuzione della Produzione per Tipo di Energia"
  )
  st.plotly_chart(fig_box)

  # --- Grafico Radar per confrontare variabili chiave per ogni Paese
  # Raggruppa per paese e calcola i totali/medie delle variabili interessate
  radar_data = filtered_df.groupby('country').agg({
      'production': 'sum',
      'gdp': 'sum',
      'population': 'mean',
      'per_capita': 'mean'
  }).reset_index()

  # Seleziona le categorie da mostrare (potresti normalizzarle se i range sono molto diversi)
  categories = ['production', 'gdp', 'population', 'per_capita']

  # Indicatori chiave
  st.write("### Indicatori Chiave")
  st.metric("Produzione Totale", f"{filtered_df['production'].sum():,.2f} MWh")
  st.metric("Produzione Media", f"{filtered_df['production'].mean():,.2f} MWh")
  st.metric("Massima Produzione", f"{filtered_df['production'].max():,.2f} MWh")



  import plotly.graph_objects as go
  fig_radar = go.Figure()
  for _, row in radar_data.iterrows():
      fig_radar.add_trace(go.Scatterpolar(
          r=[row[cat] for cat in categories],
          theta=categories,
          fill='toself',
          name=row['country']
      ))
  fig_radar.update_layout(
      polar=dict(
          radialaxis=dict(
              visible=True,
              # Se i valori sono molto diversi, potresti voler normalizzare o usare un range adatto
              # range=[0, max(radar_data[categories].max())] 
          )
      ),
      showlegend=True,
      title="Confronto Variabili per Paese"
  )
  st.plotly_chart(fig_radar)



  # Tabella con i dati filtrati
  st.write("### Dati Filtrati")
  st.dataframe(filtered_df)

  # Download dei dati filtrati
  csv = filtered_df.to_csv(index=False).encode('utf-8')
  st.download_button("Scarica CSV", csv, "dati_filtrati.csv", "text/csv")
