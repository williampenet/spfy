# Fichier : streamlit_app.py

# --- Étape 1 : Importer les bibliothèques nécessaires ---
import streamlit as st
import pandas as pd # On importe pandas pour la manipulation des données
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration de la page (DOIT être en premier) ---
st.set_page_config(page_title="Mon Spotistory", layout="wide", initial_sidebar_state="collapsed")

# --- Étape 2 : Initialiser la connexion à Supabase ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase_client: Client = create_client(supabase_url, supabase_key)
except KeyError as e:
    st.error(f"ERREUR : Les secrets Supabase ne sont pas configurés. Clé manquante : {e}")
    st.stop()
except Exception as e:
    st.error(f"ERREUR : Impossible de se connecter à Supabase. Détails : {str(e)}")
    st.stop()

# --- Étape 3 : Définir la fonction de chargement (ne change pas) ---
@st.cache_data
def load_spotify_data(_db_client: Client):
    """Charge les données depuis la table 'tracks' de Supabase."""
    response = _db_client.table('tracks').select('*').execute()
    # On convertit directement les données en DataFrame pandas
    return pd.DataFrame(response.data)

# --- Étape 4 : Construire l'interface de l'application ---

st.title("🎵 Mon Historique Spotify")

# --- On charge les données (ne change pas) ---
df_spotify = load_spotify_data(supabase_client)

# Si le chargement a échoué ou si la table est vide, on arrête
if df_spotify.empty:
    st.warning("Aucune donnée à afficher. La table 'tracks' est peut-être vide.")
else:
    # --- DÉBUT DE LA NOUVELLE PARTIE : ANALYSE ET VISUALISATION ---

    st.markdown("### Explorez vos habitudes d'écoute musicale avec des visualisations interactives")
    st.markdown("---")

    # --- Indicateurs clés (KPIs) ---
    st.header("📊 Statistiques générales")
    total_morceaux = len(df_spotify)
    # Assurez-vous que le nom de la colonne 'artistName' est correct
    total_artistes = df_spotify['master_metadata_album_artist_name'].nunique()
    total_albums = df_spotify['master_metadata_album_album_name'].nunique()
    temps_total_heures = df_spotify['ms_played'].sum() / (1000 * 60 * 60)  # Convertir en heures

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎵 Morceaux écoutés", f"{total_morceaux:,}")
    col2.metric("🎤 Artistes uniques", f"{total_artistes:,}")
    col3.metric("💿 Albums uniques", f"{total_albums:,}")
    col4.metric("⏰ Temps total", f"{temps_total_heures:.0f}h")

    # --- Top 5 Artistes ---
    st.header("🏆 Vos artistes préférés")
    # Assurez-vous que le nom de la colonne 'artistName' est correct
    top_5_artistes = df_spotify['master_metadata_album_artist_name'].value_counts().head(5).reset_index()
    top_5_artistes.columns = ['Artiste', 'Nombre d\'écoutes']
    
    # Créer un graphique horizontal avec Plotly
    fig_artistes = px.bar(top_5_artistes, 
                         x='Nombre d\'écoutes', 
                         y='Artiste',
                         orientation='h',
                         title='Top 5 des artistes les plus écoutés',
                         color='Nombre d\'écoutes',
                         color_continuous_scale='Sunset',
                         text='Nombre d\'écoutes')
    
    fig_artistes.update_traces(texttemplate='%{text}', textposition='outside')
    fig_artistes.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_artistes, use_container_width=True)

    # --- Graphique des écoutes dans le temps ---
    st.header("📈 Évolution de vos écoutes")
    # On convertit la colonne de date au bon format avec un format explicite
    df_spotify['played_date'] = pd.to_datetime(df_spotify['played_date'], format='%Y-%m-%d', errors='coerce')
    # On compte les écoutes par jour
    ecoutes_par_mois = df_spotify.groupby(df_spotify['played_date'].dt.to_period('M')).size().reset_index()
    ecoutes_par_mois.columns = ['Mois', 'Nombre d\'écoutes']
    ecoutes_par_mois['Mois'] = ecoutes_par_mois['Mois'].astype(str)
    
    # Créer le graphique avec Plotly
    fig_evolution = px.bar(ecoutes_par_mois, 
                          x='Mois', 
                          y='Nombre d\'écoutes',
                          title='Nombre d\'écoutes par mois',
                          color='Nombre d\'écoutes',
                          color_continuous_scale='Blues')
    fig_evolution.update_layout(showlegend=False)
    st.plotly_chart(fig_evolution, use_container_width=True)

    # --- Graphique de répartition par durée ---
    st.header("⏱️ Répartition des titres par durée")
    
    # Convertir ms_played en minutes
    df_spotify['duration_minutes'] = df_spotify['ms_played'] / 60000
    
    # Créer les catégories de durée
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = ['< 5 min', '5-10 min', '10-15 min', '15-20 min', '> 20 min']
    df_spotify['duration_category'] = pd.cut(df_spotify['duration_minutes'], 
                                              bins=bins, 
                                              labels=labels, 
                                              right=False)
    
    # Compter le nombre de titres par catégorie
    duration_counts = df_spotify['duration_category'].value_counts().sort_index().reset_index()
    duration_counts.columns = ['Tranche de durée', 'Nombre de titres']
    
    # Créer le graphique avec Plotly
    fig_duration = px.bar(duration_counts, 
                         x='Tranche de durée', 
                         y='Nombre de titres',
                         title='Distribution des titres écoutés par durée',
                         color='Nombre de titres',
                         color_continuous_scale='Viridis',
                         text='Nombre de titres')
    
    # Personnaliser l'apparence
    fig_duration.update_traces(texttemplate='%{text}', textposition='outside')
    fig_duration.update_layout(showlegend=False,
                              yaxis_title='Nombre de titres',
                              xaxis_title='Tranches de durée')
    
    st.plotly_chart(fig_duration, use_container_width=True)

    # --- Graphique des heures d'écoute ---
    st.header("🕐 Répartition par heure de la journée")
    
    # Extraire l'heure et le jour de la semaine avec formats explicites
    df_spotify['hour'] = pd.to_datetime(df_spotify['played_time'], format='%H:%M:%S', errors='coerce').dt.hour
    df_spotify['weekday'] = pd.to_datetime(df_spotify['played_date'], format='%Y-%m-%d', errors='coerce').dt.dayofweek
    
    # Mapper les numéros de jour aux noms en anglais
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df_spotify['weekday_name'] = df_spotify['weekday'].map(day_names)
    
    # Calculer la moyenne des écoutes par jour de semaine et heure
    heatmap_data = df_spotify.groupby(['weekday_name', 'weekday', 'hour']).size().reset_index(name='count')
    
    # Créer un DataFrame complet avec toutes les heures (0-23) et tous les jours
    all_hours = range(24)
    all_days = [(name, num) for num, name in day_names.items()]
    
    # Créer toutes les combinaisons possibles
    complete_index = pd.MultiIndex.from_product([
        [d[0] for d in all_days],  # weekday_name
        [d[1] for d in all_days],  # weekday number
        all_hours  # hour
    ], names=['weekday_name', 'weekday', 'hour'])
    
    # Créer un DataFrame avec toutes les combinaisons
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    
    # Merger avec les données réelles
    heatmap_data = complete_df.merge(
        heatmap_data, 
        on=['weekday_name', 'weekday', 'hour'], 
        how='left'
    ).fillna({'count': 0})
    
    # Calculer la moyenne par jour/heure (nombre total d'écoutes / nombre de semaines)
    n_weeks = df_spotify['played_date'].nunique() / 7  # Approximation du nombre de semaines
    heatmap_data['avg_count'] = heatmap_data['count'] / n_weeks
    
    # Pivoter pour avoir les jours en colonnes et les heures en lignes
    heatmap_pivot = heatmap_data.pivot_table(
        index='hour',
        columns='weekday',
        values='avg_count',
        fill_value=0
    )
    
    # Réordonner les colonnes pour avoir lundi en premier
    heatmap_pivot = heatmap_pivot.reindex(columns=range(7))
    
    # Créer la heatmap avec Plotly
    fig_heure = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Day", y="Hour", color="Avg plays"),
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        y=[f"{h:02d}h" for h in range(24)],  # Toutes les heures de 00h à 23h
        color_continuous_scale='Blues',
        origin='lower',  # Pour avoir 0h en bas
        title="Weekly listening patterns: average plays by day and hour",
        aspect='auto'  # Ajuste automatiquement l'aspect ratio
    )
    
    # Personnaliser l'apparence
    fig_heure.update_layout(
        height=800,  # Plus de hauteur
        xaxis_title="Day of the week",
        yaxis_title="Hour of the day",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}h" for h in range(24)]
        )
    )
    
    st.plotly_chart(fig_heure, use_container_width=True)