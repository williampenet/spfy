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

    # Préparer les données de date
    df_copy = df_spotify.copy()

    # Utiliser timestamp en priorité, sinon played_date
    if 'timestamp' in df_copy.columns:
        df_copy['date_parsed'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
    else:
        df_copy['date_parsed'] = pd.to_datetime(df_copy['played_date'], errors='coerce')

    # Filtrer uniquement les données avec des dates valides
    df_valid_dates = df_copy[df_copy['date_parsed'].notna()].copy()

    if len(df_valid_dates) > 0:
        # Grouper par mois
        ecoutes_par_mois = df_valid_dates.groupby(df_valid_dates['date_parsed'].dt.to_period('M')).size().reset_index()
        ecoutes_par_mois.columns = ['Mois', 'Nombre d\'écoutes']
        ecoutes_par_mois['Mois'] = ecoutes_par_mois['Mois'].astype(str)
        
        # Créer le graphique
        fig_evolution = px.bar(ecoutes_par_mois, 
                              x='Mois', 
                              y='Nombre d\'écoutes',
                              title='Nombre d\'écoutes par mois',
                              color='Nombre d\'écoutes',
                              color_continuous_scale='Sunset')
        fig_evolution.update_layout(showlegend=False,
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_evolution, use_container_width=True)
    else:
        st.error("Aucune donnée avec des dates valides trouvée.")

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
                         color_continuous_scale='Sunset',
                         text='Nombre de titres')
    
    # Personnaliser l'apparence
    fig_duration.update_traces(texttemplate='%{text}', textposition='outside')
    fig_duration.update_layout(showlegend=False,
                              yaxis_title='Nombre de titres',
                              xaxis_title='Tranches de durée')
    
    st.plotly_chart(fig_duration, use_container_width=True)

# Remplacer toute la section "Graphique des heures d'écoute"
# À partir de : st.header("🕐 Répartition par heure de la journée")

st.header("🕐 Répartition par heure de la journée")

# Convertir timestamp en datetime
df_spotify['timestamp_dt'] = pd.to_datetime(df_spotify['timestamp'], errors='coerce')

# Extraire l'heure et le jour de la semaine directement depuis timestamp
df_spotify['hour'] = df_spotify['timestamp_dt'].dt.hour
df_spotify['weekday'] = df_spotify['timestamp_dt'].dt.dayofweek
df_spotify['date_only'] = df_spotify['timestamp_dt'].dt.date

# Mapper les numéros de jour aux noms
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df_spotify['weekday_name'] = df_spotify['weekday'].map(day_names)

# Filtrer les lignes avec timestamp valide
df_valid_dates = df_spotify[df_spotify['timestamp_dt'].notna()].copy()

# Afficher le nombre de lignes utilisées
st.info(f"📊 Utilisation de {len(df_valid_dates):,} lignes sur {len(df_spotify):,} ({len(df_valid_dates)/len(df_spotify)*100:.1f}%) pour la heatmap")

# Compter le nombre d'occurrences de chaque jour de la semaine
days_count = df_valid_dates.groupby(['weekday', 'date_only']).size().reset_index()
occurrences_per_day = days_count.groupby('weekday').size().reset_index(name='occurrences')
occurrences_dict = dict(zip(occurrences_per_day['weekday'], occurrences_per_day['occurrences']))

# Afficher les statistiques de couverture des données
st.subheader("📊 Couverture des données")
col1, col2, col3 = st.columns(3)
with col1:
    if len(df_valid_dates) > 0:
        st.metric("Période couverte", 
                  f"{df_valid_dates['timestamp_dt'].min().strftime('%Y-%m-%d')} à {df_valid_dates['timestamp_dt'].max().strftime('%Y-%m-%d')}")
    else:
        st.metric("Période couverte", "Pas de dates valides")
with col2:
    st.metric("Jours avec données", 
              f"{df_valid_dates['date_only'].nunique()} jours")
with col3:
    # Afficher le jour le moins représenté
    if occurrences_dict:
        min_day = min(occurrences_dict.items(), key=lambda x: x[1])
        st.metric("Jour le moins couvert", 
                  f"{day_names[min_day[0]]}: {min_day[1]} occurrences")
    else:
        st.metric("Jour le moins couvert", "N/A")

# Calculer les écoutes par jour/heure
if len(df_valid_dates) == 0:
    st.warning("⚠️ Aucune donnée valide trouvée pour créer la heatmap.")
else:
    heatmap_data = df_valid_dates.groupby(['weekday_name', 'weekday', 'hour']).size().reset_index(name='count')

    # Ajouter le nombre d'occurrences pour chaque jour
    heatmap_data['day_occurrences'] = heatmap_data['weekday'].map(occurrences_dict)

    # Calculer la vraie moyenne (écoutes / nombre réel d'occurrences du jour)
    heatmap_data['avg_count'] = heatmap_data['count'] / heatmap_data['day_occurrences']

    # Créer un DataFrame complet avec toutes les heures (0-23) et tous les jours
    all_hours = range(24)
    all_days = [(name, num) for num, name in day_names.items()]

    complete_index = pd.MultiIndex.from_product([
        [d[0] for d in all_days],  # weekday_name
        [d[1] for d in all_days],  # weekday number
        all_hours  # hour
    ], names=['weekday_name', 'weekday', 'hour'])

    complete_df = pd.DataFrame(index=complete_index).reset_index()

    # Merger avec les données réelles
    heatmap_data = complete_df.merge(
        heatmap_data[['weekday_name', 'weekday', 'hour', 'avg_count']], 
        on=['weekday_name', 'weekday', 'hour'], 
        how='left'
    ).fillna({'avg_count': 0})

    # Pivoter pour la heatmap
    heatmap_pivot = heatmap_data.pivot_table(
        index='hour',
        columns='weekday',
        values='avg_count',
        fill_value=0
    )

    # Réordonner les colonnes
    heatmap_pivot = heatmap_pivot.reindex(columns=range(7))

    # Créer la heatmap
    fig_heure = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Day", y="Hour", color="Avg plays"),
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        y=[f"{h:02d}h" for h in range(24)],
        color_continuous_scale='Blues',
        origin='lower',
        title="Weekly listening patterns: average plays per occurrence",
        aspect='auto'
    )

    # Ajouter des annotations pour montrer la couverture
    annotations = []
    for i, day in enumerate(range(7)):
        if day in occurrences_dict:
            annotations.append(
                dict(
                    text=f"{occurrences_dict[day]}x",
                    x=i,
                    y=-1.5,
                    xref="x",
                    yref="y",
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
            )

    fig_heure.update_layout(
        height=800,
        xaxis_title="Day of the week (number shows data coverage)",
        yaxis_title="Hour of the day",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}h" for h in range(24)]
        ),
        annotations=annotations
    )

    st.plotly_chart(fig_heure, use_container_width=True)

    # Note sur le fuseau horaire
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.write("""
        - **Les couleurs** indiquent le nombre moyen de morceaux écoutés à chaque heure
        - **Les nombres sous chaque jour** (ex: "45x") indiquent combien de fois ce jour apparaît dans vos données
        - **La moyenne est calculée** en divisant le nombre total d'écoutes par le nombre réel d'occurrences de ce jour
        - **Note**: Les heures sont en UTC (temps universel). Pour la France, ajoutez +1h en hiver, +2h en été.
        
        Par exemple : Si vous avez 90 écoutes le lundi à 19h et que vous avez 45 lundis dans vos données, 
        la moyenne sera 90 ÷ 45 = 2 morceaux par lundi à 19h.
        """)