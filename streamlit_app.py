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

    # --- Top Artistes et Titres par Année ---
    st.header("🏆 Vos tops avec répartition annuelle")

    # Préparer les données avec l'année
    df_with_year = df_spotify.copy()
    df_with_year['timestamp_dt'] = pd.to_datetime(df_with_year['timestamp'], errors='coerce')
    df_with_year['year'] = df_with_year['timestamp_dt'].dt.year

    # Filtrer les données valides
    df_valid_year = df_with_year[df_with_year['year'].notna()].copy()

    # Créer deux colonnes pour les graphiques
    col_artists, col_tracks = st.columns(2)

    with col_artists:
        st.subheader("Top 10 Artistes par année")
        
        # Calculer le top 10 des artistes
        top_artists = df_valid_year['master_metadata_album_artist_name'].value_counts().head(10).index
        
        # Créer un DataFrame avec le compte par artiste et année
        artist_year_counts = df_valid_year[df_valid_year['master_metadata_album_artist_name'].isin(top_artists)].groupby(
            ['master_metadata_album_artist_name', 'year']
        ).size().reset_index(name='count')
        
        # Pivoter pour avoir les années en colonnes
        artist_pivot = artist_year_counts.pivot(
            index='master_metadata_album_artist_name',
            columns='year',
            values='count'
        ).fillna(0)
        
        # Réordonner selon le total décroissant
        artist_pivot['total'] = artist_pivot.sum(axis=1)
        artist_pivot = artist_pivot.sort_values('total', ascending=True).drop('total', axis=1)
        
        # Créer le graphique à barres empilées
        fig_artists = go.Figure()
        
        # Définir les couleurs par année
        year_colors = {
            2023: '#FF6B6B',  # Rouge/Rose
            2024: '#4ECDC4',  # Turquoise
            2025: '#45B7D1',  # Bleu clair
        }
        
        # Ajouter une trace pour chaque année
        for year in sorted(artist_pivot.columns):
            fig_artists.add_trace(go.Bar(
                name=str(int(year)),
                y=artist_pivot.index,
                x=artist_pivot[year],
                orientation='h',
                marker_color=year_colors.get(year, '#95A5A6'),
                hovertemplate='<b>%{y}</b><br>' +
                              f'Année {int(year)}: %{{x}} écoutes<br>' +
                              '<extra></extra>'
            ))
        
        fig_artists.update_layout(
            barmode='stack',
            title='Total des écoutes par artiste et année',
            xaxis_title='Nombre d\'écoutes',
            yaxis_title='',
            height=500,
            showlegend=True,
            legend=dict(
                title="Année",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=150)  # Plus d'espace pour les noms d'artistes
        )
        
        st.plotly_chart(fig_artists, use_container_width=True)

    with col_tracks:
        st.subheader("Top 10 Titres par année")
        
        # Calculer le top 10 des titres
        top_tracks = df_valid_year['master_metadata_track_name'].value_counts().head(10).index
        
        # Créer un DataFrame avec le compte par titre et année
        track_year_counts = df_valid_year[df_valid_year['master_metadata_track_name'].isin(top_tracks)].groupby(
            ['master_metadata_track_name', 'master_metadata_album_artist_name', 'year']
        ).size().reset_index(name='count')
        
        # Pour l'affichage, combiner titre et artiste
        track_year_counts['display_name'] = track_year_counts['master_metadata_track_name'].str[:30] + '...' + \
                                            ' (' + track_year_counts['master_metadata_album_artist_name'].str[:15] + ')'
        
        # Pivoter pour avoir les années en colonnes
        track_pivot = track_year_counts.pivot_table(
            index='display_name',
            columns='year',
            values='count',
            aggfunc='sum'
        ).fillna(0)
        
        # Réordonner selon le total décroissant
        track_pivot['total'] = track_pivot.sum(axis=1)
        track_pivot = track_pivot.sort_values('total', ascending=True).drop('total', axis=1)
        
        # Créer le graphique à barres empilées
        fig_tracks = go.Figure()
        
        # Ajouter une trace pour chaque année
        for year in sorted(track_pivot.columns):
            fig_tracks.add_trace(go.Bar(
                name=str(int(year)),
                y=track_pivot.index,
                x=track_pivot[year],
                orientation='h',
                marker_color=year_colors.get(year, '#95A5A6'),
                hovertemplate='<b>%{y}</b><br>' +
                              f'Année {int(year)}: %{{x}} écoutes<br>' +
                              '<extra></extra>'
            ))
        
        fig_tracks.update_layout(
            barmode='stack',
            title='Total des écoutes par titre et année',
            xaxis_title='Nombre d\'écoutes',
            yaxis_title='',
            height=500,
            showlegend=True,
            legend=dict(
                title="Année",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=200)  # Plus d'espace pour les titres
        )
        
        st.plotly_chart(fig_tracks, use_container_width=True)

    # Statistiques supplémentaires
    st.subheader("📊 Analyse de la diversité musicale")

    col1, col2, col3, col4 = st.columns(4)

    # Calculer les métriques par année
    years = sorted(df_valid_year['year'].unique())
    current_year = years[-1] if years else None
    previous_year = years[-2] if len(years) > 1 else None

    with col1:
        if current_year:
            current_artists = df_valid_year[df_valid_year['year'] == current_year]['master_metadata_album_artist_name'].nunique()
            st.metric(f"Artistes écoutés en {int(current_year)}", f"{current_artists:,}")

    with col2:
        if previous_year and current_year:
            prev_artists = df_valid_year[df_valid_year['year'] == previous_year]['master_metadata_album_artist_name'].nunique()
            diff = current_artists - prev_artists
            st.metric(f"Évolution vs {int(previous_year)}", 
                      f"{'+' if diff > 0 else ''}{diff}",
                      f"{diff/prev_artists*100:+.1f}%")

    with col3:
        # Artistes découverts cette année
        if current_year and previous_year:
            current_year_artists = set(df_valid_year[df_valid_year['year'] == current_year]['master_metadata_album_artist_name'].unique())
            previous_years_artists = set(df_valid_year[df_valid_year['year'] < current_year]['master_metadata_album_artist_name'].unique())
            new_artists = current_year_artists - previous_years_artists
            st.metric(f"Nouveaux artistes en {int(current_year)}", f"{len(new_artists):,}")

    with col4:
        # Concentration (part du top 10 dans le total)
        if current_year:
            total_plays = len(df_valid_year[df_valid_year['year'] == current_year])
            top10_plays = df_valid_year[df_valid_year['year'] == current_year]['master_metadata_album_artist_name'].value_counts().head(10).sum()
            concentration = top10_plays / total_plays * 100
            st.metric("Concentration Top 10", f"{concentration:.1f}%")

    # Note sur les années disponibles
    with st.expander("ℹ️ Années disponibles dans vos données"):
        years_str = ", ".join([str(int(y)) for y in sorted(df_valid_year['year'].unique())])
        st.write(f"Vos données couvrent les années : **{years_str}**")
        
        # Afficher le nombre d'écoutes par année
        yearly_counts = df_valid_year['year'].value_counts().sort_index()
        for year, count in yearly_counts.items():
            st.write(f"- **{int(year)}** : {count:,} écoutes")

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

    # --- Graphique des heures d'écoute ---
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

    # --- Graphique de valence dans le temps ---
    st.header("😊 Évolution de votre humeur musicale")

    # Préparer les données avec timestamp
    df_valence = df_spotify.copy()
    df_valence['timestamp_dt'] = pd.to_datetime(df_valence['timestamp'], errors='coerce')

    # Filtrer les données avec valence et timestamp valides
    df_valence_valid = df_valence[
        (df_valence['timestamp_dt'].notna()) & 
        (df_valence['valence'].notna())
    ].copy()

    if len(df_valence_valid) > 0:
        # Grouper par mois et calculer la valence moyenne
        df_valence_valid['year_month'] = df_valence_valid['timestamp_dt'].dt.to_period('M')
        valence_monthly = df_valence_valid.groupby('year_month').agg({
            'valence': ['mean', 'std', 'count']
        }).reset_index()
        
        # Aplatir les colonnes multi-index
        valence_monthly.columns = ['year_month', 'valence_mean', 'valence_std', 'count']
        valence_monthly['year_month_str'] = valence_monthly['year_month'].astype(str)
        
        # Créer le graphique principal avec ligne et zone d'écart-type
        fig_valence = go.Figure()
        
        # Ajouter la zone d'écart-type
        fig_valence.add_trace(go.Scatter(
            x=valence_monthly['year_month_str'],
            y=valence_monthly['valence_mean'] + valence_monthly['valence_std'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_valence.add_trace(go.Scatter(
            x=valence_monthly['year_month_str'],
            y=valence_monthly['valence_mean'] - valence_monthly['valence_std'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Écart-type',
            fillcolor='rgba(68, 138, 255, 0.2)',
            hoverinfo='skip'
        ))
        
        # Ajouter la ligne principale
        fig_valence.add_trace(go.Scatter(
            x=valence_monthly['year_month_str'],
            y=valence_monthly['valence_mean'],
            mode='lines+markers',
            name='Valence moyenne',
            line=dict(color='rgb(68, 138, 255)', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>' +
                          'Valence moyenne: %{y:.3f}<br>' +
                          '<extra></extra>'
        ))
        
        # Ajouter une ligne de référence à 0.5
        fig_valence.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Neutre (0.5)",
            annotation_position="right"
        )
        
        # Personnaliser le layout
        fig_valence.update_layout(
            title="Évolution de la valence musicale (0 = triste, 1 = joyeux)",
            xaxis_title="Mois",
            yaxis_title="Valence moyenne",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Rotation des labels x si nécessaire
        if len(valence_monthly) > 20:
            fig_valence.update_xaxis(tickangle=-45)
        
        st.plotly_chart(fig_valence, use_container_width=True)
        
        # Statistiques complémentaires
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_valence = df_valence_valid['valence'].mean()
            st.metric("Valence moyenne globale", f"{avg_valence:.3f}")
            
        with col2:
            # Trouver le mois le plus joyeux
            happiest_month = valence_monthly.loc[valence_monthly['valence_mean'].idxmax()]
            st.metric("Mois le plus joyeux", 
                      happiest_month['year_month_str'],
                      f"{happiest_month['valence_mean']:.3f}")
        
        with col3:
            # Trouver le mois le plus triste
            saddest_month = valence_monthly.loc[valence_monthly['valence_mean'].idxmin()]
            st.metric("Mois le plus mélancolique", 
                      saddest_month['year_month_str'],
                      f"{saddest_month['valence_mean']:.3f}")
        
        with col4:
            # Tendance récente (3 derniers mois vs 3 mois d'avant)
            if len(valence_monthly) >= 6:
                recent_avg = valence_monthly.tail(3)['valence_mean'].mean()
                previous_avg = valence_monthly.iloc[-6:-3]['valence_mean'].mean()
                trend = recent_avg - previous_avg
                st.metric("Tendance récente", 
                          "↑ Plus joyeux" if trend > 0 else "↓ Plus mélancolique",
                          f"{abs(trend):.3f}")
            else:
                st.metric("Tendance récente", "Pas assez de données", "—")
        
        # Graphique secondaire : Distribution de la valence
        st.subheader("Distribution de la valence")
        
        # Créer l'histogramme
        fig_dist = px.histogram(
            df_valence_valid, 
            x='valence',
            nbins=50,
            title="Répartition des morceaux par valence",
            labels={'valence': 'Valence', 'count': 'Nombre de morceaux'},
            color_discrete_sequence=['rgb(68, 138, 255)']
        )
        
        # Ajouter des zones colorées pour l'interprétation
        fig_dist.add_vrect(x0=0, x1=0.3, fillcolor="red", opacity=0.1, 
                           annotation_text="Mélancolique", annotation_position="top")
        fig_dist.add_vrect(x0=0.3, x1=0.7, fillcolor="yellow", opacity=0.1,
                           annotation_text="Neutre", annotation_position="top")
        fig_dist.add_vrect(x0=0.7, x1=1, fillcolor="green", opacity=0.1,
                           annotation_text="Joyeux", annotation_position="top")
        
        fig_dist.update_layout(
            xaxis=dict(range=[0, 1]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Explication
        with st.expander("ℹ️ Qu'est-ce que la valence ?"):
            st.write("""
            **La valence** est une mesure Spotify de 0 à 1 qui décrit la positivité musicale d'un morceau :
            
            - **0.0 - 0.3** : Morceaux tristes, mélancoliques, déprimés ou en colère
            - **0.3 - 0.7** : Morceaux neutres ou ambivalents
            - **0.7 - 1.0** : Morceaux joyeux, euphoriques, heureux
            
            Elle est calculée en analysant le tempo, le mode (majeur/mineur), le timbre et d'autres caractéristiques acoustiques.
            """)
            
    else:
        st.warning("⚠️ Pas assez de données avec valence pour créer ce graphique.")