# File: streamlit_app.py

# --- Step 1: Import necessary libraries ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page configuration (MUST be first) ---
st.set_page_config(page_title="My Spotistory", layout="centered", initial_sidebar_state="collapsed")

# --- Helper function for number formatting ---
def format_number(num):
    """Format a number with spaces as thousand separators"""
    return f"{num:,}".replace(",", " ")

# --- Step 2: Initialize Supabase connection ---
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase_client: Client = create_client(supabase_url, supabase_key)
except KeyError as e:
    st.error(f"ERROR: Supabase secrets are not configured. Missing key: {e}")
    st.stop()
except Exception as e:
    st.error(f"ERROR: Unable to connect to Supabase. Details: {str(e)}")
    st.stop()

# --- Step 3: Define data loading function ---
@st.cache_data
def load_spotify_data(_db_client: Client):
    """Load data from 'tracks' table in Supabase."""
    response = _db_client.table('tracks').select('*').execute()
    return pd.DataFrame(response.data)

# --- Step 4: Build the application interface ---

st.title("🎵 William's Spotify Personal Insights")

# --- Load data ---
df_spotify = load_spotify_data(supabase_client)

# If loading failed or table is empty, stop
if df_spotify.empty:
    st.warning("No data to display. 'tracks' table may be empty.")
else:
    # --- START OF ANALYSIS AND VISUALIZATION ---

    st.markdown("### Explore William's music listening habits with some data visualizations")
    st.markdown("---")

    # --- Key Performance Indicators (KPIs) ---
    st.header("📊 Global Stats")
    total_tracks = len(df_spotify)
    total_artists = df_spotify['master_metadata_album_artist_name'].nunique()
    total_albums = df_spotify['master_metadata_album_album_name'].nunique()
    total_hours = df_spotify['ms_played'].sum() / (1000 * 60 * 60)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎵 Tracks Listened", format_number(total_tracks))
    col2.metric("🎤 Unique Artists", format_number(total_artists))
    col3.metric("💿 Unique Albums", format_number(total_albums))
    col4.metric("⏰ Total Hours", f"{total_hours:.0f} h")

    # --- Top Artists and Tracks by Year ---
    st.header("🏆 Tops")

    # Prepare data with year
    df_with_year = df_spotify.copy()
    df_with_year['timestamp_dt'] = pd.to_datetime(df_with_year['timestamp'], errors='coerce')
    df_with_year['year'] = df_with_year['timestamp_dt'].dt.year

    # Filter valid data
    df_valid_year = df_with_year[df_with_year['year'].notna()].copy()

    # Define colors by year
    year_colors = {
        2010: '#fff2b2',  # Jaune ambré doux
        2011: '#fcefa1', # Soleil pâle
        2012: '#fcd584', # Abricot doux
        2013: '#fab07b', # Pêche dorée
        2014: '#f98a7f', # Corail tendre
        2015: '#f26489', # Rose melon
        2016: '#de4f94', # Framboise vive
        2017: '#c84a97', # Orchidée
        2018: '#a649a3', # Mauve profond
        2019: '#814ea7', # Violet prune
        2020: '#6249a2', # Indigo pastel
        2021: '#4c4798', # Bleu nuit doux
        2022: '#474f9b', # Ardoise violette
        2023: '#5669b3', # Lavande bleutée
        2024: '#6c87c8', # Bleu pervenche
        2025: '#85a7d8', # Azur givré
    }

    # --- Top 10 Artists ---
    st.subheader("🎤 Most Listened to artists")
    
    # Calculate top 10 artists
    top_artists = df_valid_year['master_metadata_album_artist_name'].value_counts().head(10).index
    
    # Create DataFrame with count by artist and year
    artist_year_counts = df_valid_year[df_valid_year['master_metadata_album_artist_name'].isin(top_artists)].groupby(
        ['master_metadata_album_artist_name', 'year']
    ).size().reset_index(name='count')
    
    # Pivot to have years as columns
    artist_pivot = artist_year_counts.pivot(
        index='master_metadata_album_artist_name',
        columns='year',
        values='count'
    ).fillna(0)
    
    # Reorder by descending total
    artist_pivot['total'] = artist_pivot.sum(axis=1)
    artist_pivot = artist_pivot.sort_values('total', ascending=True).drop('total', axis=1)
    
    # Create stacked bar chart
    fig_artists = go.Figure()
    
    # Add trace for each year
    for year in sorted(artist_pivot.columns):
        fig_artists.add_trace(go.Bar(
            name=str(int(year)),
            y=artist_pivot.index,
            x=artist_pivot[year],
            orientation='h',
            marker_color=year_colors.get(year, '#95A5A6'),
            hovertemplate='<b>%{y}</b><br>' +
                          f'Year {int(year)}: %{{x}} plays<br>' +
                          '<extra></extra>'
        ))
    
    fig_artists.update_layout(
        barmode='stack',
        title='Total plays by artist and year',
        xaxis_title='Plays',
        yaxis_title='',
        height=500,
        showlegend=True,
        legend=dict(
            title="Year",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=150)
    )
    
    st.plotly_chart(fig_artists, use_container_width=True)

    # --- Top 10 Tracks ---
    st.subheader("🎵 Most Played Tracks")
    
    # Calculate top 10 tracks
    top_tracks = df_valid_year['master_metadata_track_name'].value_counts().head(10).index
    
    # Create DataFrame with count by track and year
    track_year_counts = df_valid_year[df_valid_year['master_metadata_track_name'].isin(top_tracks)].groupby(
        ['master_metadata_track_name', 'master_metadata_album_artist_name', 'year']
    ).size().reset_index(name='count')
    
    # For display, combine track and artist (without automatic "...")
    track_year_counts['display_name'] = track_year_counts.apply(
        lambda row: f"{row['master_metadata_track_name'][:40]} ({row['master_metadata_album_artist_name'][:20]})"
        if len(row['master_metadata_track_name']) > 40 
        else f"{row['master_metadata_track_name']} ({row['master_metadata_album_artist_name'][:20]})",
        axis=1
    )
    
    # Pivot to have years as columns
    track_pivot = track_year_counts.pivot_table(
        index='display_name',
        columns='year',
        values='count',
        aggfunc='sum'
    ).fillna(0)
    
    # Reorder by descending total
    track_pivot['total'] = track_pivot.sum(axis=1)
    track_pivot = track_pivot.sort_values('total', ascending=True).drop('total', axis=1)
    
    # Create stacked bar chart
    fig_tracks = go.Figure()
    
    # Add trace for each year
    for year in sorted(track_pivot.columns):
        fig_tracks.add_trace(go.Bar(
            name=str(int(year)),
            y=track_pivot.index,
            x=track_pivot[year],
            orientation='h',
            marker_color=year_colors.get(year, '#95A5A6'),
            hovertemplate='<b>%{y}</b><br>' +
                          f'Year {int(year)}: %{{x}} plays<br>' +
                          '<extra></extra>'
        ))
    
    fig_tracks.update_layout(
        barmode='stack',
        title='Total plays by track and year',
        xaxis_title='Plays',
        yaxis_title='',
        height=500,
        showlegend=True,
        legend=dict(
            title="Year",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=200)
    )
    
    st.plotly_chart(fig_tracks, use_container_width=True)

    st.markdown("---")

    # --- Listening evolution over time ---
    st.header("📈 Listening evolution")

    # Prepare date data
    df_copy = df_spotify.copy()

    # Use timestamp first, then played_date
    if 'timestamp' in df_copy.columns:
        df_copy['date_parsed'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
    else:
        df_copy['date_parsed'] = pd.to_datetime(df_copy['played_date'], errors='coerce')

    # Filter only data with valid dates
    df_valid_dates = df_copy[df_copy['date_parsed'].notna()].copy()

    if len(df_valid_dates) > 0:
        # Group by month
        listens_per_month = df_valid_dates.groupby(df_valid_dates['date_parsed'].dt.to_period('M')).size().reset_index()
        listens_per_month.columns = ['Month', 'Number of Plays']
        listens_per_month['Month'] = listens_per_month['Month'].astype(str)
        
        # Identify gaps (more than 3 months without data)
        listens_per_month['Month_dt'] = pd.to_datetime(listens_per_month['Month'])
        listens_per_month = listens_per_month.sort_values('Month_dt')
        listens_per_month['Gap'] = listens_per_month['Month_dt'].diff() > pd.Timedelta(days=90)
        
        # Create chart with gap management
        fig_evolution = go.Figure()
        
        # Split data by continuous periods
        current_group = []
        groups = []
        
        for idx, row in listens_per_month.iterrows():
            if row['Gap'] and current_group:
                groups.append(pd.DataFrame(current_group))
                current_group = []
            current_group.append(row)
        
        if current_group:
            groups.append(pd.DataFrame(current_group))
        
        # Plot each group separately
        for i, group in enumerate(groups):
            fig_evolution.add_trace(go.Bar(
                x=group['Month'],
                y=group['Number of Plays'],
                name=f'Period {i+1}',
                marker_color='rgb(68, 138, 255)',
                showlegend=False
            ))
        
        fig_evolution.update_layout(
            title='Number of Plays per Month',
            xaxis_title='Month',
            yaxis_title='Number of Plays',
            bargap=0.2,
            height=400
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
    else:
        st.error("No data with valid dates found.")

    # --- Track duration distribution (in percentage) ---
    st.header("⏱️ Track duration distribution")
    
    # Convert ms_played to minutes
    df_spotify['duration_minutes'] = df_spotify['ms_played'] / 60000
    
    # Create duration categories
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = ['< 5 min', '5-10 min', '10-15 min', '15-20 min', '> 20 min']
    df_spotify['duration_category'] = pd.cut(df_spotify['duration_minutes'], 
                                              bins=bins, 
                                              labels=labels, 
                                              right=False)
    
    # Calculate percentages
    duration_counts = df_spotify['duration_category'].value_counts(normalize=True).sort_index() * 100
    duration_df = pd.DataFrame({
        'Duration Range': duration_counts.index,
        'Percentage': duration_counts.values
    })
    
    # Create chart
    fig_duration = px.bar(duration_df, 
                         x='Duration Range', 
                         y='Percentage',
                         title='Distribution of Played Tracks by Duration (%)',
                         color='Percentage',
                         color_continuous_scale='Sunset',
                         text=duration_df['Percentage'].round(1))
    
    # Customize appearance
    fig_duration.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_duration.update_layout(showlegend=False,
                              yaxis_title='Percentage of Tracks',
                              xaxis_title='Duration Ranges')
    
    st.plotly_chart(fig_duration, use_container_width=True)

    # --- Listening hours heatmap ---
    st.header("🕐 Listening patterns by hour")

    # Convert timestamp to datetime
    df_spotify['timestamp_dt'] = pd.to_datetime(df_spotify['timestamp'], errors='coerce')

    # Extract hour and day of week from timestamp
    df_spotify['hour'] = df_spotify['timestamp_dt'].dt.hour
    df_spotify['weekday'] = df_spotify['timestamp_dt'].dt.dayofweek
    df_spotify['date_only'] = df_spotify['timestamp_dt'].dt.date

    # Map day numbers to names
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df_spotify['weekday_name'] = df_spotify['weekday'].map(day_names)

    # Filter rows with valid timestamp
    df_valid_dates = df_spotify[df_spotify['timestamp_dt'].notna()].copy()

    # Count occurrences of each day of week
    days_count = df_valid_dates.groupby(['weekday', 'date_only']).size().reset_index()
    occurrences_per_day = days_count.groupby('weekday').size().reset_index(name='occurrences')
    occurrences_dict = dict(zip(occurrences_per_day['weekday'], occurrences_per_day['occurrences']))

    # Calculate plays by day/hour
    if len(df_valid_dates) == 0:
        st.warning("⚠️ No valid data found to create heatmap.")
    else:
        heatmap_data = df_valid_dates.groupby(['weekday_name', 'weekday', 'hour']).size().reset_index(name='count')

        # Add number of occurrences for each day
        heatmap_data['day_occurrences'] = heatmap_data['weekday'].map(occurrences_dict)

        # Calculate true average (plays / actual occurrences of day)
        heatmap_data['avg_count'] = heatmap_data['count'] / heatmap_data['day_occurrences']

        # Create complete DataFrame with all hours (0-23) and all days
        all_hours = range(24)
        all_days = [(name, num) for num, name in day_names.items()]

        complete_index = pd.MultiIndex.from_product([
            [d[0] for d in all_days],  # weekday_name
            [d[1] for d in all_days],  # weekday number
            all_hours  # hour
        ], names=['weekday_name', 'weekday', 'hour'])

        complete_df = pd.DataFrame(index=complete_index).reset_index()

        # Merge with actual data
        heatmap_data = complete_df.merge(
            heatmap_data[['weekday_name', 'weekday', 'hour', 'avg_count']], 
            on=['weekday_name', 'weekday', 'hour'], 
            how='left'
        ).fillna({'avg_count': 0})

        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot_table(
            index='hour',
            columns='weekday',
            values='avg_count',
            fill_value=0
        )

        # Reorder columns
        heatmap_pivot = heatmap_pivot.reindex(columns=range(7))

        # Create heatmap
        fig_hour = px.imshow(
            heatmap_pivot.values,
            labels=dict(x="Day", y="Hour", color="Avg plays"),
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=[f"{h:02d}h" for h in range(24)],
            color_continuous_scale='Blues',
            origin='lower',
            title="Weekly Listening Patterns: Average Plays per Occurrence",
            aspect='auto'
        )

        # Add annotations showing coverage
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

        fig_hour.update_layout(
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

        st.plotly_chart(fig_hour, use_container_width=True)

    # --- Musical mood evolution ---
    st.header("😊 Musical mood evolution")

    # Prepare data with timestamp
    df_valence = df_spotify.copy()
    df_valence['timestamp_dt'] = pd.to_datetime(df_valence['timestamp'], errors='coerce')

    # Filter data with valid valence and timestamp
    df_valence_valid = df_valence[
        (df_valence['timestamp_dt'].notna()) & 
        (df_valence['valence'].notna())
    ].copy()

    if len(df_valence_valid) > 0:
        # Group by month and calculate average valence
        df_valence_valid['year_month'] = df_valence_valid['timestamp_dt'].dt.to_period('M')
        valence_monthly = df_valence_valid.groupby('year_month').agg({
            'valence': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        valence_monthly.columns = ['year_month', 'valence_mean', 'valence_std', 'count']
        valence_monthly['year_month_str'] = valence_monthly['year_month'].astype(str)
        
        # Create main chart with line and standard deviation area
        fig_valence = go.Figure()
        
        # Add standard deviation area
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
            name='Standard Deviation',
            fillcolor='rgba(68, 138, 255, 0.2)',
            hoverinfo='skip'
        ))
        
        # Add main line
        fig_valence.add_trace(go.Scatter(
            x=valence_monthly['year_month_str'],
            y=valence_monthly['valence_mean'],
            mode='lines+markers',
            name='Average Valence',
            line=dict(color='rgb(68, 138, 255)', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>' +
                          'Average Valence: %{y:.3f}<br>' +
                          '<extra></extra>'
        ))
        
        # Add reference line at 0.5
        fig_valence.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Neutral (0.5)",
            annotation_position="right"
        )
        
        # Customize layout
        fig_valence.update_layout(
            title="Musical valence evolution (0 = sad, 1 = happy)",
            xaxis_title="Month",
            yaxis_title="Average Valence",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Rotate x labels if necessary (fixed method)
        if len(valence_monthly) > 20:
            fig_valence.update_layout(xaxis=dict(tickangle=-45))
        
        st.plotly_chart(fig_valence, use_container_width=True)
        
        # Complementary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_valence = df_valence_valid['valence'].mean()
            st.metric("Overall Average Valence", f"{avg_valence:.3f}")
            
        with col2:
            # Find happiest month
            happiest_month = valence_monthly.loc[valence_monthly['valence_mean'].idxmax()]
            st.metric("Happiest Month", 
                      happiest_month['year_month_str'],
                      f"{happiest_month['valence_mean']:.3f}")
        
        with col3:
            # Find saddest month
            saddest_month = valence_monthly.loc[valence_monthly['valence_mean'].idxmin()]
            st.metric("Most Melancholic Month", 
                      saddest_month['year_month_str'],
                      f"{saddest_month['valence_mean']:.3f}")
        
        
        # Explanation
        with st.expander("ℹ️ What is Valence?"):
            st.write("""
            **Valence** is a Spotify measure from 0 to 1 describing the musical positivity of a track:
            
            - **0.0 - 0.3**: Sad, melancholic, depressed, or angry tracks
            - **0.3 - 0.7**: Neutral or ambivalent tracks
            - **0.7 - 1.0**: Happy, euphoric, cheerful tracks
            
            It's calculated by analyzing tempo, mode (major/minor), timbre, and other acoustic features.
            """)
            
    else:
        st.warning("⚠️ Not enough data with valence to create this chart.")

    # --- Discovery Score - Musical Exploration Index ---
    st.header("🧭 Discovery Score - William's musical exploration index")

    # Prepare data
    df_discovery = df_spotify.copy()
    df_discovery['timestamp_dt'] = pd.to_datetime(df_discovery['timestamp'], errors='coerce')
    df_discovery = df_discovery[df_discovery['timestamp_dt'].notna()].copy()

    if len(df_discovery) > 0:
        # Add time periods
        df_discovery['date'] = df_discovery['timestamp_dt'].dt.date
        df_discovery['week'] = df_discovery['timestamp_dt'].dt.to_period('W')
        df_discovery['month'] = df_discovery['timestamp_dt'].dt.to_period('M')
        df_discovery['year_month'] = df_discovery['timestamp_dt'].dt.strftime('%Y-%m')
        
        # Calculate metrics by month
        monthly_stats = []
        
        for month in df_discovery['month'].unique():
            month_data = df_discovery[df_discovery['month'] == month]
            
            # Basic metrics
            total_plays = len(month_data)
            unique_artists = month_data['master_metadata_album_artist_name'].nunique()
            unique_tracks = month_data['master_metadata_track_name'].nunique()
            unique_albums = month_data['master_metadata_album_album_name'].nunique()
            
            # Identify new discoveries
            previous_months = df_discovery[df_discovery['month'] < month]
            previous_artists = set(previous_months['master_metadata_album_artist_name'].unique())
            previous_tracks = set(previous_months['master_metadata_track_name'].unique())
            
            current_artists = set(month_data['master_metadata_album_artist_name'].unique())
            current_tracks = set(month_data['master_metadata_track_name'].unique())
            
            new_artists = len(current_artists - previous_artists)
            new_tracks = len(current_tracks - previous_tracks)
            
            # Calculate repetition rate (obsessive loops)
            track_play_counts = month_data['master_metadata_track_name'].value_counts()
            obsessive_loops = len(track_play_counts[track_play_counts > 20])  # Tracks played >20 times
            max_repeat = track_play_counts.max() if len(track_play_counts) > 0 else 0
            
            # Genre diversity (approximated by artist diversity)
            artist_distribution = month_data['master_metadata_album_artist_name'].value_counts()
            # Shannon entropy to measure diversity
            artist_probs = artist_distribution / artist_distribution.sum()
            diversity_score = -np.sum(artist_probs * np.log(artist_probs + 1e-10))
            
            # Calculate Discovery Score
            discovery_score = (
                (new_artists * 3 + new_tracks * 1) / (total_plays + 1) * 100 * 2 +
                diversity_score * 5 -
                (obsessive_loops * 2)
            )
            discovery_score = max(0, min(100, discovery_score))
            
            monthly_stats.append({
                'month': month,
                'month_str': str(month),
                'total_plays': total_plays,
                'unique_artists': unique_artists,
                'unique_tracks': unique_tracks,
                'new_artists': new_artists,
                'new_tracks': new_tracks,
                'discovery_score': discovery_score,
                'obsessive_loops': obsessive_loops,
                'max_repeat': max_repeat,
                'diversity_score': diversity_score
            })
        
        df_monthly_stats = pd.DataFrame(monthly_stats).sort_values('month')
        
        # 1. MAIN GAUGE - Current Discovery Score
        current_month_stats = df_monthly_stats.iloc[-1]
        current_score = current_month_stats['discovery_score']
        
        # Determine category
        if current_score < 30:
            category = "Creature of Habit 🔄"
            color = "red"
        elif current_score < 60:
            category = "Balanced Explorer ⚖️"
            color = "yellow"
        else:
            category = "Musical Adventurer 🧭"
            color = "green"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Circular gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_score,
                title={'text': f"Current Score<br><span style='font-size:0.8em'>{category}</span>"},
                delta={'reference': df_monthly_stats.iloc[-2]['discovery_score'] if len(df_monthly_stats) > 1 else current_score},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "gray"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Current month statistics
            st.subheader(f"📊 {current_month_stats['month_str']} Stats")
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("New Artists Discovered", f"{current_month_stats['new_artists']} 🎤")
                st.metric("New Tracks Discovered", f"{current_month_stats['new_tracks']} 🎵")
            with subcol2:
                st.metric("Unique Artists Played", format_number(current_month_stats['unique_artists']))
                st.metric("Obsessive Loops", f"{current_month_stats['obsessive_loops']} tracks")
        
        # 2. TIMELINE - Exploration vs Comfort
        st.subheader("📈 Exploration vs Comfort Over Time")
        
        fig_timeline = go.Figure()
        
        # New discoveries line
        fig_timeline.add_trace(go.Scatter(
            x=df_monthly_stats['month_str'],
            y=df_monthly_stats['new_artists'],
            name='New Artists',
            line=dict(color='green', width=3),
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # New tracks line
        fig_timeline.add_trace(go.Scatter(
            x=df_monthly_stats['month_str'],
            y=df_monthly_stats['new_tracks'],
            name='New Tracks',
            line=dict(color='lightgreen', width=2, dash='dot'),
            mode='lines+markers'
        ))
        
        # Global score line
        fig_timeline.add_trace(go.Scatter(
            x=df_monthly_stats['month_str'],
            y=df_monthly_stats['discovery_score'],
            name='Discovery Score',
            line=dict(color='blue', width=3),
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig_timeline.update_layout(
            xaxis_title="Month",
            yaxis_title="New Discoveries",
            yaxis2=dict(
                title="Discovery Score",
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        
        # 4. INSIGHTS AND ACHIEVEMENTS
        st.subheader("🏆 Discovery Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Best exploration month
            best_month = df_monthly_stats.loc[df_monthly_stats['discovery_score'].idxmax()]
            st.info(f"""
            **Best Discovery Month**  
            {best_month['month_str']}  
            Score: {best_month['discovery_score']:.0f}/100  
            {best_month['new_artists']} new artists discovered!
            """)
        
        with col2:
            # Biggest obsession
            obsession_month = df_monthly_stats.loc[df_monthly_stats['max_repeat'].idxmax()]
            most_played = df_discovery[df_discovery['month'] == obsession_month['month']]['master_metadata_track_name'].value_counts().iloc[0]
            st.warning(f"""
            **Biggest Obsession**  
            {obsession_month['month_str']}  
            One track played {obsession_month['max_repeat']} times!  
            That's {obsession_month['max_repeat']/30:.1f} times per day!
            """)
        
        with col3:
            # Overall trend
            recent_trend = df_monthly_stats.tail(3)['discovery_score'].mean() - df_monthly_stats.head(3)['discovery_score'].mean()
            trend_emoji = "📈" if recent_trend > 0 else "📉"
            st.success(f"""
            **Overall Trend {trend_emoji}**  
            {'+' if recent_trend > 0 else ''}{recent_trend:.1f} points  
            You're {'expanding' if recent_trend > 0 else 'consolidating'} your musical horizons!
            """)
        
    else:
        st.warning("⚠️ Not enough data to calculate discovery score.")