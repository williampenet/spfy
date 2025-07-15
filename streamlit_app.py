# File: streamlit_app.py

# --- Step 1: Import necessary libraries ---
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go

# --- Page configuration (MUST be first) ---
st.set_page_config(page_title="My Spotistory", layout="centered", initial_sidebar_state="collapsed")

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

st.title("🎵 My Spotify Personal Insights")

# --- Load data ---
df_spotify = load_spotify_data(supabase_client)

# If loading failed or table is empty, stop
if df_spotify.empty:
    st.warning("No data to display. 'tracks' table may be empty.")
else:
    # --- START OF ANALYSIS AND VISUALIZATION ---

    st.markdown("### Explore your music listening habits with interactive visualizations")
    st.markdown("---")

    # --- Key Performance Indicators (KPIs) ---
    st.header("📊 Global Stats")
    total_tracks = len(df_spotify)
    total_artists = df_spotify['master_metadata_album_artist_name'].nunique()
    total_albums = df_spotify['master_metadata_album_album_name'].nunique()
    total_hours = df_spotify['ms_played'].sum() / (1000 * 60 * 60)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎵 Tracks Listened", f"{total_tracks:,}")
    col2.metric("🎤 Unique Artists", f"{total_artists:,}")
    col3.metric("💿 Unique Albums", f"{total_albums:,}")
    col4.metric("⏰ Total Hours", f"{total_hours:.0f}h")

    # --- Top Artists and Tracks by Year ---
    st.header("🏆 My Tops")

    # Prepare data with year
    df_with_year = df_spotify.copy()
    df_with_year['timestamp_dt'] = pd.to_datetime(df_with_year['timestamp'], errors='coerce')
    df_with_year['year'] = df_with_year['timestamp_dt'].dt.year

    # Filter valid data
    df_valid_year = df_with_year[df_with_year['year'].notna()].copy()

    # Define colors by year
    year_colors = {
        2023: '#FF6B6B',  # Red/Pink
        2024: '#4ECDC4',  # Turquoise
        2025: '#45B7D1',  # Light blue
    }

    # --- Top 10 Artists ---
    st.subheader("🎤 Most Listened to Artists")
    
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
        title='Total Plays by Artist and Year',
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
    
    # For display, combine track and artist
    track_year_counts['display_name'] = track_year_counts['master_metadata_track_name'].str[:30] + '...' + \
                                        ' (' + track_year_counts['master_metadata_album_artist_name'].str[:15] + ')'
    
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
        title='Total Plays by Track and Year',
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

    # Note on available years
    with st.expander("ℹ️ Years Available in Your Data"):
        years_str = ", ".join([str(int(y)) for y in sorted(df_valid_year['year'].unique())])
        st.write(f"Your data covers the years: **{years_str}**")
        
        # Display number of plays per year
        yearly_counts = df_valid_year['year'].value_counts().sort_index()
        for year, count in yearly_counts.items():
            st.write(f"- **{int(year)}**: {count:,} plays")

    # --- Listening evolution over time ---
    st.header("📈 Listening Evolution")

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
        
        # Create chart
        fig_evolution = px.bar(listens_per_month, 
                              x='Month', 
                              y='Number of Plays',
                              title='Number of Plays per Month',
                              color='Number of Plays',
                              color_continuous_scale='Sunset')
        fig_evolution.update_layout(showlegend=False,
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_evolution, use_container_width=True)
    else:
        st.error("No data with valid dates found.")

    # --- Track duration distribution ---
    st.header("⏱️ Track Duration Distribution")
    
    # Convert ms_played to minutes
    df_spotify['duration_minutes'] = df_spotify['ms_played'] / 60000
    
    # Create duration categories
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = ['< 5 min', '5-10 min', '10-15 min', '15-20 min', '> 20 min']
    df_spotify['duration_category'] = pd.cut(df_spotify['duration_minutes'], 
                                              bins=bins, 
                                              labels=labels, 
                                              right=False)
    
    # Count tracks per category
    duration_counts = df_spotify['duration_category'].value_counts().sort_index().reset_index()
    duration_counts.columns = ['Duration Range', 'Number of Tracks']
    
    # Create chart
    fig_duration = px.bar(duration_counts, 
                         x='Duration Range', 
                         y='Number of Tracks',
                         title='Distribution of Played Tracks by Duration',
                         color='Number of Tracks',
                         color_continuous_scale='Sunset',
                         text='Number of Tracks')
    
    # Customize appearance
    fig_duration.update_traces(texttemplate='%{text}', textposition='outside')
    fig_duration.update_layout(showlegend=False,
                              yaxis_title='Number of Tracks',
                              xaxis_title='Duration Ranges')
    
    st.plotly_chart(fig_duration, use_container_width=True)

    # --- Listening hours heatmap ---
    st.header("🕐 Listening Patterns by Hour")

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

    # Display data coverage statistics
    st.subheader("📊 Data Coverage")
    col1, col2, col3 = st.columns(3)
    with col1:
        if len(df_valid_dates) > 0:
            st.metric("Period Covered", 
                      f"{df_valid_dates['timestamp_dt'].min().strftime('%Y-%m-%d')} to {df_valid_dates['timestamp_dt'].max().strftime('%Y-%m-%d')}")
        else:
            st.metric("Period Covered", "No valid dates")
    with col2:
        st.metric("Days with Data", 
                  f"{df_valid_dates['date_only'].nunique()} days")
    with col3:
        # Display least represented day
        if occurrences_dict:
            min_day = min(occurrences_dict.items(), key=lambda x: x[1])
            st.metric("Least Covered Day", 
                      f"{day_names[min_day[0]]}: {min_day[1]} occurrences")
        else:
            st.metric("Least Covered Day", "N/A")

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
            xaxis_title="Day of the Week (number shows data coverage)",
            yaxis_title="Hour of the Day",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(24)),
                ticktext=[f"{h:02d}h" for h in range(24)]
            ),
            annotations=annotations
        )

        st.plotly_chart(fig_hour, use_container_width=True)

    # --- Musical mood evolution ---
    st.header("😊 Musical Mood Evolution")

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
            title="Musical Valence Evolution (0 = sad, 1 = happy)",
            xaxis_title="Month",
            yaxis_title="Average Valence",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Rotate x labels if necessary
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
        
        with col4:
            # Recent trend (last 3 months vs previous 3)
            if len(valence_monthly) >= 6:
                recent_avg = valence_monthly.tail(3)['valence_mean'].mean()
                previous_avg = valence_monthly.iloc[-6:-3]['valence_mean'].mean()
                trend = recent_avg - previous_avg
                st.metric("Recent Trend", 
                          "↑ Happier" if trend > 0 else "↓ More Melancholic",
                          f"{abs(trend):.3f}")
            else:
                st.metric("Recent Trend", "Not Enough Data", "—")
        
        # Secondary chart: Valence distribution
        st.subheader("Valence Distribution")
        
        # Create histogram
        fig_dist = px.histogram(
            df_valence_valid, 
            x='valence',
            nbins=50,
            title="Track Distribution by Valence",
            labels={'valence': 'Valence', 'count': 'Number of Tracks'},
            color_discrete_sequence=['rgb(68, 138, 255)']
        )
        
        # Add colored zones for interpretation
        fig_dist.add_vrect(x0=0, x1=0.3, fillcolor="red", opacity=0.1, 
                           annotation_text="Melancholic", annotation_position="top")
        fig_dist.add_vrect(x0=0.3, x1=0.7, fillcolor="yellow", opacity=0.1,
                           annotation_text="Neutral", annotation_position="top")
        fig_dist.add_vrect(x0=0.7, x1=1, fillcolor="green", opacity=0.1,
                           annotation_text="Happy", annotation_position="top")
        
        fig_dist.update_layout(
            xaxis=dict(range=[0, 1]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
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