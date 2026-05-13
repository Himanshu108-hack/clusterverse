import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering Explained 🔥",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1rem;
    backdrop-filter: blur(12px);
}
div[data-testid="metric-container"] label { color: #a78bfa !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.8rem !important; font-weight: 700; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #34d399 !important; }

/* Headers */
h1, h2, h3, h4 { color: #f8fafc !important; }

/* All text */
p, li, label, .stMarkdown { color: #cbd5e1 !important; }

/* Selectbox, slider labels */
.stSelectbox label, .stSlider label, .stRadio label { color: #a78bfa !important; font-weight: 600; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.5);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
}

/* Expander */
details { background: rgba(255,255,255,0.04); border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
summary { color: #a78bfa !important; font-weight: 600; padding: 0.6rem; }

/* Info/success/warning boxes */
.stAlert { border-radius: 12px; border: none; backdrop-filter: blur(8px); }

/* Input boxes */
.stNumberInput input, .stTextInput input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #f8fafc !important;
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }

/* Vibe card */
.vibe-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
.tag {
    display: inline-block;
    background: rgba(124,58,237,0.3);
    color: #c4b5fd;
    border: 1px solid rgba(124,58,237,0.5);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
.gradient-text {
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Space Grotesk", color="#cbd5e1"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(t=40, b=40, l=40, r=40),
)
CLUSTER_COLORS = ["#a78bfa", "#60a5fa", "#34d399", "#fb923c", "#f472b6", "#facc15", "#4ade80", "#38bdf8"]

# ── Dataset factory ───────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(name, n=400, noise=0.08, seed=42):
    rng = np.random.RandomState(seed)
    if name == "🎵 Spotify Listeners":
        centers = [[-3, -3], [0, 3], [3, -2], [-1, 1]]
        labels = ["Chill Vibes", "Hype Beast", "Indie Soul", "Pop Stan"]
        data = []
        for i, (c, lbl) in enumerate(zip(centers, labels)):
            n_pts = n // 4
            pts = rng.randn(n_pts, 2) * 0.9 + c
            df_tmp = pd.DataFrame(pts, columns=["Energy", "Danceability"])
            df_tmp["True_Group"] = lbl
            data.append(df_tmp)
        df = pd.concat(data, ignore_index=True)
        df["Followers_K"] = (df["Energy"] * 15 + rng.randn(len(df)) * 5 + 50).clip(1)
        df["Streams_M"] = (df["Danceability"] * 8 + rng.randn(len(df)) * 3 + 20).clip(1)
        return df, ["Energy", "Danceability"], "Spotify Listeners"

    elif name == "🛍️ Shopping Habits":
        centers = [[-2, 3], [3, 3], [-3, -2], [2, -3]]
        labels = ["Luxury Shopper", "Deal Hunter", "Impulse Buyer", "Brand Loyalist"]
        data = []
        for c, lbl in zip(centers, labels):
            pts = rng.randn(n // 4, 2) * 1.0 + c
            df_tmp = pd.DataFrame(pts, columns=["Avg_Spend", "Visit_Frequency"])
            df_tmp["True_Group"] = lbl
            data.append(df_tmp)
        df = pd.concat(data, ignore_index=True)
        df["Cart_Size"] = (df["Avg_Spend"] * 3 + rng.randn(len(df)) * 4 + 20).clip(1)
        df["Discount_Sensitivity"] = (df["Visit_Frequency"] * (-2) + rng.randn(len(df)) * 3 + 15).clip(1)
        return df, ["Avg_Spend", "Visit_Frequency"], "Shopping Habits"

    elif name == "🏋️ Gym Members":
        centers = [[-2, 2], [3, 1], [0, -3], [-3, -1]]
        labels = ["Cardio King", "Powerlifter", "Weekend Warrior", "Newbie"]
        data = []
        for c, lbl in zip(centers, labels):
            pts = rng.randn(n // 4, 2) * 0.85 + c
            df_tmp = pd.DataFrame(pts, columns=["Workout_Hours", "Protein_Intake"])
            df_tmp["True_Group"] = lbl
            data.append(df_tmp)
        df = pd.concat(data, ignore_index=True)
        df["Weight_Lost_kg"] = (df["Workout_Hours"] * 2 + rng.randn(len(df)) * 2 + 5).clip(0)
        df["Sessions_per_Week"] = (df["Protein_Intake"] * 1.5 + rng.randn(len(df)) * 1.5 + 4).clip(1)
        return df, ["Workout_Hours", "Protein_Intake"], "Gym Members"

    elif name == "🎮 Gamers":
        centers = [[-3, 2], [2, 3], [-1, -3], [3, -2]]
        labels = ["Casual Gamer", "Hardcore Grinder", "Story Explorer", "Competitive Pro"]
        data = []
        for c, lbl in zip(centers, labels):
            pts = rng.randn(n // 4, 2) * 0.9 + c
            df_tmp = pd.DataFrame(pts, columns=["Hours_Played", "Win_Rate"])
            df_tmp["True_Group"] = lbl
            data.append(df_tmp)
        df = pd.concat(data, ignore_index=True)
        df["Games_Owned"] = (df["Hours_Played"] * 5 + rng.randn(len(df)) * 8 + 30).clip(1)
        df["Spend_USD"] = (df["Win_Rate"] * 4 + rng.randn(len(df)) * 10 + 50).clip(0)
        return df, ["Hours_Played", "Win_Rate"], "Gamers"

    else:  # Social Media
        centers = [[-2, 2], [3, 2], [-3, -2], [1, -3]]
        labels = ["Influencer", "Lurker", "Content Creator", "Troll"]
        data = []
        for c, lbl in zip(centers, labels):
            pts = rng.randn(n // 4, 2) * 0.95 + c
            df_tmp = pd.DataFrame(pts, columns=["Posts_per_Day", "Engagement_Rate"])
            df_tmp["True_Group"] = lbl
            data.append(df_tmp)
        df = pd.concat(data, ignore_index=True)
        df["Followers_K"] = (df["Posts_per_Day"] * 10 + rng.randn(len(df)) * 15 + 50).clip(1)
        df["Likes_per_Post"] = (df["Engagement_Rate"] * 8 + rng.randn(len(df)) * 5 + 30).clip(1)
        return df, ["Posts_per_Day", "Engagement_Rate"], "Social Media Users"


def run_clustering(df, features, algo, k, eps, min_samples):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        centroids_scaled = model.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        centroids = None
    else:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        centroids = None

    sil = None
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_found >= 2 and len(set(labels)) > 1:
        try:
            sil = silhouette_score(X_scaled, labels)
        except:
            pass

    noise_count = int(np.sum(labels == -1))
    return labels, centroids, sil, n_clusters_found, noise_count


def elbow_data(df, features, max_k=10):
    X = StandardScaler().fit_transform(df[features].values)
    wcss, sil_scores = [], []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, lbl))
    return list(K_range), wcss, sil_scores


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <div style='font-size:2.5rem;'>🧠</div>
      <div style='font-size:1.3rem; font-weight:700; background: linear-gradient(135deg,#a78bfa,#60a5fa);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>ML Clustering</div>
      <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>No cap, just vibes & math ✨</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 Pick Your Dataset")
    dataset_name = st.selectbox(
        "Dataset",
        ["🎵 Spotify Listeners", "🛍️ Shopping Habits", "🏋️ Gym Members", "🎮 Gamers", "📱 Social Media Users"],
        label_visibility="collapsed"
    )
    n_points = st.slider("📊 Data Points", 200, 800, 400, 50)
    noise_level = st.slider("🌪️ Data Noise", 0.0, 0.5, 0.1, 0.05)

    st.markdown("---")
    st.markdown("#### 🤖 Algorithm")
    algo = st.radio("", ["K-Means", "DBSCAN", "Hierarchical"], label_visibility="collapsed")

    st.markdown("---")
    if algo in ["K-Means", "Hierarchical"]:
        st.markdown("#### ⚙️ Parameters")
        k = st.slider("Number of Clusters (K)", 2, 8, 4)
        eps, min_samples = 0.5, 5
    else:
        st.markdown("#### ⚙️ DBSCAN Parameters")
        eps = st.slider("Epsilon (ε) — Search Radius", 0.1, 2.0, 0.5, 0.05)
        min_samples = st.slider("MinPts — Min Neighbors", 2, 20, 5)
        k = 4

    st.markdown("---")
    run_btn = st.button("🚀 Run Clustering!", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.3);
         border-radius:12px; padding:1rem; font-size:0.8rem; color:#c4b5fd;'>
    <b>💡 Quick Tip</b><br>
    Try the <b>Elbow Method</b> tab to find the perfect K before clustering!
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
df_raw, features, ds_title = generate_dataset(dataset_name, n=n_points, noise=noise_level)
labels_result, centroids, sil_score, n_clusters_found, noise_pts = run_clustering(
    df_raw, features, algo, k, eps, min_samples
)
df_raw["Cluster"] = labels_result.astype(str)
df_raw["Cluster_Name"] = df_raw["Cluster"].map(
    lambda x: f"Cluster {int(x)+1}" if x != "-1" else "🚨 Noise"
)

X_scaled = StandardScaler().fit_transform(df_raw[features].values)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center; padding: 2rem 0 1rem;'>
  <div style='font-size:0.85rem; color:#7c3aed; font-weight:600; letter-spacing:0.15em; text-transform:uppercase;
       background:rgba(124,58,237,0.15); display:inline-block; padding:4px 16px; border-radius:20px;
       border:1px solid rgba(124,58,237,0.4); margin-bottom:12px;'>
    Machine Learning · Unsupervised Learning
  </div>
  <h1 style='font-size:2.8rem; margin:0; font-weight:700;'>
    <span style='background:linear-gradient(135deg,#a78bfa,#60a5fa,#34d399);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    Clustering, Explained fr fr 🔥
    </span>
  </h1>
  <p style='font-size:1.05rem; color:#94a3b8; margin-top:8px;'>
    Dataset: <b style='color:#a78bfa;'>{ds_title}</b> &nbsp;·&nbsp; Algorithm: <b style='color:#60a5fa;'>{algo}</b>
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("📦 Total Points", f"{len(df_raw):,}")
with c2:
    st.metric("🎯 Clusters Found", n_clusters_found)
with c3:
    st.metric("📐 Silhouette Score", f"{sil_score:.3f}" if sil_score else "N/A",
              delta="Great!" if sil_score and sil_score > 0.5 else ("Ok" if sil_score and sil_score > 0.3 else None))
with c4:
    st.metric("🚨 Noise Points", noise_pts if algo == "DBSCAN" else "N/A")
with c5:
    st.metric("🔧 Algorithm", algo)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WHAT IS CLUSTERING — Gen Z explainer
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("🤔 Wait... what even IS clustering? (Click me, bestie)"):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
        <div class='vibe-card'>
        <h3 style='color:#a78bfa; margin-top:0;'>🧠 The Simple Version</h3>
        <p>Imagine you dump all your <b>Spotify playlists</b> into one big pile.
        Clustering is the algorithm that automatically separates the <i>chill lo-fi</i>
        from the <i>gym bangers</i> from the <i>sad girl autumn</i> songs — <b>without
        you telling it anything first.</b></p>
        <p>No labels. No teacher. Just pure math finding hidden groups. That's
        <b>unsupervised learning</b>, no cap.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='vibe-card'>
        <h3 style='color:#60a5fa; margin-top:0;'>🌍 Real-Life Use Cases</h3>
        <span class='tag'>🎵 Spotify Recommendations</span>
        <span class='tag'>🛒 Amazon Segments</span>
        <span class='tag'>🏥 Cancer Detection</span>
        <span class='tag'>🔐 Fraud Detection</span>
        <span class='tag'>📰 Google News Groups</span>
        <span class='tag'>🌆 City Planning</span>
        <br><br>
        <p><b style='color:#34d399;'>Key idea:</b> Points in the same cluster = similar to each other.
        Points in different clusters = very different. Simple math, crazy powerful.</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎨 Cluster Viz", "📈 Elbow Method", "🧬 Dendrogram", "📊 Deep Stats", "📖 Algorithm Guide"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLUSTER VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown("#### 🗺️ Cluster Map — See the Groups!")
        fig = px.scatter(
            df_raw, x=features[0], y=features[1],
            color="Cluster_Name",
            hover_data={c: True for c in df_raw.columns if c not in ["Cluster", "True_Group"]},
            color_discrete_sequence=CLUSTER_COLORS,
            size_max=10,
            opacity=0.85,
        )
        # Add centroids for K-Means
        if centroids is not None:
            for i, c in enumerate(centroids):
                fig.add_trace(go.Scatter(
                    x=[c[0]], y=[c[1]],
                    mode="markers+text",
                    marker=dict(symbol="star", size=22, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                                line=dict(color="white", width=2)),
                    text=[f"μ{i+1}"],
                    textposition="top center",
                    textfont=dict(color="white", size=12),
                    showlegend=False, name=f"Centroid {i+1}"
                ))
        fig.update_traces(marker=dict(size=7))
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
                          xaxis_title=features[0], yaxis_title=features[1])
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("#### 🍩 Cluster Split")
        cluster_counts = df_raw["Cluster_Name"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_pie = px.pie(cluster_counts, values="Count", names="Cluster",
                         color_discrete_sequence=CLUSTER_COLORS, hole=0.55)
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        fig_pie.update_layout(height=280,
                               showlegend=True,
                               legend=dict(orientation="v", yanchor="middle", y=0.5))
        fig_pie.update_traces(textfont_size=11)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📋 Cluster Sizes")
        for _, row in cluster_counts.iterrows():
            pct = row["Count"] / len(df_raw) * 100
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;font-size:0.82rem;color:#e2e8f0;margin-bottom:3px;'>
                <span>{row["Cluster"]}</span><span>{row["Count"]} ({pct:.0f}%)</span>
              </div>
              <div style='background:rgba(255,255,255,0.08);border-radius:6px;height:8px;overflow:hidden;'>
                <div style='background:linear-gradient(90deg,#7c3aed,#60a5fa);height:8px;
                     width:{pct}%;border-radius:6px;transition:width 0.5s;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔮 Feature Relationships by Cluster")
    extra_cols = [c for c in df_raw.columns if c not in ["Cluster", "Cluster_Name", "True_Group"] + features]
    if extra_cols:
        col_x, col_y = st.columns(2)
        with col_x:
            feat_x = st.selectbox("X-axis feature", features + extra_cols, key="fx")
        with col_y:
            feat_y = st.selectbox("Y-axis feature", [f for f in features + extra_cols if f != feat_x],
                                  index=1, key="fy")
        try:
            fig2 = px.scatter(df_raw, x=feat_x, y=feat_y, color="Cluster_Name",
                              marginal_x="histogram", marginal_y="box",
                              color_discrete_sequence=CLUSTER_COLORS, opacity=0.75, trendline="lowess")
        except Exception:
            fig2 = px.scatter(df_raw, x=feat_x, y=feat_y, color="Cluster_Name",
                              marginal_x="histogram", marginal_y="box",
                              color_discrete_sequence=CLUSTER_COLORS, opacity=0.75)
            st.warning("Plotly LOWESS trendline requires the optional dependency `statsmodels`. Showing the scatter plot without smoothing.")
        fig2.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ELBOW METHOD
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📈 Find the Perfect K — Elbow & Silhouette")
    st.markdown("""
    <div class='vibe-card'>
    <b style='color:#a78bfa;'>🤔 The vibe:</b> More clusters = lower error, but at some point you're just
    over-fitting. The <b>elbow</b> is where adding more clusters stops being worth it.
    The <b>silhouette score</b> tells you how well-separated your clusters actually are (higher = better, max = 1.0).
    </div>
    """, unsafe_allow_html=True)

    max_k_elbow = st.slider("Max K to test", 3, 12, 8, key="elbow_k")
    K_vals, wcss_vals, sil_vals = elbow_data(df_raw, features, max_k_elbow)

    fig_elbow = make_subplots(rows=1, cols=2,
                               subplot_titles=["WCSS (Lower = tighter clusters)",
                                               "Silhouette Score (Higher = better)"])
    fig_elbow.add_trace(go.Scatter(x=K_vals, y=wcss_vals, mode="lines+markers",
                                    line=dict(color="#a78bfa", width=2.5),
                                    marker=dict(size=9, color="#a78bfa",
                                                line=dict(color="white", width=1.5)),
                                    name="WCSS"), row=1, col=1)
    best_k_sil = K_vals[np.argmax(sil_vals)]
    fig_elbow.add_trace(go.Scatter(x=K_vals, y=sil_vals, mode="lines+markers",
                                    line=dict(color="#34d399", width=2.5),
                                    marker=dict(size=9, color=["#f472b6" if kv == best_k_sil else "#34d399"
                                                                for kv in K_vals],
                                                line=dict(color="white", width=1.5)),
                                    name="Silhouette"), row=1, col=2)
    fig_elbow.add_vline(x=best_k_sil, line_dash="dot", line_color="#f472b6",
                         annotation_text=f"Best K={best_k_sil}", row=1, col=2)
    fig_elbow.update_layout(**PLOTLY_LAYOUT, height=380,
                              showlegend=False)
    for ax in ["xaxis", "xaxis2"]:
        fig_elbow.layout[ax].update(gridcolor="rgba(255,255,255,0.06)")
    for ax in ["yaxis", "yaxis2"]:
        fig_elbow.layout[ax].update(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.success(f"✅ Silhouette score suggests **K = {best_k_sil}** is the optimal number of clusters for this dataset!")

    # Show per-K table
    with st.expander("📋 Full K vs Score Table"):
        df_elbow = pd.DataFrame({"K": K_vals, "WCSS": [round(w, 2) for w in wcss_vals],
                                  "Silhouette Score": [round(s, 4) for s in sil_vals]})
        df_elbow["Best K?"] = df_elbow["K"].apply(lambda x: "⭐ Yes!" if x == best_k_sil else "")
        st.dataframe(df_elbow, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — DENDROGRAM (Hierarchical)
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🧬 Dendrogram — Hierarchical Clustering Tree")
    st.markdown("""
    <div class='vibe-card'>
    <b style='color:#60a5fa;'>📖 What's this?</b>
    A dendrogram is like a <b>family tree for your data</b>. It shows how clusters
    merge step by step. Cut the tree at any height to get your desired number of groups.
    The <b>taller the bar</b>, the more different those two clusters are when they merge.
    </div>
    """, unsafe_allow_html=True)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        link_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"],
                                    format_func=lambda x: {"ward": "Ward (minimize variance)",
                                                            "complete": "Complete (max distance)",
                                                            "average": "Average linkage",
                                                            "single": "Single (min distance)"}[x])
    with col_d2:
        n_dend = st.slider("Sample size for dendrogram", 30, 150, 80)

    sample_idx = np.random.choice(len(df_raw), min(n_dend, len(df_raw)), replace=False)
    X_dend = StandardScaler().fit_transform(df_raw[features].values[sample_idx])
    Z = linkage(X_dend, method=link_method)

    fig_dend, ax = plt.subplots(figsize=(14, 5))
    fig_dend.patch.set_alpha(0)
    ax.set_facecolor("none")
    dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="#64748b",
               leaf_rotation=90, leaf_font_size=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_ylabel("Distance", color="#94a3b8", fontsize=10)
    ax.set_title(f"Hierarchical Clustering Dendrogram ({link_method} linkage)", color="#e2e8f0", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_dend)
    plt.close()

    with st.expander("📚 Linkage Methods Explained"):
        st.markdown("""
        | Method | How it measures distance | Best for |
        |--------|--------------------------|----------|
        | **Ward** | Minimizes variance increase when merging | Compact, similar-sized clusters |
        | **Complete** | Uses the MAX distance between any two points | Well-separated clusters |
        | **Average** | Uses the MEAN distance between all pairs | General use |
        | **Single** | Uses the MIN distance (closest points) | Long, chain-like clusters |
        """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — DEEP STATS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 📊 Deep Dive Stats")

    real_clusters = [c for c in df_raw["Cluster"].unique() if c != "-1"]
    stats_rows = []
    for c in real_clusters:
        mask = df_raw["Cluster"] == c
        row = {"Cluster": f"Cluster {int(c)+1}", "Count": int(mask.sum())}
        for f in features:
            row[f"Mean {f}"] = round(df_raw.loc[mask, f].mean(), 3)
            row[f"Std {f}"] = round(df_raw.loc[mask, f].std(), 3)
        stats_rows.append(row)
    df_stats = pd.DataFrame(stats_rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🌡️ Feature Distribution by Cluster")
    feat_sel = st.selectbox("Select feature", features, key="dist_feat")
    fig_box = px.violin(df_raw[df_raw["Cluster"] != "-1"], x="Cluster_Name", y=feat_sel,
                         color="Cluster_Name", box=True, points="outliers",
                         color_discrete_sequence=CLUSTER_COLORS)
    fig_box.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    col_pca1, col_pca2 = st.columns(2)
    with col_pca1:
        st.markdown("#### 🔬 PCA 2D Projection")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca["Cluster_Name"] = df_raw["Cluster_Name"].values
        fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster_Name",
                              color_discrete_sequence=CLUSTER_COLORS, opacity=0.8)
        fig_pca.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig_pca, use_container_width=True)
        ev = pca.explained_variance_ratio_
        st.caption(f"PCA explains {ev[0]*100:.1f}% + {ev[1]*100:.1f}% = {sum(ev)*100:.1f}% of total variance")

    with col_pca2:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        num_cols = [c for c in df_raw.columns if df_raw[c].dtype in [np.float64, np.int64]
                    and c not in ["Cluster"]]
        corr = df_raw[num_cols].corr()
        fig_heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              aspect="auto", text_auto=".2f")
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=340,
                                coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📁 Raw Clustered Data")
    st.dataframe(df_raw.drop(columns=["Cluster"]).rename(columns={"Cluster_Name": "Assigned Cluster"}),
                 use_container_width=True, height=260)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ALGORITHM GUIDE
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### 📖 Which Algorithm Should You Use? (fr fr)")

    algo_data = [
        {
            "name": "K-Means", "emoji": "🎯",
            "vibe": "The classic. Fast, simple, works for most stuff.",
            "when": "When your clusters are roughly round and similar sized.",
            "real": "Spotify user segments, customer groups at H&M",
            "pros": "Super fast · Easy to understand · Works on big data",
            "cons": "Need to pick K beforehand · Bad with odd shapes · Sensitive to outliers",
            "math": "Minimizes: J = Σ Σ ||xᵢ - μₖ||²",
            "color": "#7c3aed"
        },
        {
            "name": "DBSCAN", "emoji": "🔍",
            "vibe": "The outlier hunter. Finds weird-shaped clusters.",
            "when": "When you have noise/outliers or non-circular cluster shapes.",
            "real": "Fraud detection, GPS hotspot mapping, anomaly detection",
            "pros": "No K needed · Handles outliers · Works on any shape",
            "cons": "Two params to tune · Struggles with varying density",
            "math": "Core: |N_ε(x)| ≥ MinPts",
            "color": "#0891b2"
        },
        {
            "name": "Hierarchical", "emoji": "🌳",
            "vibe": "The family tree builder. Great for visualizing structure.",
            "when": "When you want to explore data at multiple zoom levels.",
            "real": "Gene expression analysis, document organization",
            "pros": "No K needed · Dendrogram is super interpretable · Any distance metric",
            "cons": "Slow on big data O(n²) · Hard to undo merges",
            "math": "Agglomerative: merge argmin d(Cᵢ, Cⱼ)",
            "color": "#059669"
        },
        {
            "name": "GMM", "emoji": "🫧",
            "vibe": "Soft clustering — points can belong to multiple clusters with probabilities.",
            "when": "When clusters overlap or you need confidence scores.",
            "real": "Medical diagnosis, speech recognition, image segmentation",
            "pros": "Soft assignments · Handles elliptical clusters · Probabilistic",
            "cons": "Assumes Gaussian distribution · Needs good initialization",
            "math": "P(x) = Σ πₖ N(x | μₖ, Σₖ)",
            "color": "#dc2626"
        },
    ]

    for a in algo_data:
        st.markdown(f"""
        <div class='vibe-card' style='border-left: 3px solid {a["color"]};'>
          <div style='display:flex; align-items:center; gap:10px; margin-bottom:10px;'>
            <span style='font-size:1.8rem;'>{a["emoji"]}</span>
            <div>
              <h3 style='color:{a["color"]} !important; margin:0; font-size:1.1rem;'>{a["name"]}</h3>
              <span style='color:#94a3b8; font-size:0.85rem;'>{a["vibe"]}</span>
            </div>
          </div>
          <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; font-size:0.82rem;'>
            <div>
              <b style='color:#a78bfa;'>When to use</b><br>
              <span style='color:#cbd5e1;'>{a["when"]}</span>
            </div>
            <div>
              <b style='color:#34d399;'>Real-life use</b><br>
              <span style='color:#cbd5e1;'>{a["real"]}</span>
            </div>
            <div>
              <b style='color:#60a5fa;'>Math key</b><br>
              <code style='color:#f472b6; font-size:0.78rem; background:rgba(255,255,255,0.05);
                   padding:3px 8px; border-radius:6px;'>{a["math"]}</code>
            </div>
          </div>
          <div style='margin-top:10px; font-size:0.8rem;'>
            <span style='color:#34d399;'>✅ {a["pros"]}</span> &nbsp;|&nbsp;
            <span style='color:#fb923c;'>⚠️ {a["cons"]}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='vibe-card' style='border:1px solid rgba(167,139,250,0.4);'>
    <h3 style='color:#a78bfa; margin-top:0;'>🧠 Key Formulas — Print These Out bestie</h3>

    | Concept | Formula | What it means |
    |---------|---------|---------------|
    | Euclidean Distance | `d = √(Σ(xᵢ - yᵢ)²)` | How far apart two points are |
    | K-Means Centroid | `μₖ = (1/n) Σ xᵢ` | Average position of a cluster |
    | WCSS | `J = Σₖ Σᵢ ∥xᵢ - μₖ∥²` | Total "tightness" of clusters |
    | Silhouette | `s = (b - a) / max(a,b)` | How well a point fits its cluster |
    | Silhouette range | `-1 to +1` | +1 = perfect, 0 = meh, -1 = wrong cluster |
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:1.5rem 0; color:#475569; font-size:0.82rem;'>
  Built with 💜 using Streamlit · Plotly · scikit-learn<br>
  <span style='color:#7c3aed;'>Clustering Dashboard</span> — No cap, just clusters ✨
</div>
""", unsafe_allow_html=True)